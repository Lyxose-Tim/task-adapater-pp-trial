"""创新点 3 · 阶段 1：顺序敏感性诊断（3a）。

手册 §2.3：由 run.py 的 test() 派生。每个 episode 内视觉前向做**一次**并复用，
文本侧对 6 个子动作排列各算一次语义分（配对、同一批 episode）。条件矩阵：

  C0 正序文本 / 正常帧        —— 基准（=B0 评测口径）
  C1 反转文本 (2,1,0) / 正常帧 —— 最强顺序破坏
  C2 全部 5 个非恒等排列 / 正常帧 —— 报逐排列与均值
  C3 正序文本 / 帧反转         —— 视觉侧对照
  C4 正序文本 / 帧随机打乱     —— 每 episode 固定随机排列（记录种子）

指标（手册 §2.2）：各条件 Acc(融合/仅语义/仅视觉)；OS = Acc(C0) − mean Acc(C2)，
配对 95% CI；内置 sanity：文本扰动（C1/C2）不得改变仅视觉分（共用同一视觉前向，
构造上保证）。O-2：帧扰动对全部 query 同步施加。

用法（仓库根目录，加载 B0 权重）::

    TA_CONFIG=config_3a_dev.yaml \
    CLIP_VIT_B16_PATH=dataset/checkpoints/ViT-B-16.pt \
    python diagnose.py 2>&1 | tee diagnose_3a.log

`config` 需 `test_model: True` 与 `checkpoint: <B0 seed916 ckpt 绝对/相对路径>`；
`diagnose_episodes`（缺省 test_episode）与 `frame_perm_seed`（缺省 916）可选。
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from utils import read_yaml
from dataset import SetDataManager, truncation_repeat_fraction
from models import TaskAdapter
from fsar.order import stage_permutations, make_frame_permutation
from fsar.diagnostics import summarize_mean_ci95, ConditionAccumulator


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _perturb_query_frames(x, n_support, permutation):
    """对 episode 张量 [n_way, sq, t, c, h, w] 的 query 帧（sq 后段）同步重排时间维。

    support 部分不动；O-2：全部 query 用同一帧序。permutation 为长度 t 的下标序。
    """
    index = torch.as_tensor(permutation, dtype=torch.long, device=x.device)
    out = x.clone()
    out[:, n_support:] = x[:, n_support:].index_select(2, index)
    return out


def _accuracy(scores, y_query):
    # scores [N_Q, n_way]，y_query 为按类序的真类下标
    pred = scores.detach().argmax(dim=1).cpu().numpy()
    return float(np.mean(pred == y_query))


@torch.no_grad()
def diagnose(test_loader, model, params):
    model.eval()

    text_perms = list(stage_permutations(3))          # 6 个，索引 0 = 正序 (0,1,2)
    identity = text_perms[0]
    reverse = tuple(reversed(identity))               # C1 = (2,1,0)
    c2_perms = [p for p in text_perms if p != identity]   # 5 个非恒等 = C2
    n_way, n_support, n_query = model.n_way, model.n_support, model.n_query
    y_query = np.repeat(range(n_way), n_query)
    frame_seed = int(getattr(params, "frame_perm_seed", 916))

    acc = ConditionAccumulator()                      # 融合分：逐 episode 逐条件
    sem_acc = ConditionAccumulator()                  # 仅语义分
    vis_acc = ConditionAccumulator()                  # 仅视觉分

    is_ot = getattr(model, "align_mode", "window") == "ot"
    sem_maxabs = []                                   # 逐 episode：max|sem(C0)-sem(C2 各排列)|（U1 逐元素）
    ot_stats_accum = {}                               # OT 插桩：各统计量的逐 (q,c) 值汇集（align_mode=ot）

    iter_num = len(test_loader)
    started = time.time()
    for episode_index, (x, label) in enumerate(test_loader):
        x = x.cuda()                                  # [n_way, sq, t, c, h, w]
        label_idx = list(label[:, 0].numpy())
        _, sq, t, c, h, w = x.shape

        def flat(xx):
            return xx.reshape(n_way * sq * t, c, h, w)

        # ---- 正常帧：一次视觉前向，供 C0/C1/C2 复用 ----
        z_query, z_proto, q_aft_tm = model.episode_visual(flat(x))
        vis_normal = model.visual_scores(z_query, z_proto)   # [N_Q, n_way]，文本无关

        episode_fused, episode_sem, episode_vis, episode_sem_raw = {}, {}, {}, {}

        def score_text(cond, q_aft, vis, perm):
            sem = model.semantic_scores(q_aft, label_idx, perm, z_query=z_query)
            episode_sem_raw[cond] = sem.detach()
            episode_fused[cond] = _accuracy(vis * sem, y_query)
            episode_sem[cond] = _accuracy(sem, y_query)
            episode_vis[cond] = _accuracy(vis, y_query)

        score_text("C0", q_aft_tm, vis_normal, identity)
        score_text("C1", q_aft_tm, vis_normal, reverse)
        c2_fused = []
        for k, perm in enumerate(c2_perms):
            name = f"C2p{k}"
            score_text(name, q_aft_tm, vis_normal, perm)
            c2_fused.append(episode_fused[name])
        episode_fused["C2mean"] = float(np.mean(c2_fused))

        # U1 逐元素：正序 vs 各 C2 排列的 sem 最大绝对差（平衡 OT/λ=0 应≈0，比 OS=0 更严）
        sem_maxabs.append(max(
            float((episode_sem_raw["C0"] - episode_sem_raw[f"C2p{k}"]).abs().max())
            for k in range(len(c2_perms))))

        # OT 数值健康插桩（仅 align_mode=ot；detach，不改分数/梯度）
        if is_ot:
            stats = model.ot_diagnostics(q_aft_tm, label_idx, z_query)
            for key, val in stats.items():
                ot_stats_accum.setdefault(key, []).extend(val.detach().flatten().cpu().tolist())

        # ---- 帧反转（C3）与帧随机打乱（C4）：各自重做一次视觉前向，文本正序 ----
        rev_perm = make_frame_permutation(t, mode="reverse")
        zq_r, zp_r, qa_r = model.episode_visual(flat(_perturb_query_frames(x, n_support, rev_perm)))
        score_text("C3", qa_r, model.visual_scores(zq_r, zp_r), identity)

        shuf_perm = make_frame_permutation(t, mode="random", seed=frame_seed + episode_index)
        zq_s, zp_s, qa_s = model.episode_visual(flat(_perturb_query_frames(x, n_support, shuf_perm)))
        score_text("C4", qa_s, model.visual_scores(zq_s, zp_s), identity)

        acc.add_episode(episode_fused)
        sem_acc.add_episode(episode_sem)
        vis_acc.add_episode(episode_vis)

        if (episode_index + 1) % 200 == 0:
            os_now = acc.order_sensitivity("C0", ["C2mean"])
            print(f"[{episode_index+1}/{iter_num}] C0={summarize_mean_ci95(acc.values('C0')).mean*100:.2f} "
                  f"OS={os_now.mean*100:.2f}±{os_now.ci95*100:.2f}", flush=True)

    # ---- 汇总 ----
    def pct(mci):
        return {"mean": mci.mean * 100, "ci95": mci.ci95 * 100}

    conditions = ["C0", "C1", "C3", "C4"] + [f"C2p{k}" for k in range(len(c2_perms))]
    report = {
        "episodes": iter_num,
        "wall_time_s": time.time() - started,
        "frame_perm_seed": frame_seed,
        "fused": {cond: pct(summarize_mean_ci95(acc.values(cond))) for cond in conditions},
        "fused_C2mean": pct(summarize_mean_ci95(acc.values("C2mean"))),
        "semantic": {cond: pct(summarize_mean_ci95(sem_acc.values(cond))) for cond in ["C0", "C1"] + [f"C2p{k}" for k in range(len(c2_perms))]},
        "visual": {cond: pct(summarize_mean_ci95(vis_acc.values(cond))) for cond in ["C0", "C1", "C3", "C4"] + [f"C2p{k}" for k in range(len(c2_perms))]},
        # 顺序敏感度：配对 (C0 − C2mean)、以及 (C0 − C1)
        "OS_C0_minus_C2mean": pct(acc.order_sensitivity("C0", ["C2mean"])),
        "delta_C0_minus_C1": pct(acc.paired_difference("C0", "C1")),
        "text_perms": {"identity": list(identity), "reverse": list(reverse), "C2": [list(p) for p in c2_perms]},
        # U1 逐元素（比 OS=0 严）：正序 vs C2 各排列 sem 的逐 episode 最大绝对差
        "u1_sem_max_abs_diff_C0_vs_C2": {
            "max": float(np.max(sem_maxabs)), "mean": float(np.mean(sem_maxabs)),
            "p99": float(np.percentile(sem_maxabs, 99)),
        },
    }

    # 逐 episode C0 融合 acc（供跨 checkpoint 配对 ΔAcc；同 seed 916 episode 流对齐）
    report["per_episode_c0_fused"] = [float(v) for v in acc.values("C0")]

    # OT 数值健康插桩汇总（align_mode=ot 才有）：mean/p95/p99/max，不逐 (q,c) 打印
    if is_ot and ot_stats_accum:
        def agg(vals):
            arr = np.asarray(vals, dtype=np.float64)
            return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)),
                    "p99": float(np.percentile(arr, 99)), "max": float(arr.max()),
                    "min": float(arr.min())}
        report["ot_plan_stats"] = {key: agg(vals) for key, vals in ot_stats_accum.items()}

    # sanity（手册 §2.2）：文本扰动不得改变仅视觉分 —— C0/C1/C2 的 vis 应逐 episode 相同
    vis_c0 = np.asarray(vis_acc.values("C0"))
    vis_c1 = np.asarray(vis_acc.values("C1"))
    sanity_ok = bool(np.allclose(vis_c0, vis_c1)) and all(
        np.allclose(vis_c0, np.asarray(vis_acc.values(f"C2p{k}"))) for k in range(len(c2_perms))
    )
    report["sanity_text_perturb_leaves_visual_unchanged"] = sanity_ok

    _keys = ["episodes", "fused", "OS_C0_minus_C2mean", "delta_C0_minus_C1",
             "u1_sem_max_abs_diff_C0_vs_C2", "sanity_text_perturb_leaves_visual_unchanged"]
    if "ot_plan_stats" in report:
        _keys.append("ot_plan_stats")
    print(json.dumps({k: report[k] for k in _keys}, ensure_ascii=False, indent=2))
    if not sanity_ok:
        print("WARNING: vis-only accuracy changed under text permutation — code bug (手册 §2.2)", flush=True)
    return report


@torch.no_grad()
def probe(test_loader, model, params):
    """步 B 前置·只推理 λ 探针（内容贡献门控，用户 2026-09-05 决策）。

    固定 ckpt 的 ε=params.ot_eps、ρ、iters，对 λ∈params.ot_probe_lams 只前向不训练、
    在同一批配对 episode 上记录（逐 (q,c)/(q)/(c) 汇总 mean/p95/p99/max/min）：
      row_cond_entropy   逐帧条件分配熵/logK（λ↑ 应↓，锐度）；
      band_mass          最近对角带质量占比（λ↑ 应↑）；
      content_l1_full_vs_pos  ‖π_full−π_pos‖₁（π_pos=OT(λD) 纯几何；→0=软窗口退化）；
      cross_class_l1 / cross_query_l1  plan 随类/query 内容的成对 L1（π_pos 恒 0，>噪声=内容自适应）；
      aux_acc_fused/sem、aux_OS         仅辅助，不据此选正式 λ。
    判据与放行逻辑见交接包⑫。α=ot 时才有意义（align_mode 必须 ot）。
    """
    model.eval()
    n_way, n_support, n_query = model.n_way, model.n_support, model.n_query
    y_query = np.repeat(range(n_way), n_query)
    lam_list = [float(v) for v in getattr(params, "ot_probe_lams", [0.0, 0.1, 0.3, 1.0, 3.0])]
    c2_perms = [p for p in stage_permutations(3) if p != (0, 1, 2)]    # 5 个非恒等 = C2（辅助 OS）

    stat_keys = ["row_cond_entropy", "band_mass", "content_l1_full_vs_pos",
                 "cross_class_l1", "cross_query_l1"]
    accum = {lam: {k: [] for k in stat_keys} for lam in lam_list}
    acc_fused = {lam: [] for lam in lam_list}
    acc_sem = {lam: [] for lam in lam_list}
    os_paired = {lam: [] for lam in lam_list}                         # 逐 episode (C0−C2mean) 融合

    iter_num = len(test_loader)
    started = time.time()
    for episode_index, (x, label) in enumerate(test_loader):
        x = x.cuda()
        label_idx = list(label[:, 0].numpy())
        _, sq, t, c, h, w = x.shape
        z_query, z_proto, q_aft_tm = model.episode_visual(x.reshape(n_way * sq * t, c, h, w))
        vis = model.visual_scores(z_query, z_proto)                   # [N_Q, n_way]
        per_lam = model.ot_probe(q_aft_tm, label_idx, lam_list, z_query=z_query, c2_perms=c2_perms)
        for lam, d in per_lam.items():
            for k in stat_keys:
                accum[lam][k].extend(d[k].detach().flatten().cpu().tolist())
            S = d["score"]
            fused_c0 = _accuracy(vis * S, y_query)
            acc_fused[lam].append(fused_c0)
            acc_sem[lam].append(_accuracy(S, y_query))
            c2_fused = [_accuracy(vis * d["score_c2"][j], y_query) for j in range(len(c2_perms))]
            os_paired[lam].append(fused_c0 - float(np.mean(c2_fused)))
        if (episode_index + 1) % 200 == 0:
            msg = " ".join(
                f"λ{lam:g}:H={np.mean(accum[lam]['row_cond_entropy']):.3f}"
                f"/band={np.mean(accum[lam]['band_mass']):.3f}"
                f"/cL1={np.mean(accum[lam]['content_l1_full_vs_pos']):.3f}"
                for lam in lam_list)
            print(f"[probe {episode_index+1}/{iter_num}] {msg}", flush=True)

    def agg(vals):
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)), "max": float(arr.max()), "min": float(arr.min())}

    def acc_ci(vals):
        m = summarize_mean_ci95(vals)
        return {"mean": m.mean * 100, "ci95": m.ci95 * 100}

    report = {
        "mode": "ot_probe",
        "episodes": iter_num,
        "wall_time_s": time.time() - started,
        "ot_eps": float(model.ot_eps),
        "ot_rho": (None if model.ot_rho is None else float(model.ot_rho)),
        "ot_iters": int(model.ot_iters),
        "ot_weight": getattr(model, "ot_weight", "mass"),
        "frame_source": getattr(model, "frame_source", "ca"),
        "lambdas": lam_list,
        "per_lambda": {
            f"{lam:g}": {
                **{k: agg(accum[lam][k]) for k in stat_keys},
                "aux_acc_fused": acc_ci(acc_fused[lam]),
                "aux_acc_sem": acc_ci(acc_sem[lam]),
                "aux_OS_fused": acc_ci(os_paired[lam]),
            } for lam in lam_list
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


@torch.no_grad()
def robust_eval(model, params, test_file):
    """创新点 1 阶段 D·时序鲁棒截断评测（手册 §四；步 C 每档 ρ 同步跑）。

    同一 episode 集（每窗前 setup_seed 复位→配对），四种评测帧采样窗：
      normal 全片 / 掐头(0.25,1) / 去尾(0,0.75) / 收缩(0.125,0.875)。
    指标：各窗标准融合 Acc（=C0 口径 vis*sem 正序）±95%CI，及 ΔAcc(相对 normal，逐 episode 配对)。
    训练路径不改（仅评测 sample_window 生效）。边界重复采样占比一并记录。
    """
    model.eval()
    windows = [("normal", None), ("head", (0.25, 1.0)), ("tail", (0.0, 0.75)), ("shrink", (0.125, 0.875))]
    n_way, n_support, n_query = model.n_way, model.n_support, model.n_query
    y_query = np.repeat(range(n_way), n_query)
    seed = int(getattr(params, "seed", 916))
    episodes = int(getattr(params, "diagnose_episodes", params.test_episode))
    num_workers = getattr(params, "num_workers", 8)
    is_ot = getattr(model, "align_mode", "window") == "ot"
    video_list = [x.strip().split(" ") for x in open(test_file)]

    per_ep, stage_relax = {}, {}                              # 各窗：逐 episode 融合 acc / 阶段质量放松 col_residual
    for name, win in windows:
        setup_seed(seed)                                      # 复位→四窗同 episode 流（配对）
        dm = SetDataManager(224, n_query=n_query, num_segments=params.num_segments,
                            n_eposide=episodes, n_way=params.test_n_way,
                            n_support=params.n_shot, num_workers=num_workers)
        loader = dm.get_data_loader(test_file, aug=False, sample_window=win)
        accs, relax = [], []
        for x, label in loader:
            x = x.cuda()
            label_idx = list(label[:, 0].numpy())
            _, sq, t, c, h, w = x.shape
            z_query, z_proto, q_aft_tm = model.episode_visual(x.reshape(n_way * sq * t, c, h, w))
            vis = model.visual_scores(z_query, z_proto)
            sem = model.semantic_scores(q_aft_tm, label_idx, None, z_query=z_query)
            accs.append(_accuracy(vis * sem, y_query))
            if is_ot:                                         # 阶段质量放松 ‖stage_mass−1/K‖₁（col_residual）
                st = model.ot_diagnostics(q_aft_tm, label_idx, z_query)
                relax.extend(st["col_residual"].detach().flatten().cpu().tolist())
        per_ep[name] = np.asarray(accs, dtype=np.float64)
        stage_relax[name] = np.asarray(relax, dtype=np.float64) if relax else None
        rf = truncation_repeat_fraction(video_list, params.num_segments, win)
        print(f"[robust {name:>6}] Acc={per_ep[name].mean()*100:.2f} repeat_frac={rf:.3f}"
              + (f" stage_relax={stage_relax[name].mean():.4f}" if is_ot else ""), flush=True)

    base = per_ep["normal"]
    report = {"mode": "robust_eval", "episodes": episodes, "seed": seed,
              "ot": {"align_mode": getattr(model, "align_mode", "window"),
                     "eps": float(getattr(model, "ot_eps", 0.0)),
                     "lam": float(getattr(model, "ot_lam", 0.0)),
                     "rho": (None if getattr(model, "ot_rho", None) is None else float(model.ot_rho))},
              "windows": {}}
    for name, win in windows:
        arr = per_ep[name]
        mean, ci = arr.mean(), 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
        if name == "normal":
            dmean = dci = 0.0
        else:                                                 # 逐 episode 配对差
            d = arr - base
            dmean, dci = d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        entry = {
            "acc": float(mean * 100), "ci95": float(ci * 100),
            "dAcc_mean": float(dmean * 100), "dAcc_ci95": float(dci * 100),
            "repeat_frac": float(truncation_repeat_fraction(video_list, params.num_segments, win)),
            # 逐 episode 融合 acc（供跨配置 DiD_t = ΔAcc(ρ10) − ΔAcc(None) 及配对 CI；四格同 2500 episode 流对齐）
            "per_episode_fused": [float(v) for v in arr]}
        if is_ot and stage_relax[name] is not None:           # 阶段质量放松（验 ρ 真不平衡；平衡档≈0）
            sm = stage_relax[name]
            entry["stage_mass_relax"] = {"mean": float(sm.mean()), "p95": float(np.percentile(sm, 95))}
        report["windows"][name] = entry
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}, ensure_ascii=False))
    for nm in [w[0] for w in windows]:                        # 摘要打印不含逐 episode 长数组
        w = report["windows"][nm]
        extra = f" stage_relax(mean/p95)={w['stage_mass_relax']['mean']:.4f}/{w['stage_mass_relax']['p95']:.4f}" if "stage_mass_relax" in w else ""
        print(f"  {nm:>6}: acc={w['acc']:.2f}±{w['ci95']:.2f} dAcc={w['dAcc_mean']:+.2f}±{w['dAcc_ci95']:.2f}{extra}")
    return report


if __name__ == "__main__":
    params = argparse.Namespace(**read_yaml())
    setup_seed(getattr(params, "seed", 916))

    base_path = params.dataset_base_path
    if params.dataset == "somethingcmn":
        test_file = base_path + "smsm_cmn/annotations/test.txt"
        adapter_depth, text_depth = 6, 2
    else:
        raise ValueError("diagnose.py 目前仅接 somethingcmn（SSv2-Small）")

    n_query = params.n_query
    num_workers = getattr(params, "num_workers", 8)
    episodes = int(getattr(params, "diagnose_episodes", params.test_episode))
    test_datamgr = SetDataManager(224, n_query=n_query, num_segments=params.num_segments,
                                  n_eposide=episodes, n_way=params.test_n_way,
                                  n_support=params.n_shot, num_workers=num_workers)
    test_loader = test_datamgr.get_data_loader(test_file, aug=False)

    model = TaskAdapter(params.train_n_way, params.n_shot, params.n_query,
                        dataset=params.dataset, adapter_depth=adapter_depth, text_depth=text_depth)
    model = model.cuda()
    model.distribute_backbone([i for i in range(params.num_gpus)])

    checkpoint = torch.load(params.checkpoint, map_location="cpu", weights_only=True)["state"]
    model.load_state_dict(checkpoint)

    is_probe = bool(getattr(params, "ot_probe", False))              # 步B前置 λ 探针（只推理）
    is_robust = bool(getattr(params, "robust_eval", False))          # 阶段D 时序鲁棒截断评测
    if is_robust:
        report, sub = robust_eval(model, params, test_file), "robust_eval"
    elif is_probe:
        report, sub = probe(test_loader, model, params), "ot_probe"
    else:
        report, sub = diagnose(test_loader, model, params), "diagnose_3a"

    out_dir = os.path.join(params.work_dir, sub)
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    prefix = {"robust_eval": "robust_eval", "ot_probe": "probe_lambda", "diagnose_3a": "diagnose_3a"}[sub]
    with open(os.path.join(out_dir, f"{prefix}_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
