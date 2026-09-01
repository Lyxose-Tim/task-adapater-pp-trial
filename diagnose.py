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
from dataset import SetDataManager
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

        episode_fused, episode_sem, episode_vis = {}, {}, {}

        def score_text(cond, q_aft, vis, perm):
            sem = model.semantic_scores(q_aft, label_idx, perm)
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
    }

    # sanity（手册 §2.2）：文本扰动不得改变仅视觉分 —— C0/C1/C2 的 vis 应逐 episode 相同
    vis_c0 = np.asarray(vis_acc.values("C0"))
    vis_c1 = np.asarray(vis_acc.values("C1"))
    sanity_ok = bool(np.allclose(vis_c0, vis_c1)) and all(
        np.allclose(vis_c0, np.asarray(vis_acc.values(f"C2p{k}"))) for k in range(len(c2_perms))
    )
    report["sanity_text_perturb_leaves_visual_unchanged"] = sanity_ok

    print(json.dumps({k: report[k] for k in ("episodes", "fused", "OS_C0_minus_C2mean",
          "delta_C0_minus_C1", "sanity_text_perturb_leaves_visual_unchanged")},
          ensure_ascii=False, indent=2))
    if not sanity_ok:
        print("WARNING: vis-only accuracy changed under text permutation — code bug (手册 §2.2)", flush=True)
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

    report = diagnose(test_loader, model, params)

    out_dir = os.path.join(params.work_dir, "diagnose_3a")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    with open(os.path.join(out_dir, f"diagnose_3a_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
