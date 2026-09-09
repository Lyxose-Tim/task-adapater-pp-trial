"""Offline Stage-E heatmaps from plan_dump (no model forward).

手册：π 为帧×阶段，帧缩略图拼在横轴。交接包⑯主图 A 固定窗口 → B λ=0 → C λ=0.3。
展示冻结 12 episode 中第一个含 pretend 类的 episode（ep1，类 78），query 取该类（q=3）。
CA 轴 T=7：横轴 7 列对应采样帧 0..6（第 8 帧只参与 CA 残差，不单独成列）。
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "returns" / "2026-09-09_E-I1-plan-dump"
FIG = Path(__file__).resolve().parents[1] / "experiments" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
THUMB = ROOT / "thumbs"

def _latest(subdir, suffix):
    files = sorted((ROOT / subdir).glob(f"plan_dump_*.{suffix}"))
    if not files:
        raise FileNotFoundError(subdir)
    return files[-1]


lam0 = np.load(_latest("lam0", "npz"))
lam03 = np.load(_latest("lam03", "npz"))
rho1 = np.load(_latest("lam03_rho1", "npz"))
s917 = np.load(_latest("lam03_s917", "npz"))
B0 = np.load(_latest("B0", "npz"))
meta = json.loads(_latest("lam03", "json").read_text(encoding="utf-8"))

EP, Q = 1, 3  # 冻结流中首个 pretend episode；query = 类 78
Q2 = 0        # 同 episode 另一 query，证非固定模板
labs = meta["episodes"][EP]["label_idx"]
qmeta = meta["episodes"][EP]["query_meta"][Q]


def band_mass_np(plan):
    """与 ot_align.band_mass(..., bandwidth=0) 同式，[T,K] → 标量。"""
    t, k = plan.shape
    ii = np.arange(t)[:, None] / max(t - 1, 1)
    ss = np.arange(k)[None, :] / max(k - 1, 1)
    nearest = ((ii - ss) ** 2).argmin(1)
    row = plan.sum(1).clip(min=1e-12)
    return float((plan[np.arange(t), nearest] / row).mean())


def load_thumbs(qm, n_ca=7):
    vid = Path(qm["path"]).name
    fids = [int(f) for f in qm["frame_id"][:n_ca]]
    imgs = []
    for fid in fids:
        p = THUMB / vid / f"img_{fid:05d}.jpg"
        imgs.append(np.asarray(Image.open(p).convert("RGB")) if p.exists() else None)
    return imgs


def panel_with_thumbs(fig, gs_cell, matrix, title, thumbs, vmin, vmax, ylabel=False):
    """matrix [T,K] → 横轴帧、纵轴阶段；下方拼缩略图。"""
    inner = gs_cell.subgridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.04)
    ax = fig.add_subplot(inner[0])
    ax_t = fig.add_subplot(inner[1])
    im = ax.imshow(matrix.T, aspect="auto", origin="upper", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(matrix.shape[0]))
    ax.set_xticklabels([str(i) for i in range(matrix.shape[0])], fontsize=8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["k=0", "k=1", "k=2"], fontsize=8)
    if ylabel:
        ax.set_ylabel("stage")
    ax.tick_params(bottom=False, labelbottom=False)
    n = matrix.shape[0]
    ax_t.set_xlim(0, n)
    ax_t.set_ylim(0, 1)
    ax_t.axis("off")
    for i, imarr in enumerate(thumbs):
        if imarr is None:
            continue
        ax_t.imshow(imarr, extent=(i + 0.08, i + 0.92, 0.05, 0.95), aspect="auto")
    ax_t.set_xlabel("CA frame t (thumb = sampled frame t)")
    return im


thumbs = load_thumbs(qmeta)
thumbs2 = load_thumbs(meta["episodes"][EP]["query_meta"][Q2])
p0 = lam0["plan"][EP, Q, Q]
p3 = lam03["plan"][EP, Q, Q]
wm = B0["window_mask"]
vmax = max(0.25, float(max(p0.max(), p3.max(), wm.max())))
b0 = band_mass_np(p0)
b3 = band_mass_np(p3)

fig = plt.figure(figsize=(12.4, 4.6))
gs = GridSpec(1, 3, figure=fig, wspace=0.18)
panel_with_thumbs(fig, gs[0], wm, "A  B0 fixed window (eq.16)", thumbs, 0, vmax, ylabel=True)
panel_with_thumbs(fig, gs[1], p0, rf"B  OT $\lambda$=0  band={b0:.2f}", thumbs, 0, vmax)
panel_with_thumbs(fig, gs[2], p3, rf"C  OT $\lambda$=0.3  band={b3:.2f}", thumbs, 0, vmax)
fig.suptitle(
    f"ep{EP} query{Q}  pretend class {labs[Q]}  labels={labs}  (seed 916, content-blind)",
    fontsize=10,
)
fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.08, wspace=0.22)
fig.savefig(FIG / "i1e_heatmap_ABC.png", dpi=140)
plt.close()

fig = plt.figure(figsize=(8.6, 6.4))
gs = GridSpec(2, 2, figure=fig, wspace=0.16, hspace=0.28)
for row, qq, th, lab in (
    (0, Q, thumbs, f"q{Q} class {labs[Q]} (pretend)"),
    (1, Q2, thumbs2, f"q{Q2} class {labs[Q2]}"),
):
    a = lam0["plan"][EP, qq, qq]
    b = lam03["plan"][EP, qq, qq]
    panel_with_thumbs(fig, gs[row, 0], a, rf"$\lambda$=0  {lab}  band={band_mass_np(a):.2f}", th, 0, 0.35, ylabel=True)
    panel_with_thumbs(fig, gs[row, 1], b, rf"$\lambda$=0.3  {lab}  band={band_mass_np(b):.2f}", th, 0, 0.35)
fig.suptitle(f"same ep{EP}: two queries — λ=0.3 more banded, not one template")
fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.06, wspace=0.20, hspace=0.32)
fig.savefig(FIG / "i1e_heatmap_twoquery.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(5.6, 3.4))
x, w = np.arange(3), 0.35
ax.bar(x - w / 2, lam03["mass"].mean((0, 1, 2)), w, label=r"$\lambda$=0.3 $\rho$=None", color="#F58518")
ax.bar(x + w / 2, rho1["mass"].mean((0, 1, 2)), w, label=r"$\lambda$=0.3 $\rho$=1", color="#E45756")
ax.axhline(1 / 3, color="0.4", ls="--", lw=0.8)
ax.set_xticks(x, ["k=0", "k=1", "k=2"])
ax.set_ylabel("mean stage mass")
ax.set_ylim(0.30, 0.36)
ax.legend(frameon=False, fontsize=8)
ax.set_title(r"Appendix: $\rho$=1 relaxes stage mass vs $1/K$")
fig.tight_layout()
fig.savefig(FIG / "i1e_rho1_stagemass.png", dpi=140)
plt.close()

fig = plt.figure(figsize=(8.4, 4.4))
gs = GridSpec(1, 2, figure=fig, wspace=0.18)
a = lam03["plan"][EP, Q, Q]
b = s917["plan"][EP, Q, Q]
panel_with_thumbs(fig, gs[0], a, rf"seed916 $\lambda$=0.3  band={band_mass_np(a):.2f}", thumbs, 0, 0.35, ylabel=True)
panel_with_thumbs(fig, gs[1], b, rf"seed917 $\lambda$=0.3  band={band_mass_np(b):.2f}", thumbs, 0, 0.35)
fig.suptitle("mechanism check: 2nd seed, same 12-episode stream")
fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.10, wspace=0.20)
fig.savefig(FIG / "i1e_heatmap_seed917.png", dpi=140)
plt.close()

print("ABC band lam0/lam03", round(b0, 4), round(b3, 4))
print("wrote figures", "thumbs", sum(x is not None for x in thumbs), "/", len(thumbs))
