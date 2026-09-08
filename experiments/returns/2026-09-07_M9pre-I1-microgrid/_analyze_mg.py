"""I1-mg 2x2: I_Acc + truncation DiD_t (paired 95% CI) + stage_mass checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "robust"
FILES = {
    "l03r0": ROOT / "robust_eval_2026-09-07_13-22-37.json",  # λ=0.3, ρ=None
    "l03r1": ROOT / "robust_eval_2026-09-07_14-43-25.json",  # λ=0.3, ρ=1
    "l1r0": ROOT / "robust_eval_2026-09-07_16-03-56.json",   # λ=1, ρ=None
    "l1r1": ROOT / "robust_eval_2026-09-07_17-25-01.json",   # λ=1, ρ=1
}
LABEL = {
    "l03r0": "(λ=0.3, ρ=None)",
    "l03r1": "(λ=0.3, ρ=1)",
    "l1r0": "(λ=1, ρ=None)",
    "l1r1": "(λ=1, ρ=1)",
}


def ci95(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    half = float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
    return mean, half


def main() -> None:
    data = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in FILES.items()}
    n = None
    for k, r in data.items():
        ot = r["ot"]
        print(f"== {k} {LABEL[k]}  episodes={r['episodes']}  ot={ot}")
        for wname, w in r["windows"].items():
            n_ep = len(w["per_episode_fused"])
            n = n_ep if n is None else n
            assert n_ep == n, (k, wname, n_ep, n)
            extra = ""
            if "OS_mean" in w:
                extra += f" OS={w['OS_mean']:.4f}±{w['OS_ci95']:.4f}"
            if "stage_mass_relax" in w:
                sm = w["stage_mass_relax"]
                extra += (
                    f" relax={sm['mean']:.6f}/{sm['median']:.6f}/{sm['p95']:.6f}"
                    f" prof={['%.5f' % v for v in w['stage_mass_profile']]}"
                )
            print(
                f"  {wname:>6}: acc={w['acc']:.4f}±{w['ci95']:.4f}"
                f" dAcc={w['dAcc_mean']:+.4f}±{w['dAcc_ci95']:.4f}"
                f" rf={w['repeat_frac']:.6g} n={n_ep}{extra}"
            )
        assert r["episodes"] == n

    # paired I_Acc on normal fused acc (fraction, report in pt)
    a = {k: np.asarray(data[k]["windows"]["normal"]["per_episode_fused"], dtype=np.float64) for k in FILES}
    for k in FILES:
        assert len(a[k]) == n
    i_ep = (a["l1r1"] - a["l1r0"]) - (a["l03r1"] - a["l03r0"])
    i_mean, i_half = ci95(i_ep)
    print("\n== I_Acc (paired, pt)  [Acc(1,1)-Acc(1,None)] - [Acc(0.3,1)-Acc(0.3,None)]")
    print(f"   {i_mean * 100:+.4f} ± {i_half * 100:.4f}   CI=[{(i_mean - i_half)*100:+.4f}, {(i_mean + i_half)*100:+.4f}]")
    print(f"   Acc diffs: (1,1)-(1,None)={(a['l1r1']-a['l1r0']).mean()*100:+.4f}; "
          f"(0.3,1)-(0.3,None)={(a['l03r1']-a['l03r0']).mean()*100:+.4f}")

    print("\n== DiD_t on dAcc (truncation interaction, pt)")
    print("   DiD_t = [dAcc(1,1)-dAcc(1,None)] - [dAcc(0.3,1)-dAcc(0.3,None)]")
    for wname in ("head", "tail", "shrink"):
        d = {}
        for k in FILES:
            win = np.asarray(data[k]["windows"][wname]["per_episode_fused"], dtype=np.float64)
            d[k] = win - a[k]
        did = (d["l1r1"] - d["l1r0"]) - (d["l03r1"] - d["l03r0"])
        m, h = ci95(did)
        print(f"   {wname:>6}: {m*100:+.4f} ± {h*100:.4f}  CI=[{(m-h)*100:+.4f}, {(m+h)*100:+.4f}]")

    print("\n== DiD on window Acc (same 2x2 form, pt)")
    for wname in ("normal", "head", "tail", "shrink"):
        accw = {k: np.asarray(data[k]["windows"][wname]["per_episode_fused"], dtype=np.float64) for k in FILES}
        did = (accw["l1r1"] - accw["l1r0"]) - (accw["l03r1"] - accw["l03r0"])
        m, h = ci95(did)
        print(f"   {wname:>6}: {m*100:+.4f} ± {h*100:.4f}  CI=[{(m-h)*100:+.4f}, {(m+h)*100:+.4f}]")

    print("\n== stage_mass mechanism")
    for k in ("l03r0", "l03r1", "l1r0", "l1r1"):
        rel = data[k]["windows"]["normal"]["stage_mass_relax"]["mean"]
        print(f"   {k} normal relax mean={rel:.6f}")
    none_rel = np.mean([data[k]["windows"]["normal"]["stage_mass_relax"]["mean"] for k in ("l03r0", "l1r0")])
    rho1_rel = np.mean([data[k]["windows"]["normal"]["stage_mass_relax"]["mean"] for k in ("l03r1", "l1r1")])
    print(f"   rho=None mean relax={none_rel:.6f}; rho=1 mean relax={rho1_rel:.6f}; ratio={rho1_rel/max(none_rel,1e-12):.1f}x")
    for k in ("l03r1", "l1r1"):
        nprof = np.asarray(data[k]["windows"]["normal"]["stage_mass_profile"], dtype=np.float64)
        hprof = np.asarray(data[k]["windows"]["head"]["stage_mass_profile"], dtype=np.float64)
        tprof = np.asarray(data[k]["windows"]["tail"]["stage_mass_profile"], dtype=np.float64)
        print(f"   {k} Δprof head-normal k0={hprof[0]-nprof[0]:+.6f}  tail-normal k2={tprof[2]-nprof[2]:+.6f}")
        print(f"      normal {nprof} head {hprof} tail {tprof}")


if __name__ == "__main__":
    main()
