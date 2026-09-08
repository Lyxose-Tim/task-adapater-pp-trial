import json, glob, sys
files = sorted(glob.glob("/root/rivermind-data/ta-pp/workspace_robust_mg_smoke/robust_eval/robust_eval_*.json"))
print("json", files)
assert files, "no json"
r = json.load(open(files[-1], encoding="utf-8"))
print("episodes", r["episodes"], "ot", r["ot"])
for name, w in r["windows"].items():
    print(name, "acc", w["acc"], "dAcc", w["dAcc_mean"], "n_ep", len(w.get("per_episode_fused", [])))
    if "OS_mean" in w:
        print("  OS", w["OS_mean"], w["OS_ci95"])
    if "stage_mass_relax" in w:
        print("  relax", w["stage_mass_relax"], "prof", w["stage_mass_profile"])
w = r["windows"]["normal"]
assert r["episodes"] == 20
assert "OS_mean" in w and "stage_mass_relax" in w and "stage_mass_profile" in w
assert len(w["per_episode_fused"]) == 20
assert len(w["stage_mass_profile"]) == 3
print("SMOKE_OK")
