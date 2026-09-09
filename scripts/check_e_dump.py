import json, glob, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/root/rivermind-data/ta-pp/workspace_E_dump_lam03/plan_dump"
js = sorted(glob.glob(os.path.join(root, "plan_dump_*.json")))
npz = sorted(glob.glob(os.path.join(root, "plan_dump_*.npz")))
assert js and npz, (js, npz)
meta = json.load(open(js[-1], encoding="utf-8"))
arr = np.load(npz[-1])
print("json", js[-1])
print("npz", npz[-1], "keys", list(arr.keys()))
print("mode", meta["mode"], "align", meta["align_mode"], "n", meta["dump_episodes"], "md5", meta.get("ckpt_md5"))
assert meta["dump_episodes"] == 12
assert len(meta["episodes"]) == 12
empty = 0
for ep in meta["episodes"]:
    for k in ("label_idx", "y_query", "pred", "vis", "sem", "fused", "query_meta"):
        assert k in ep, k
    for k in ("vis", "sem", "fused"):
        a = np.asarray(ep[k], dtype=np.float64)
        assert np.isfinite(a).all(), (ep["episode"], k, a)
    assert len(ep["query_meta"]) >= 1
    for q in ep["query_meta"]:
        assert q and q.get("path") and q.get("frame_id") and q.get("num_frames") is not None, q
        if not q.get("path"):
            empty += 1
print("query_meta ok; sample", meta["episodes"][0]["query_meta"][0])
if meta["align_mode"] == "ot":
    print("plan", arr["plan"].shape, "cost", arr["cost"].shape, "mass", arr["mass"].shape, "D", arr["D"].shape)
    E = 12
    assert arr["plan"].shape[0] == E and arr["plan"].ndim == 5
    assert arr["cost"].shape == arr["plan"].shape
    assert arr["mass"].shape[:3] == arr["plan"].shape[:3]
    assert arr["D"].ndim == 2
else:
    print("window_mask", arr["window_mask"].shape)
    assert arr["window_mask"].shape[1] == 3
assert meta.get("ckpt_md5")
print("DUMP_OK")
