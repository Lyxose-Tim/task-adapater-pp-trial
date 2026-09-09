import ast, re
from pathlib import Path

import numpy as np
import torch

text = Path("/root/rivermind-data/ta-pp/utils.py").read_text(encoding="utf-8")
smsm_c = ast.literal_eval(re.search(r"smsm_c = (\[.*?\])\nsmsm_cls", text, re.S).group(1))
PRETEND = {i for i, n in enumerate(smsm_c) if "pretend" in n.lower()}
video_list = [x.strip().split(" ") for x in open("/root/rivermind-data/ta-pp/data/full/smsm_cmn/annotations/test.txt")]
cl_list = np.unique([int(v[2]) for v in video_list]).tolist()
torch.manual_seed(916)
torch.cuda.manual_seed_all(916)
np.random.seed(916)
n_way, n_max = 5, 120
seen, n_star = set(), None
for ep in range(n_max):
    idx = torch.randperm(len(cl_list))[:n_way]
    labs = [int(cl_list[i]) for i in idx.tolist()]
    seen |= {c for c in labs if c in PRETEND}
    if ep < 12:
        print("ep", ep, "labs", labs, "pretend", [c for c in labs if c in PRETEND], flush=True)
    if n_star is None and ep + 1 >= 12 and len(seen) >= 3:
        n_star = ep + 1
        print("N*", n_star, "pretend", sorted(seen), [smsm_c[c] for c in sorted(seen)], flush=True)
        break
else:
    print("NOT_FOUND", sorted(seen), flush=True)
if n_star:
    print("RESULT", n_star, flush=True)
