"""Scan seed-916 episode stream until >=3 unique pretend classes (content-blind)."""
import ast, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/rivermind-data/ta-pp")
import dataset as ds

text = Path("/root/rivermind-data/ta-pp/utils.py").read_text(encoding="utf-8")
m = re.search(r"smsm_c = (\[.*?\])\nsmsm_cls", text, re.S)
smsm_c = ast.literal_eval(m.group(1))
PRETEND = {i for i, n in enumerate(smsm_c) if "pretend" in n.lower()}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(916)
test_file = "/root/rivermind-data/ta-pp/data/full/smsm_cmn/annotations/test.txt"
n_max = 80
dm = ds.SetDataManager(224, n_query=1, num_segments=8, n_eposide=n_max, n_way=5, n_support=1, num_workers=0)
loader = dm.get_data_loader(test_file, aug=False, sample_window=None)
seen, n_star = set(), None
for ep, (_x, label) in enumerate(loader):
    labs = [int(v) for v in label[:, 0].numpy()]
    seen |= {c for c in labs if c in PRETEND}
    if ep < 12:
        print(f"ep{ep} labs={labs} pretend={[c for c in labs if c in PRETEND]}")
    if n_star is None and ep + 1 >= 12 and len(seen) >= 3:
        n_star = ep + 1
        print("N*", n_star, "pretend_ids", sorted(seen), "names", [smsm_c[c] for c in sorted(seen)])
        break
else:
    print("NOT_FOUND", "seen", sorted(seen), [smsm_c[c] for c in sorted(seen)])
if n_star:
    print("RESULT dump_episodes", n_star)
