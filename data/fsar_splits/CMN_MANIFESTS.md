# Canonical CMN few-shot manifests

The vendored lists under `ssv2_cmn/` and `kinetics_cmn/` come from the
official [ffmpbgrnn/CMN](https://github.com/ffmpbgrnn/CMN) repository.  Each
dataset contains 6,400 train, 1,200 validation, and 2,400 test rows (64/12/24
classes, 100 videos per class).  Rows use `class name/video_id` ordering.

SSv2 raw sources:

- `https://raw.githubusercontent.com/ffmpbgrnn/CMN/master/smsm-100/train.list`
- `https://raw.githubusercontent.com/ffmpbgrnn/CMN/master/smsm-100/val.list`
- `https://raw.githubusercontent.com/ffmpbgrnn/CMN/master/smsm-100/test.list`

Kinetics raw sources use the corresponding `kinetics-100/` paths.  The local
SHA-256 values are recorded by `scripts/prepare_ssv2.py` in
`dataset/smsm_cmn/ssv2_small_provenance.json`; changing any list changes the
recorded experiment provenance.
