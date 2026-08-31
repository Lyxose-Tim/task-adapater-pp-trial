# Windows 11：Kinetics-400 官方数据下载

本项目从 CVDF 官方 S3 下载 Kinetics-400。根目录固定为：

```text
D:\task-adapter-pp-trial\dataset\Kinetics400
```

## 与 Linux 官方脚本的区别

官方仓库提供的 `k400_downloader.sh` 使用 Bash、`wget -c -i` 和 POSIX 路径；Windows 11 默认不具备这些接口。本项目使用：

- `scripts/prepare_kinetics400_windows.py`；
- Windows `aria2c.exe` 多连接断点续传（推荐）或 `curl.exe --continue-at -`；
- Python 线程池控制对象级并发；
- `.part` 临时文件和完成后的原子改名；
- 官方 `Content-Length` 作为下载完成硬门禁；
- JSONL 事件日志和可重复执行的状态检查。

不需要 WSL、Git Bash、GNU `wget`、`parallel` 或 7-Zip。

## 文件布局与保留策略

```text
dataset/Kinetics400/
├── manifests/
│   ├── k400_train_path.txt
│   ├── k400_val_path.txt
│   ├── k400_test_path.txt
│   └── remote_objects.json
├── raw/
│   ├── train/          # 242 个官方 tar.gz
│   ├── val/            # 20 个官方 tar.gz
│   ├── test/           # 39 个官方 tar.gz
│   ├── replacements/   # 官方损坏视频替换包
│   ├── annotations/    # 原始 CSV
│   └── metadata/       # 官方 README
└── logs/
```

`raw` 中的官方对象永不自动删除或覆盖。下载中的文件使用 `.part` 后缀；达到远端记录的精确字节数后才原子改名。再次执行同一下载命令会跳过完整对象并续传 `.part`。

S3 返回的归档 ETag 带有分片后缀，因此不能直接视为 MD5。工具保留 ETag，并以官方字节数、完整 tar 流读取和可选本地 SHA-256 作为验收证据。

## 精确规模

2026-07-17 从官方对象 HEAD 固化的清单为 306 个对象，共 `467,287,479,542` 字节（约 435.2 GiB）：

| 分组 | 对象数 | 字节数 |
|---|---:|---:|
| train | 242 | 370,322,270,904 |
| val | 20 | 30,354,517,239 |
| test | 39 | 56,094,802,765 |
| replacements | 1 | 10,502,888,181 |
| annotations | 3 | 13,000,121 |
| metadata | 1 | 332 |

完整原始归档可放入当前 D 盘，但在保留归档后，剩余空间不足以再创建一份同等规模的全量解压副本。因此当前阶段优先完成并验证全部原始对象；后续针对 Task-Adapter++ 的 Kinetics-100/CMN 清单做选择性解包，或在增加存储后执行全量解包。

## 命令

生成/刷新远端对象清单：

```powershell
conda activate task_adapter_pp
$env:PYTHONUTF8 = "1"
python scripts/prepare_kinetics400_windows.py metadata --workers 16
```

下载或恢复全部对象：

```powershell
python scripts/prepare_kinetics400_windows.py download --groups all `
  --engine aria2 --workers 4 --connections-per-file 8
```

aria2 为每个对象保留 `.part.aria2` 控制文件。控制文件存在时，`.part` 的逻辑长度可能已等于远端文件长度，但不代表所有分片都已下载；只有 aria2 成功退出、控制文件消失且字节数精确匹配后，脚本才会将其原子改名为最终归档。不要手工删除 `.part` 或 `.aria2` 文件。

检查进度：

```powershell
python scripts/prepare_kinetics400_windows.py status
Get-Content dataset/Kinetics400/logs/download_stdout.log -Tail 20
Get-Content dataset/Kinetics400/logs/download_stderr.log -Tail 20
Get-Content dataset/Kinetics400/logs/download_events.jsonl -Tail 20
```

下载完成后逐归档流式验收；`--sha256` 会额外生成本地 SHA-256，但需要再次完整读取约 435 GiB：

```powershell
python scripts/prepare_kinetics400_windows.py verify `
  --groups train val test replacements --sha256
```

当前后台控制器 PID 仅是本次运行信息，不属于数据协议。若 Windows 重启或进程停止，直接重新执行 `download` 命令即可安全续传。

## 官方来源

- <https://github.com/cvdfoundation/kinetics-dataset>
- <https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt>
- <https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt>
- <https://s3.amazonaws.com/kinetics/400/test/k400_test_path.txt>
- <https://s3.amazonaws.com/kinetics/400/replacement_for_corrupted_k400.tgz>
