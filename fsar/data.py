"""Data utilities for the few-shot action-recognition experiments.

The original Task-Adapter++ loader expects extracted JPEG directories and
parses annotation rows with ``split(" ")``.  This module keeps the annotation
format (``path num_frames label``), but parses its two numeric fields from the
right so that paths containing spaces are valid.  Videos are decoded directly
with decord and, when requested, only the sampled frames are cached as JPEGs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


PathLike = Union[str, os.PathLike]
TemporalWindow = Optional[Tuple[Union[int, float], Union[int, float]]]
VIDEO_EXTENSIONS = (".webm", ".mp4", ".avi", ".mov", ".mkv", ".m4v")
SSV2_NAMES = {
    "ssv2",
    "ssv2-small",
    "something",
    "something-something",
    "something-something-v2",
    "somethingv2",
    "somethingcmn",
    "smsm",
    "smsm-cmn",
    "smsm_cmn",
}


@dataclass(frozen=True)
class AnnotationRecord:
    """One ``path num_frames label`` annotation row."""

    path: str
    num_frames: int
    label: int

    @property
    def video_path(self) -> str:
        """Compatibility alias used by some experiment scripts."""

        return self.path

    def to_line(self) -> str:
        return "{} {} {}".format(self.path, self.num_frames, self.label)


# A shorter name is convenient in notebooks and backwards-compatible tests.
VideoRecord = AnnotationRecord


@dataclass
class ValidationReport:
    """Structured validation result returned by annotation validators."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def ok(self) -> bool:
        return self.valid

    def __bool__(self) -> bool:
        return self.valid

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "stats": self.stats,
        }


def parse_annotation_line(
    line: str,
    *,
    line_number: Optional[int] = None,
    source: Optional[PathLike] = None,
) -> AnnotationRecord:
    """Parse ``path num_frames label`` by taking numeric fields from the right.

    ``rsplit(maxsplit=2)`` is deliberate: a row such as
    ``D:/datasets/a clip/video.webm 42 7`` retains the complete path.
    """

    raw = line.strip()
    location = ""
    if source is not None:
        location = str(source)
    if line_number is not None:
        location = "{}:{}".format(location, line_number) if location else "line {}".format(line_number)
    if location:
        location += ": "

    if not raw or raw.startswith("#"):
        raise ValueError("{}empty/comment line is not an annotation".format(location))
    fields = raw.rsplit(maxsplit=2)
    if len(fields) != 3:
        raise ValueError(
            "{}expected 'path num_frames label', got {!r}".format(location, raw)
        )
    path, num_frames_text, label_text = fields
    if not path:
        raise ValueError("{}video path is empty".format(location))
    try:
        num_frames = int(num_frames_text)
        label = int(label_text)
    except ValueError as exc:
        raise ValueError(
            "{}num_frames and label must be integers, got {!r} {!r}".format(
                location, num_frames_text, label_text
            )
        ) from exc
    if num_frames <= 0:
        raise ValueError("{}num_frames must be positive, got {}".format(location, num_frames))
    if label < 0:
        raise ValueError("{}label must be non-negative, got {}".format(location, label))
    return AnnotationRecord(path=path, num_frames=num_frames, label=label)


def iter_annotations(path: PathLike) -> Iterator[AnnotationRecord]:
    """Yield records, ignoring blank lines and full-line comments."""

    annotation_path = Path(path)
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            yield parse_annotation_line(
                line, line_number=line_number, source=annotation_path
            )


def read_annotations(path: PathLike) -> List[AnnotationRecord]:
    return list(iter_annotations(path))


# Common singular spellings used by downstream scripts.
load_annotations = read_annotations
read_annotation_file = read_annotations
parse_annotation = parse_annotation_line


def write_annotations(
    records: Iterable[AnnotationRecord],
    path: PathLike,
    *,
    sort_records: bool = False,
) -> Path:
    """Write an annotation file atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    if sort_records:
        rows.sort(key=lambda row: (row.label, row.path))
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.to_line() + "\n")
    os.replace(str(temporary), str(destination))
    return destination


write_annotation_file = write_annotations


def _resolve_record_path(record_path: str, path_root: Optional[PathLike]) -> Path:
    path = Path(record_path).expanduser()
    if not path.is_absolute() and path_root is not None:
        path = Path(path_root).expanduser() / path
    return path


def validate_annotations(
    annotations: Union[PathLike, Iterable[AnnotationRecord]],
    *,
    check_paths: bool = True,
    path_root: Optional[PathLike] = None,
    expected_labels: Optional[Iterable[int]] = None,
    expected_class_count: Optional[int] = None,
    min_samples_per_class: int = 1,
    verify_frame_counts: bool = False,
    frame_count_fn: Optional[Callable[[PathLike], int]] = None,
) -> ValidationReport:
    """Validate one annotation collection without opening video payloads."""

    report = ValidationReport()
    try:
        records = (
            read_annotations(annotations)
            if isinstance(annotations, (str, os.PathLike))
            else list(annotations)
        )
    except (OSError, ValueError) as exc:
        report.errors.append(str(exc))
        return report

    class_counts = Counter(record.label for record in records)
    canonical_paths: MutableMapping[str, int] = Counter()
    missing_paths: List[str] = []
    frame_count_mismatches: List[Dict[str, Any]] = []
    frame_count_errors: List[Dict[str, str]] = []
    extension_counts: MutableMapping[str, int] = Counter()
    for record in records:
        resolved = _resolve_record_path(record.path, path_root)
        canonical = os.path.normcase(os.path.abspath(str(resolved)))
        canonical_paths[canonical] += 1
        extension_counts[resolved.suffix.lower() or "<directory>"] += 1
        if check_paths and not resolved.exists():
            missing_paths.append(record.path)
        elif verify_frame_counts:
            counter = frame_count_fn or probe_video_num_frames
            try:
                actual_count = int(counter(resolved))
            except (OSError, RuntimeError, ValueError, ImportError) as exc:
                frame_count_errors.append({"path": record.path, "error": str(exc)})
            else:
                if actual_count != record.num_frames:
                    frame_count_mismatches.append(
                        {
                            "path": record.path,
                            "annotated": record.num_frames,
                            "actual": actual_count,
                        }
                    )

    duplicates = sorted(path for path, count in canonical_paths.items() if count > 1)
    if not records:
        report.errors.append("annotation contains no records")
    if duplicates:
        report.errors.append("{} duplicate video path(s)".format(len(duplicates)))
    if missing_paths:
        report.errors.append("{} annotated path(s) do not exist".format(len(missing_paths)))
    if frame_count_errors:
        report.errors.append(
            "could not verify frame count for {} video(s)".format(len(frame_count_errors))
        )
    if frame_count_mismatches:
        report.errors.append(
            "{} annotated frame count(s) differ from decoded videos".format(
                len(frame_count_mismatches)
            )
        )
    too_small = {
        label: count
        for label, count in class_counts.items()
        if count < int(min_samples_per_class)
    }
    if too_small:
        report.errors.append(
            "classes below min_samples_per_class={}: {}".format(
                min_samples_per_class, sorted(too_small.items())
            )
        )

    actual_labels = set(class_counts)
    if expected_labels is not None:
        expected = {int(label) for label in expected_labels}
        missing_labels = sorted(expected - actual_labels)
        unexpected_labels = sorted(actual_labels - expected)
        if missing_labels:
            report.errors.append("missing expected labels: {}".format(missing_labels))
        if unexpected_labels:
            report.errors.append("unexpected labels: {}".format(unexpected_labels))
    if expected_class_count is not None and len(class_counts) != expected_class_count:
        report.errors.append(
            "expected {} classes, found {}".format(expected_class_count, len(class_counts))
        )

    frame_counts = [record.num_frames for record in records]
    report.stats.update(
        {
            "records": len(records),
            "classes": len(class_counts),
            "class_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
            "min_frames": min(frame_counts) if frame_counts else None,
            "max_frames": max(frame_counts) if frame_counts else None,
            "duplicate_paths": duplicates,
            "missing_paths": missing_paths,
            "frame_count_mismatches": frame_count_mismatches,
            "frame_count_errors": frame_count_errors,
            "extensions": dict(sorted(extension_counts.items())),
        }
    )
    return report


validate_annotation_file = validate_annotations


def validate_split_annotations(
    train: PathLike,
    val: PathLike,
    test: PathLike,
    *,
    check_paths: bool = True,
    path_root: Optional[PathLike] = None,
    require_disjoint_labels: bool = True,
    expected_class_counts: Optional[Mapping[str, int]] = None,
    min_samples_per_class: int = 1,
    verify_frame_counts: bool = False,
    frame_count_fn: Optional[Callable[[PathLike], int]] = None,
) -> ValidationReport:
    """Validate the complete train/val/test annotation contract.

    Few-shot SSv2-CMN is a class-disjoint 64/12/24 split, hence label overlap
    checking is enabled by default.  It can be disabled for conventional
    instance-disjoint datasets.
    """

    report = ValidationReport(stats={"splits": {}})
    paths = {"train": train, "val": val, "test": test}
    records_by_split: Dict[str, List[AnnotationRecord]] = {}
    canonical_to_splits: Dict[str, set] = defaultdict(set)

    for split, annotation_path in paths.items():
        try:
            records = read_annotations(annotation_path)
        except (OSError, ValueError) as exc:
            report.errors.append("{}: {}".format(split, exc))
            records = []
        records_by_split[split] = records
        split_report = validate_annotations(
            records,
            check_paths=check_paths,
            path_root=path_root,
            expected_class_count=(expected_class_counts or {}).get(split),
            min_samples_per_class=min_samples_per_class,
            verify_frame_counts=verify_frame_counts,
            frame_count_fn=frame_count_fn,
        )
        report.errors.extend("{}: {}".format(split, item) for item in split_report.errors)
        report.warnings.extend("{}: {}".format(split, item) for item in split_report.warnings)
        report.stats["splits"][split] = split_report.stats
        for record in records:
            resolved = _resolve_record_path(record.path, path_root)
            canonical = os.path.normcase(os.path.abspath(str(resolved)))
            canonical_to_splits[canonical].add(split)

    cross_split_duplicates = {
        path: sorted(splits)
        for path, splits in canonical_to_splits.items()
        if len(splits) > 1
    }
    if cross_split_duplicates:
        report.errors.append(
            "{} video path(s) occur in multiple splits".format(
                len(cross_split_duplicates)
            )
        )

    label_sets = {
        split: {record.label for record in records}
        for split, records in records_by_split.items()
    }
    label_overlap: Dict[str, List[int]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(label_sets[left] & label_sets[right])
        label_overlap["{}-{}".format(left, right)] = overlap
        if require_disjoint_labels and overlap:
            report.errors.append(
                "{} and {} share labels {}".format(left, right, overlap)
            )

    report.stats["cross_split_duplicate_paths"] = cross_split_duplicates
    report.stats["label_overlap"] = label_overlap
    report.stats["total_records"] = sum(len(rows) for rows in records_by_split.values())
    report.stats["total_labels"] = len(set().union(*label_sets.values()))
    return report


def _normalise_temporal_window(
    num_frames_total: int, temporal_window: TemporalWindow
) -> Tuple[int, int]:
    if num_frames_total <= 0:
        raise ValueError("num_frames_total must be positive")
    if temporal_window is None:
        return 0, num_frames_total
    if len(temporal_window) != 2:
        raise ValueError("temporal_window must contain (start, end)")
    start, end = temporal_window
    # Windows in [0, 1] are normalized, matching the robustness protocol
    # ([.25, 1], [0, .75], [.125, .875]).  Other integral values are indices.
    normalized = (
        isinstance(start, float)
        or isinstance(end, float)
        or (0 <= start <= 1 and 0 <= end <= 1)
    )
    if normalized:
        start_float, end_float = float(start), float(end)
        if not (0.0 <= start_float < end_float <= 1.0):
            raise ValueError("normalized temporal_window must satisfy 0 <= start < end <= 1")
        first = int(math.floor(start_float * num_frames_total))
        stop = int(math.ceil(end_float * num_frames_total))
    else:
        if isinstance(start, float) or isinstance(end, float):
            raise ValueError("frame-index temporal_window endpoints must be integers")
        first, stop = int(start), int(end)
        if not (0 <= first < stop <= num_frames_total):
            raise ValueError(
                "frame-index temporal_window must satisfy 0 <= start < end <= num_frames_total"
            )
    first = min(first, num_frames_total - 1)
    stop = max(first + 1, min(stop, num_frames_total))
    return first, stop


def sample_frame_indices(
    num_frames_total: int,
    num_frames: int = 8,
    random_select: bool = False,
    *,
    strategy: Optional[str] = None,
    temporal_window: TemporalWindow = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Sample one frame from each of ``num_frames`` temporal intervals.

    ``strategy='interval'`` selects interval midpoints. ``'random'`` selects a
    uniformly random index inside every interval.  If an interval is narrower
    than one frame, the nearest valid frame is repeated.  Returned indices are
    zero based and always lie inside ``temporal_window``.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames_total <= 0:
        raise ValueError("num_frames_total must be positive")
    if strategy is None:
        strategy = "random" if random_select else "interval"
    strategy = strategy.lower().replace("_", "-")
    if strategy in {"uniform", "center", "centre", "deterministic"}:
        strategy = "interval"
    if strategy not in {"interval", "random"}:
        raise ValueError("strategy must be 'interval' or 'random'")
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")

    first, stop = _normalise_temporal_window(num_frames_total, temporal_window)
    edges = np.linspace(first, stop, num_frames + 1, dtype=np.float64)
    generator = rng if rng is not None else np.random.default_rng(seed)
    indices: List[int] = []
    for index in range(num_frames):
        low = int(math.floor(edges[index]))
        high = int(math.ceil(edges[index + 1]))
        low = min(max(low, first), stop - 1)
        high = min(max(high, low + 1), stop)
        if strategy == "random":
            selected = int(generator.integers(low, high))
        else:
            selected = int(math.floor((edges[index] + edges[index + 1]) / 2.0))
            selected = min(max(selected, low), high - 1)
        indices.append(selected)
    return indices


def probe_video_num_frames(video_path: PathLike, *, num_threads: int = 1) -> int:
    """Return a video's decodable frame count using a lazy decord import."""

    try:
        from decord import VideoReader, cpu
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError("decord is required to inspect video files") from exc
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=num_threads)
    count = int(len(reader))
    if count <= 0:
        raise ValueError("video contains no decodable frames: {}".format(video_path))
    return count


class JPEGFrameCache:
    """On-demand cache for sampled frames (not a full-video extraction cache)."""

    def __init__(self, root: PathLike, quality: int = 95) -> None:
        self.root = Path(root)
        self.quality = int(quality)
        if not 1 <= self.quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")

    def _video_directory(self, video_path: PathLike) -> Path:
        path = Path(video_path)
        try:
            stat_key = "{}:{}:{}".format(path.resolve(), path.stat().st_size, path.stat().st_mtime_ns)
        except OSError:
            stat_key = str(path.absolute())
        digest = hashlib.sha1(stat_key.encode("utf-8")).hexdigest()[:12]
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:48] or "video"
        return self.root / "{}-{}".format(safe_stem, digest)

    def frame_path(self, video_path: PathLike, frame_index: int) -> Path:
        return self._video_directory(video_path) / "frame_{:08d}.jpg".format(frame_index)

    def load(self, video_path: PathLike, frame_index: int) -> Optional[np.ndarray]:
        cached = self.frame_path(video_path, frame_index)
        if not cached.is_file():
            return None
        from PIL import Image

        with Image.open(cached) as image:
            return np.asarray(image.convert("RGB")).copy()

    def store(self, video_path: PathLike, frame_index: int, frame: np.ndarray) -> Path:
        from PIL import Image

        destination = self.frame_path(video_path, frame_index)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            "{}.{}.{}.tmp".format(destination.name, os.getpid(), id(frame))
        )
        Image.fromarray(frame.astype(np.uint8), mode="RGB").save(
            temporary, format="JPEG", quality=self.quality
        )
        os.replace(str(temporary), str(destination))
        return destination


def _frames_to_tensor(frames: Sequence[np.ndarray]) -> torch.Tensor:
    array = np.stack(frames, axis=0)
    tensor = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).contiguous()
    return tensor.to(dtype=torch.float32).div_(255.0)


def _apply_frame_transform(
    frames: Sequence[np.ndarray], transform: Callable[[Any], Any]
) -> torch.Tensor:
    from PIL import Image

    transformed: List[torch.Tensor] = []
    for frame in frames:
        value = transform(Image.fromarray(frame.astype(np.uint8), mode="RGB"))
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if not torch.is_tensor(value):
            value = torch.from_numpy(np.asarray(value).copy())
        if value.ndim == 3 and value.shape[-1] in (1, 3, 4):
            value = value.permute(2, 0, 1)
        value = value.to(dtype=torch.float32)
        if value.numel() and value.max().item() > 1.0:
            value = value / 255.0
        transformed.append(value)
    return torch.stack(transformed, dim=0)


def load_video_frames_decord(
    video_path: PathLike,
    num_frames: int = 8,
    transform: Optional[Callable[[Any], Any]] = None,
    *,
    random_select: bool = False,
    strategy: Optional[str] = None,
    temporal_window: TemporalWindow = None,
    seed: Optional[int] = None,
    jpeg_cache: Optional[Union[PathLike, JPEGFrameCache]] = None,
    num_threads: int = 1,
) -> torch.Tensor:
    """Decode only sampled video frames with decord and return ``[T,C,H,W]``.

    When ``jpeg_cache`` is supplied, cache hits bypass decoding for those frame
    indices and cache misses are fetched in one decord batch.  The entire video
    is never materialized or extracted to JPEG.
    """

    try:
        from decord import VideoReader, cpu
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError("decord is required for direct video loading") from exc

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=num_threads)
    indices = sample_frame_indices(
        len(reader),
        num_frames,
        random_select=random_select,
        strategy=strategy,
        temporal_window=temporal_window,
        seed=seed,
    )
    cache = (
        jpeg_cache
        if isinstance(jpeg_cache, JPEGFrameCache)
        else JPEGFrameCache(jpeg_cache)
        if jpeg_cache is not None
        else None
    )

    decoded_by_index: Dict[int, np.ndarray] = {}
    missing: List[int] = []
    for frame_index in dict.fromkeys(indices):
        cached = cache.load(video_path, frame_index) if cache is not None else None
        if cached is None:
            missing.append(frame_index)
        else:
            decoded_by_index[frame_index] = cached
    if missing:
        batch = reader.get_batch(missing).asnumpy()
        for frame_index, frame in zip(missing, batch):
            rgb = np.asarray(frame, dtype=np.uint8)
            decoded_by_index[frame_index] = rgb
            if cache is not None:
                cache.store(video_path, frame_index, rgb)
    frames = [decoded_by_index[index] for index in indices]

    if transform is None:
        return _frames_to_tensor(frames)
    if getattr(transform, "expects_clip", False):
        return transform(_frames_to_tensor(frames))
    return _apply_frame_transform(frames, transform)


@dataclass
class VideoClipTransform:
    """Consistent spatial transform for all frames of one clip."""

    image_size: int = 224
    is_train: bool = True
    horizontal_flip: bool = True
    resize_scale: float = 1.14
    mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    expects_clip: bool = field(default=True, init=False)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 4 or clip.shape[1] != 3:
            raise ValueError("clip must have shape [T, 3, H, W]")
        clip = clip.to(dtype=torch.float32)
        target_short = max(self.image_size, int(round(self.image_size * self.resize_scale)))
        height, width = int(clip.shape[-2]), int(clip.shape[-1])
        scale = target_short / float(min(height, width))
        resized_height = max(self.image_size, int(round(height * scale)))
        resized_width = max(self.image_size, int(round(width * scale)))
        clip = F.interpolate(
            clip,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        if self.is_train:
            max_top = resized_height - self.image_size
            max_left = resized_width - self.image_size
            top = int(torch.randint(max_top + 1, (1,)).item()) if max_top else 0
            left = int(torch.randint(max_left + 1, (1,)).item()) if max_left else 0
        else:
            top = (resized_height - self.image_size) // 2
            left = (resized_width - self.image_size) // 2
        clip = clip[:, :, top : top + self.image_size, left : left + self.image_size]
        if self.is_train and self.horizontal_flip and torch.rand(()) < 0.5:
            clip = torch.flip(clip, dims=(-1,))
        mean = clip.new_tensor(self.mean).view(1, 3, 1, 1)
        std = clip.new_tensor(self.std).view(1, 3, 1, 1)
        return (clip - mean) / std


def is_ssv2_dataset(dataset_name: Optional[str]) -> bool:
    if dataset_name is None:
        return False
    normalized = dataset_name.strip().lower().replace("_", "-").replace(" ", "-")
    return normalized in SSV2_NAMES or "something-something" in normalized


def build_video_transform(
    image_size: int = 224,
    is_train: bool = True,
    *,
    dataset_name: Optional[str] = None,
    horizontal_flip: Optional[bool] = None,
) -> VideoClipTransform:
    """Build a clip transform; horizontal flipping is always off for SSv2."""

    if is_ssv2_dataset(dataset_name):
        allow_flip = False
    elif horizontal_flip is None:
        allow_flip = True
    else:
        allow_flip = bool(horizontal_flip)
    return VideoClipTransform(
        image_size=image_size,
        is_train=is_train,
        horizontal_flip=allow_flip,
    )


class DirectVideoDataset(Dataset):
    """Dataset backed by direct video decoding instead of extracted frames."""

    def __init__(
        self,
        annotation_file: PathLike,
        *,
        num_frames: int = 8,
        transform: Optional[Callable[[Any], Any]] = None,
        is_train: bool = False,
        dataset_name: Optional[str] = None,
        random_select: Optional[bool] = None,
        temporal_window: TemporalWindow = None,
        image_size: int = 224,
        path_root: Optional[PathLike] = None,
        jpeg_cache: Optional[Union[PathLike, JPEGFrameCache]] = None,
        seed: Optional[int] = None,
        return_record: bool = False,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.records = read_annotations(self.annotation_file)
        self.num_frames = int(num_frames)
        self.is_train = bool(is_train)
        self.dataset_name = dataset_name
        self.random_select = self.is_train if random_select is None else bool(random_select)
        self.temporal_window = temporal_window
        self.image_size = int(image_size)
        self.path_root = Path(path_root) if path_root is not None else None
        self.jpeg_cache = jpeg_cache
        self.seed = seed
        self.epoch = 0
        self.return_record = return_record
        self.transform = transform or build_video_transform(
            image_size=self.image_size,
            is_train=self.is_train,
            dataset_name=dataset_name,
        )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Select reproducible epoch-specific sampling and augmentation seeds."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def _seed_for_index(self, index: int) -> Optional[int]:
        if self.seed is not None:
            return (int(self.seed) + 1_000_003 * self.epoch + int(index)) % (2**32)
        if not self.random_select:
            return None
        # torch initial_seed is worker-specific under a DataLoader.
        return int(torch.initial_seed() + index) % (2**32)

    def __getitem__(self, index: int) -> Any:
        record = self.records[index]
        video_path = _resolve_record_path(record.path, self.path_root)
        clip = load_video_frames_decord(
            video_path,
            self.num_frames,
            self.transform,
            random_select=self.random_select,
            temporal_window=self.temporal_window,
            seed=self._seed_for_index(index),
            jpeg_cache=self.jpeg_cache,
        )
        if self.return_record:
            return {"video": clip, "label": record.label, "record": record}
        return clip, record.label


FSARVideoDataset = DirectVideoDataset
VideoDataset = DirectVideoDataset


def _load_label_map(label_map: Optional[Union[PathLike, Mapping[str, int]]]) -> Optional[Dict[str, int]]:
    if label_map is None:
        return None
    if isinstance(label_map, Mapping):
        return {str(key): int(value) for key, value in label_map.items()}
    with Path(label_map).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {str(label): index for index, label in enumerate(payload)}
    if not isinstance(payload, dict):
        raise ValueError("label map must be a JSON object or list")
    return {str(key): int(value) for key, value in payload.items()}


def _find_video(video_root: Path, source_path: str) -> Path:
    candidate = Path(source_path).expanduser()
    if not candidate.is_absolute():
        candidate = video_root / candidate
    if candidate.is_file():
        return candidate
    if not candidate.suffix:
        for extension in VIDEO_EXTENSIONS:
            with_extension = candidate.with_suffix(extension)
            if with_extension.is_file():
                return with_extension
    return candidate


def _manifest_rows(manifest: PathLike) -> Iterator[Tuple[str, Optional[int], Any]]:
    """Yield ``path, optional_num_frames, label`` from text or JSON manifests."""

    path = Path(manifest)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("samples", payload.get("videos", payload.get("data", payload)))
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must contain a list of sample objects")
        for item in payload:
            if not isinstance(item, Mapping):
                raise ValueError("JSON manifest entries must be objects")
            source_path = next(
                (item[key] for key in ("path", "video_path", "video", "id") if key in item),
                None,
            )
            label = next(
                (item[key] for key in ("label", "class_id", "class", "template") if key in item),
                None,
            )
            if source_path is None or label is None:
                raise ValueError("manifest entry requires path/id and label/class")
            yield str(source_path), int(item["num_frames"]) if "num_frames" in item else None, label
        return

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Prefer the full annotation format if both rightmost fields are ints.
            fields = stripped.rsplit(maxsplit=2)
            if len(fields) == 3:
                try:
                    yield fields[0], int(fields[1]), int(fields[2])
                    continue
                except ValueError:
                    pass
            pair = stripped.rsplit(maxsplit=1)
            if len(pair) != 2:
                raise ValueError("{}:{}: expected 'path label'".format(path, line_number))
            yield pair[0], None, pair[1]


def records_from_manifest(
    manifest: PathLike,
    video_root: PathLike,
    *,
    label_map: Optional[Union[PathLike, Mapping[str, int]]] = None,
    frame_count_fn: Callable[[PathLike], int] = probe_video_num_frames,
    absolute_paths: bool = True,
    require_paths: bool = True,
) -> List[AnnotationRecord]:
    """Resolve a split manifest and obtain missing frame counts on demand."""

    root = Path(video_root).expanduser()
    labels = _load_label_map(label_map)
    records: List[AnnotationRecord] = []
    for source_path, num_frames, raw_label in _manifest_rows(manifest):
        video_path = _find_video(root, source_path)
        if require_paths and not video_path.is_file():
            raise FileNotFoundError("video not found: {}".format(video_path))
        if labels is None:
            try:
                label = int(raw_label)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "non-numeric label {!r} requires --label-map".format(raw_label)
                ) from exc
        else:
            label_key = str(raw_label)
            if label_key in labels:
                label = labels[label_key]
            else:
                try:
                    label = int(raw_label)
                except (TypeError, ValueError) as exc:
                    raise KeyError(
                        "label {!r} is absent from label map".format(label_key)
                    ) from exc
        if num_frames is None:
            num_frames = int(frame_count_fn(video_path))
        stored_path = str(video_path.resolve()) if absolute_paths else os.path.relpath(video_path, root)
        records.append(AnnotationRecord(stored_path, int(num_frames), int(label)))
    return records


def _discover_split_records(
    split_root: Path,
    video_root: Path,
    label_map: Mapping[str, int],
    frame_count_fn: Callable[[PathLike], int],
    absolute_paths: bool,
) -> List[AnnotationRecord]:
    records: List[AnnotationRecord] = []
    for video_path in sorted(split_root.rglob("*")):
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        class_name = video_path.parent.name
        if class_name not in label_map:
            raise KeyError("directory class {!r} is absent from label map".format(class_name))
        stored_path = (
            str(video_path.resolve())
            if absolute_paths
            else os.path.relpath(video_path, video_root)
        )
        records.append(
            AnnotationRecord(
                stored_path,
                int(frame_count_fn(video_path)),
                int(label_map[class_name]),
            )
        )
    return records


def generate_split_annotations(
    video_root: PathLike,
    output_dir: PathLike,
    *,
    split_manifests: Optional[Mapping[str, PathLike]] = None,
    label_map: Optional[Union[PathLike, Mapping[str, int]]] = None,
    frame_count_fn: Callable[[PathLike], int] = probe_video_num_frames,
    absolute_paths: bool = True,
    require_paths: bool = True,
    validate: bool = True,
    require_disjoint_labels: bool = True,
    expected_class_counts: Optional[Mapping[str, int]] = None,
) -> Dict[str, Path]:
    """Generate ``train.txt``, ``val.txt`` and ``test.txt`` annotations.

    With ``split_manifests``, each manifest accepts ``path label`` or the full
    annotation format.  Without manifests, ``video_root/{train,val,test}`` is
    scanned, with class names taken from immediate parent directories.
    """

    root = Path(video_root).expanduser()
    destination = Path(output_dir)
    required_splits = ("train", "val", "test")
    labels = _load_label_map(label_map)
    records_by_split: Dict[str, List[AnnotationRecord]] = {}

    if split_manifests is not None:
        missing = [split for split in required_splits if split not in split_manifests]
        if missing:
            raise ValueError("missing split manifests: {}".format(missing))
        for split in required_splits:
            records_by_split[split] = records_from_manifest(
                split_manifests[split],
                root,
                label_map=labels,
                frame_count_fn=frame_count_fn,
                absolute_paths=absolute_paths,
                require_paths=require_paths,
            )
    else:
        split_roots = {split: root / split for split in required_splits}
        missing = [str(path) for path in split_roots.values() if not path.is_dir()]
        if missing:
            raise FileNotFoundError("missing split directories: {}".format(missing))
        if labels is None:
            class_names = sorted(
                {
                    path.name
                    for split_root in split_roots.values()
                    for path in split_root.iterdir()
                    if path.is_dir()
                }
            )
            labels = {name: index for index, name in enumerate(class_names)}
        for split, split_root in split_roots.items():
            records_by_split[split] = _discover_split_records(
                split_root, root, labels, frame_count_fn, absolute_paths
            )

    outputs = {
        split: write_annotations(records_by_split[split], destination / "{}.txt".format(split))
        for split in required_splits
    }
    if validate:
        report = validate_split_annotations(
            outputs["train"],
            outputs["val"],
            outputs["test"],
            check_paths=require_paths,
            path_root=None if absolute_paths else root,
            require_disjoint_labels=require_disjoint_labels,
            expected_class_counts=expected_class_counts,
        )
        if not report.valid:
            raise ValueError("generated annotations failed validation: {}".format(report.errors))
    return outputs


generate_annotations = generate_split_annotations


def _histogram(values: Sequence[int], bins: Union[int, Sequence[float]]) -> Dict[str, Any]:
    if not values:
        return {"counts": [], "bin_edges": []}
    counts, edges = np.histogram(np.asarray(values), bins=bins)
    return {
        "counts": [int(value) for value in counts],
        "bin_edges": [float(value) for value in edges],
    }


def audit_annotations(
    annotations: Union[
        PathLike,
        Iterable[AnnotationRecord],
        Mapping[str, Union[PathLike, Iterable[AnnotationRecord]]],
    ],
    *,
    histogram_bins: Union[int, Sequence[float]] = 10,
    sample_count: int = 3,
    seed: int = 916,
    check_paths: bool = True,
    path_root: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Return JSON-serializable class counts, frame histogram and samples."""

    if isinstance(annotations, Mapping):
        sources = dict(annotations)
    else:
        sources = {"all": annotations}
    rng = random.Random(seed)
    output: Dict[str, Any] = {"splits": {}, "seed": int(seed)}
    all_records: List[Tuple[str, AnnotationRecord]] = []

    for split, source in sources.items():
        records = (
            read_annotations(source)
            if isinstance(source, (str, os.PathLike))
            else list(source)
        )
        validation = validate_annotations(
            records, check_paths=check_paths, path_root=path_root
        )
        frame_counts = [record.num_frames for record in records]
        class_counts = Counter(record.label for record in records)
        output["splits"][str(split)] = {
            "records": len(records),
            "classes": len(class_counts),
            "class_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
            "frame_counts": {
                "min": min(frame_counts) if frame_counts else None,
                "max": max(frame_counts) if frame_counts else None,
                "mean": float(np.mean(frame_counts)) if frame_counts else None,
                "median": float(np.median(frame_counts)) if frame_counts else None,
                "histogram": _histogram(frame_counts, histogram_bins),
            },
            "validation": validation.to_dict(),
        }
        all_records.extend((str(split), record) for record in records)

    chosen = rng.sample(all_records, min(max(0, sample_count), len(all_records)))
    output["random_samples"] = [
        {"split": split, **asdict(record)} for split, record in chosen
    ]
    output["total_records"] = len(all_records)
    output["valid"] = all(
        entry["validation"]["valid"] for entry in output["splits"].values()
    )
    return output


audit_annotation_file = audit_annotations


def save_audit_previews(
    audit: Mapping[str, Any],
    output_dir: PathLike,
    *,
    path_root: Optional[PathLike] = None,
) -> List[str]:
    """Save one 8-frame contact sheet for every sampled audit record."""

    from PIL import Image, ImageDraw

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for number, sample in enumerate(audit.get("random_samples", []), 1):
        video_path = _resolve_record_path(str(sample["path"]), path_root)
        clip = load_video_frames_decord(video_path, 8, transform=None)
        clip = (clip.clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).numpy())
        thumbnails: List[Image.Image] = []
        for frame in clip:
            image = Image.fromarray(frame, mode="RGB")
            image.thumbnail((224, 224))
            thumbnails.append(image)
        width = sum(image.width for image in thumbnails)
        height = max(image.height for image in thumbnails) + 28
        sheet = Image.new("RGB", (width, height), "white")
        left = 0
        for image in thumbnails:
            sheet.paste(image, (left, 28))
            left += image.width
        ImageDraw.Draw(sheet).text(
            (4, 5),
            "{} label={} frames={}".format(sample["split"], sample["label"], sample["num_frames"]),
            fill="black",
        )
        output_path = destination / "sample_{:02d}.jpg".format(number)
        sheet.save(output_path, quality=92)
        written.append(str(output_path))
    return written


def parse_temporal_window(value: str) -> Tuple[float, float]:
    """Argparse helper accepting ``START,END`` or ``START:END``."""

    fields = re.split(r"[:,]", value)
    if len(fields) != 2:
        raise ValueError("temporal window must be START,END")
    start, end = float(fields[0]), float(fields[1])
    # Validate against an arbitrary positive length; normalized CLI windows are
    # the public protocol.
    _normalise_temporal_window(100, (start, end))
    return start, end


__all__ = [
    "AnnotationRecord",
    "VideoRecord",
    "ValidationReport",
    "VIDEO_EXTENSIONS",
    "parse_annotation_line",
    "parse_annotation",
    "iter_annotations",
    "read_annotations",
    "read_annotation_file",
    "load_annotations",
    "write_annotations",
    "write_annotation_file",
    "validate_annotations",
    "validate_annotation_file",
    "validate_split_annotations",
    "sample_frame_indices",
    "probe_video_num_frames",
    "JPEGFrameCache",
    "load_video_frames_decord",
    "VideoClipTransform",
    "build_video_transform",
    "is_ssv2_dataset",
    "DirectVideoDataset",
    "FSARVideoDataset",
    "VideoDataset",
    "records_from_manifest",
    "generate_split_annotations",
    "generate_annotations",
    "audit_annotations",
    "audit_annotation_file",
    "save_audit_previews",
    "parse_temporal_window",
]
