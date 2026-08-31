"""Training, evaluation, and paired order-diagnostic loops for episodic FSAR."""

from __future__ import annotations

import csv
import json
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from fsar.diagnostics import ConditionAccumulator, summarize_mean_ci95
from fsar.fusion import (
    AlphaSelection,
    fuse_scores,
    kip_hard_gate,
    select_alpha_from_cached_scores,
)
from fsar.order import (
    NON_IDENTITY_STAGE_PERMUTATIONS,
    OrderContrastiveLoss,
    STAGE_PERMUTATIONS,
    all_permutation_scores,
    make_frame_permutation,
    permute_query_frames,
)


def cuda_amp_enabled(device: torch.device, amp: bool = False) -> bool:
    """Return whether CUDA float16 autocast is active for this invocation."""

    return bool(amp) and device.type == "cuda"


def _autocast_context(device: torch.device, amp: bool = False):
    """Create a CUDA float16 autocast context, or a no-op on CPU/when disabled."""

    if not cuda_amp_enabled(device, amp):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def create_grad_scaler(device: torch.device, amp: bool = False):
    """Build a version-compatible CUDA GradScaler with safe CPU fallback."""

    enabled = cuda_amp_enabled(device, amp)
    modern_amp = getattr(torch, "amp", None)
    if modern_amp is not None and hasattr(modern_amp, "GradScaler"):
        try:
            return modern_amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover - transitional PyTorch signatures
            return modern_amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def episode_targets(n_way: int, n_query: int, device: torch.device) -> torch.Tensor:
    """Class-major query targets matching the repository episode layout."""

    return torch.arange(n_way, device=device, dtype=torch.long).repeat_interleave(n_query)


def unpack_episode(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        return batch["images"], batch["labels"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("episode loader must yield (images, labels) or an images/labels mapping")


def _accuracy(scores: torch.Tensor, targets: torch.Tensor) -> float:
    return float((scores.argmax(dim=1) == targets).float().mean().item())


def save_transport_heatmaps(
    transport: Any,
    output_dir: str | Path,
    *,
    episode_index: int,
    max_query_class_pairs: int = 25,
) -> Sequence[Path]:
    """Persist OT frame-by-stage plans as lossless PNG heatmaps and tensors."""

    if transport is None:
        return ()
    output = Path(output_dir) / "episode_{:05d}".format(episode_index)
    output.mkdir(parents=True, exist_ok=True)
    if isinstance(transport, tuple):
        plans = [item.plan[:, 0] for item in transport]  # C * [Q,T,Kc]
        pairs = (
            (query_index, class_index, plans[class_index][query_index])
            for class_index in range(len(plans))
            for query_index in range(plans[class_index].shape[0])
        )
    else:
        plan = transport.plan
        pairs = (
            (query_index, class_index, plan[query_index, class_index])
            for query_index in range(plan.shape[0])
            for class_index in range(plan.shape[1])
        )
    from PIL import Image

    saved = []
    metadata = []
    for pair_index, (query_index, class_index, value) in enumerate(pairs):
        if pair_index >= max_query_class_pairs:
            break
        array = value.detach().float().cpu().numpy()
        minimum, maximum = float(array.min()), float(array.max())
        normalized = (array - minimum) / max(maximum - minimum, 1.0e-12)
        rgb = np.stack(
            (
                normalized,
                1.0 - np.abs(2.0 * normalized - 1.0),
                1.0 - normalized,
            ),
            axis=-1,
        )
        image = Image.fromarray((255.0 * rgb).round().astype(np.uint8), mode="RGB")
        image = image.resize((array.shape[1] * 48, array.shape[0] * 48), resample=Image.Resampling.NEAREST)
        destination = output / "q{:03d}_c{:02d}.png".format(query_index, class_index)
        image.save(destination)
        np.save(output / "q{:03d}_c{:02d}.npy".format(query_index, class_index), array)
        saved.append(destination)
        metadata.append(
            {
                "query": query_index,
                "class": class_index,
                "shape": list(array.shape),
                "minimum": minimum,
                "maximum": maximum,
                "png": destination.name,
            }
        )
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return tuple(saved)


def _correct_score_and_margin(scores: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    correct = scores.gather(1, targets[:, None]).squeeze(1)
    masked = scores.clone()
    masked.scatter_(1, targets[:, None], float("-inf"))
    margin = correct - masked.max(dim=1).values
    return float(correct.mean().item()), float(margin.mean().item())


def compute_episode_loss(
    output: Mapping[str, Any],
    targets: torch.Tensor,
    *,
    logit_scale: float = 64.0,
    order_weight: float = 0.0,
    order_mode: str = "margin",
    order_margin: float = 0.1,
    order_temperature: float = 0.07,
    negative_orders: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute paper CE plus optional innovation-3 order contrastive loss."""

    final_logits = output["final_logits"]
    ce = F.cross_entropy(final_logits * float(logit_scale), targets)
    zero = ce.new_zeros(())
    order_loss = zero
    if order_weight > 0:
        features = output.get("features")
        if features is None:
            raise ValueError("return_aux=True is required when order regularisation is enabled")
        orders = [STAGE_PERMUTATIONS[0]]
        orders.extend(negative_orders or NON_IDENTITY_STAGE_PERMUTATIONS)
        permutation_scores = output.get("order_permutation_scores")
        if permutation_scores is None:
            if not torch.is_tensor(features.stage_text):
                raise ValueError("order regularisation needs a shared K within the episode")
            permutation_scores = all_permutation_scores(
                features.aligned_query,
                features.stage_text,
                orders=orders,
            )
        criterion = OrderContrastiveLoss(
            mode=order_mode,
            margin=order_margin,
            temperature=order_temperature,
        )
        order_loss = criterion.from_permutation_scores(permutation_scores, targets, identity_index=0)
    total = ce + float(order_weight) * order_loss
    return {"loss": total, "loss_ce": ce, "loss_order": order_loss}


def train_one_epoch(
    loader: Iterable[Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool = False,
    scaler: Optional[Any] = None,
    **loss_kwargs: Any,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "loss_ce": 0.0, "loss_order": 0.0, "accuracy": 0.0}
    count = 0
    for batch in loader:
        images, labels = unpack_episode(batch)
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            output = model(images, labels, return_aux=True)
            if float(loss_kwargs.get("order_weight", 0.0)) > 0:
                features = output["features"]
                orders = [STAGE_PERMUTATIONS[0]]
                orders.extend(
                    loss_kwargs.get("negative_orders")
                    or NON_IDENTITY_STAGE_PERMUTATIONS
                )
                output["order_permutation_scores"] = torch.stack(
                    [
                        model.score_episode(features, stage_permutation=order)[
                            "semantic_logits"
                        ]
                        for order in orders
                    ],
                    dim=0,
                )
            targets = episode_targets(model.n_way, model.n_query, device)
            losses = compute_episode_loss(output, targets, **loss_kwargs)
        if cuda_amp_enabled(device, amp):
            scaler = scaler or create_grad_scaler(device, amp=True)
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            optimizer.step()
        for key in ("loss", "loss_ce", "loss_order"):
            totals[key] += float(losses[key].detach().item())
        totals["accuracy"] += _accuracy(output["final_logits"], targets)
        count += 1
    if not count:
        raise ValueError("training loader yielded no episodes")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    loader: Iterable[Any],
    model: torch.nn.Module,
    device: torch.device,
    *,
    transport_dir: Optional[str | Path] = None,
    amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    fused, visual, semantic = [], [], []
    for episode_index, batch in enumerate(loader):
        images, labels = unpack_episode(batch)
        with _autocast_context(device, amp):
            output = model(images.to(device), labels.to(device), return_aux=True)
        targets = episode_targets(model.n_way, model.n_query, device)
        fused.append(_accuracy(output["final_logits"], targets))
        visual.append(_accuracy(output["visual_logits"], targets))
        semantic.append(_accuracy(output["semantic_logits"], targets))
        if transport_dir is not None and output.get("transport") is not None:
            save_transport_heatmaps(
                output["transport"], transport_dir, episode_index=episode_index
            )
    if not fused:
        raise ValueError("evaluation loader yielded no episodes")
    result: Dict[str, float] = {}
    for name, values in (("fused", fused), ("visual", visual), ("semantic", semantic)):
        summary = summarize_mean_ci95(values)
        result[f"{name}_accuracy"] = summary.mean
        result[f"{name}_ci95"] = summary.ci95
    result["episodes"] = len(fused)
    return result


def _pseudo_query_view(
    support: torch.Tensor,
    *,
    n_query: int,
    rng: random.Random,
    crop_scale: float,
) -> torch.Tensor:
    """Create class-major temporal/spatial support views without flipping.

    ``support`` is ``[C,shot,T,3,H,W]``. One shot is sampled per class, frame
    indices receive small monotone-preserving offsets, and a shared random crop
    is resized to the original resolution.  No query tensor is accepted.
    """

    if support.ndim != 6:
        raise ValueError("support must have [C,shot,T,3,H,W] shape")
    if not 0 < crop_scale <= 1:
        raise ValueError("crop_scale must be in (0,1]")
    classes, shots, frames, channels, height, width = support.shape
    selected = []
    for class_index in range(classes):
        shot_index = rng.randrange(shots)
        source = support[class_index, shot_index]
        offsets = torch.tensor(
            [rng.choice((-1, 0, 1)) for _ in range(frames)],
            device=source.device,
            dtype=torch.long,
        )
        indices = (torch.arange(frames, device=source.device) + offsets).clamp(0, frames - 1)
        indices = torch.cummax(indices, dim=0).values
        selected.append(source.index_select(0, indices))
    view = torch.stack(selected, dim=0)  # [C,T,3,H,W]
    crop_h = max(1, int(round(height * crop_scale)))
    crop_w = max(1, int(round(width * crop_scale)))
    top = rng.randint(0, height - crop_h) if crop_h < height else 0
    left = rng.randint(0, width - crop_w) if crop_w < width else 0
    cropped = view[..., top : top + crop_h, left : left + crop_w]
    resized = F.interpolate(
        cropped.reshape(classes * frames, channels, crop_h, crop_w),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).reshape(classes, frames, channels, height, width)
    return resized[:, None].expand(classes, n_query, frames, channels, height, width).clone()


@dataclass(frozen=True)
class PseudoValidationResult:
    selection: AlphaSelection
    visual_scores: torch.Tensor
    semantic_scores: torch.Tensor
    targets: torch.Tensor
    view_forward_calls: int
    wall_time_seconds: float

    def as_dict(self) -> Dict[str, Any]:
        value = self.selection.as_dict()
        value.update(
            {
                "view_forward_calls": self.view_forward_calls,
                "wall_time_seconds": self.wall_time_seconds,
                "horizontal_flip": False,
            }
        )
        return value


@torch.no_grad()
def pseudo_validate_episode_alpha(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha_grid: Sequence[float],
    global_alpha: float,
    views: int = 4,
    seed: int = 916,
    mode: str = "probability_product",
    crop_scale: float = 0.9,
    flat_tolerance: float = 1.0e-12,
    amp: bool = False,
) -> PseudoValidationResult:
    """Select alpha using augmented support only, never true query images.

    Each augmented pseudo episode performs one model forward.  The subsequent
    alpha sweep consumes cached branch scores and therefore performs exactly
    zero additional model forwards, as recorded by :class:`AlphaSelection`.
    """

    if images.ndim != 6:
        raise ValueError("images must have [C,support+query,T,3,H,W] shape")
    if views < 1:
        raise ValueError("views must be positive")
    support = images[:, : model.n_support]
    rng = random.Random(int(seed))
    visual_rows, semantic_rows = [], []
    started = time.perf_counter()
    for _ in range(views):
        pseudo_query = _pseudo_query_view(
            support,
            n_query=model.n_query,
            rng=rng,
            crop_scale=crop_scale,
        )
        pseudo_episode = torch.cat((support, pseudo_query), dim=1)
        with _autocast_context(images.device, amp):
            output = model(pseudo_episode, labels, return_aux=True)
        # Cached calibration/fusion math remains in fp32 even when encoders use AMP.
        visual_rows.append(output["visual_logits"].float())
        semantic_rows.append(output["semantic_logits"].float())
    visual = torch.stack(visual_rows, dim=0)
    semantic = torch.stack(semantic_rows, dim=0)
    targets = episode_targets(model.n_way, model.n_query, images.device)
    expanded_targets = targets.unsqueeze(0).expand(views, -1)
    fusion = getattr(model, "fusion_mode", mode)
    if fusion == "legacy_product":
        fusion = mode
    selection = select_alpha_from_cached_scores(
        visual,
        semantic,
        expanded_targets,
        alpha_grid=alpha_grid,
        global_alpha=global_alpha,
        mode=fusion,
        visual_temperature=float(getattr(model, "visual_temperature", 1.0)),
        semantic_temperature=float(getattr(model, "semantic_temperature", 1.0)),
        flat_tolerance=flat_tolerance,
    )
    return PseudoValidationResult(
        selection,
        visual,
        semantic,
        expanded_targets,
        views,
        time.perf_counter() - started,
    )


@torch.no_grad()
def evaluate_adaptive_fusion(
    loader: Iterable[Any],
    model: torch.nn.Module,
    device: torch.device,
    *,
    alpha_grid: Sequence[float],
    global_alpha: float,
    views: int = 4,
    seed: int = 916,
    mode: str = "probability_product",
    flat_tolerance: float = 1.0e-12,
    amp: bool = False,
) -> Dict[str, Any]:
    """Evaluate per-episode alpha beside global-alpha and KIP baselines."""

    model.eval()
    scores = {name: [] for name in ("adaptive", "global_alpha", "kip")}
    alpha_values, pseudo_rows, extra_times = [], [], []
    for episode_index, batch in enumerate(loader):
        images, labels = unpack_episode(batch)
        images, labels = images.to(device), labels.to(device)
        pseudo = pseudo_validate_episode_alpha(
            model,
            images,
            labels,
            alpha_grid=alpha_grid,
            global_alpha=global_alpha,
            views=views,
            seed=seed + episode_index,
            mode=mode,
            flat_tolerance=flat_tolerance,
            amp=amp,
        )
        with _autocast_context(device, amp):
            output = model(images, labels, return_aux=True)
        targets = episode_targets(model.n_way, model.n_query, device)
        visual = output["visual_logits"].float()
        semantic = output["semantic_logits"].float()
        adaptive = fuse_scores(visual, semantic, mode=mode, alpha=pseudo.selection.alpha)
        global_scores = fuse_scores(visual, semantic, mode=mode, alpha=global_alpha)
        kip = kip_hard_gate(visual, semantic).scores
        scores["adaptive"].append(_accuracy(adaptive, targets))
        scores["global_alpha"].append(_accuracy(global_scores, targets))
        scores["kip"].append(_accuracy(kip, targets))
        alpha_values.append(pseudo.selection.alpha)
        pseudo_rows.append(pseudo.as_dict())
        extra_times.append(pseudo.wall_time_seconds)
    if not alpha_values:
        raise ValueError("evaluation loader yielded no episodes")
    result: Dict[str, Any] = {
        "episodes": len(alpha_values),
        "alpha_values": alpha_values,
        "fallback_rate": float(
            np.mean([row["used_fallback"] for row in pseudo_rows])
        ),
        "mean_additional_wall_time_seconds": float(np.mean(extra_times)),
        "pseudo_validation": pseudo_rows,
    }
    for name, values in scores.items():
        summary = summarize_mean_ci95(values)
        result[name] = {"accuracy": summary.mean, "ci95": summary.ci95}
    return result


@torch.no_grad()
def calibrate_global_alpha(
    loader: Iterable[Any],
    model: torch.nn.Module,
    device: torch.device,
    *,
    alpha_grid: Sequence[float],
    mode: str = "probability_product",
    tie_anchor: float = 0.5,
    flat_tolerance: float = 1.0e-12,
    amp: bool = False,
) -> AlphaSelection:
    """Select dataset-level alpha on validation episodes with cached forwards."""

    model.eval()
    visual_rows, semantic_rows, targets_rows = [], [], []
    for batch in loader:
        images, labels = unpack_episode(batch)
        with _autocast_context(device, amp):
            output = model(images.to(device), labels.to(device), return_aux=True)
        visual_rows.append(output["visual_logits"].float())
        semantic_rows.append(output["semantic_logits"].float())
        targets_rows.append(episode_targets(model.n_way, model.n_query, device))
    if not visual_rows:
        raise ValueError("validation loader yielded no episodes")
    visual = torch.stack(visual_rows, dim=0)
    semantic = torch.stack(semantic_rows, dim=0)
    targets = torch.stack(targets_rows, dim=0)
    return select_alpha_from_cached_scores(
        visual,
        semantic,
        targets,
        alpha_grid=alpha_grid,
        global_alpha=tie_anchor,
        mode=mode,
        visual_temperature=float(getattr(model, "visual_temperature", 1.0)),
        semantic_temperature=float(getattr(model, "semantic_temperature", 1.0)),
        flat_tolerance=flat_tolerance,
    )


@torch.no_grad()
def diagnose_order(
    loader: Iterable[Any],
    model: torch.nn.Module,
    device: torch.device,
    *,
    seed: int = 916,
    include_double_reverse: bool = True,
    amp: bool = False,
) -> Dict[str, Any]:
    """Run C0-C5 on one paired episode stream and retain raw evidence."""

    model.eval()
    permutation_names = ["perm_" + "".join(str(i + 1) for i in order) for order in STAGE_PERMUTATIONS]
    text_condition_names = permutation_names
    frame_condition_names = ["frame_reverse", "frame_random"]
    if include_double_reverse:
        frame_condition_names.append("double_reverse")
    all_names = text_condition_names + frame_condition_names
    accumulators = {
        branch: ConditionAccumulator(all_names) for branch in ("fused", "visual", "semantic")
    }
    raw_rows = []

    for episode_index, batch in enumerate(loader):
        images, labels = unpack_episode(batch)
        images = images.to(device)
        labels = labels.to(device)
        targets = episode_targets(model.n_way, model.n_query, device)
        with _autocast_context(device, amp):
            features = model.extract_episode_features(images, labels)
            if not torch.is_tensor(features.stage_text):
                raise ValueError("order diagnosis requires one shared K across episode classes")
            condition_outputs = [
                model.score_episode(features, stage_permutation=order)
                for order in STAGE_PERMUTATIONS
            ]
        perm_scores = torch.stack(
            [output["semantic_logits"].float() for output in condition_outputs], dim=0
        )

        branch_episode: Dict[str, Dict[str, float]] = {
            branch: {} for branch in accumulators
        }
        for order_index, condition in enumerate(text_condition_names):
            condition_output = condition_outputs[order_index]
            sem = condition_output["semantic_logits"]
            vis = condition_output["visual_logits"]
            fused = condition_output["final_logits"]
            branch_episode["fused"][condition] = _accuracy(fused, targets)
            branch_episode["visual"][condition] = _accuracy(vis, targets)
            branch_episode["semantic"][condition] = _accuracy(sem, targets)

        random_order = make_frame_permutation(model.num_frames, "random", seed + episode_index)
        perturbed = {
            "frame_reverse": permute_query_frames(
                images, make_frame_permutation(model.num_frames, "reverse"), model.n_support
            ),
            "frame_random": permute_query_frames(images, random_order, model.n_support),
        }
        for condition, changed_images in perturbed.items():
            with _autocast_context(device, amp):
                changed = model(changed_images, labels, return_aux=True)
            for branch, key in (
                ("fused", "final_logits"),
                ("visual", "visual_logits"),
                ("semantic", "semantic_logits"),
            ):
                branch_episode[branch][condition] = _accuracy(changed[key], targets)

        if include_double_reverse:
            with _autocast_context(device, amp):
                changed = model(
                    perturbed["frame_reverse"],
                    labels,
                    stage_permutation=(2, 1, 0),
                    return_aux=True,
                )
            for branch, key in (
                ("fused", "final_logits"),
                ("visual", "visual_logits"),
                ("semantic", "semantic_logits"),
            ):
                branch_episode[branch]["double_reverse"] = _accuracy(changed[key], targets)

        # Text interventions must reuse the exact visual score matrix.
        visual_values = [branch_episode["visual"][name] for name in text_condition_names]
        if any(value != visual_values[0] for value in visual_values[1:]):
            raise AssertionError("text permutation changed the visual-only result")

        for branch, accumulator in accumulators.items():
            accumulator.add_episode(branch_episode[branch])

        correct_score, margin = _correct_score_and_margin(perm_scores[0], targets)
        row: Dict[str, Any] = {
            "episode": episode_index,
            "random_seed": seed + episode_index,
            "random_frame_order": list(random_order),
            "correct_semantic_score": correct_score,
            "correct_semantic_margin": margin,
        }
        for branch in accumulators:
            for condition, value in branch_episode[branch].items():
                row[f"{branch}_{condition}"] = value
        raw_rows.append(row)

    if not raw_rows:
        raise ValueError("diagnostic loader yielded no episodes")
    identity_name = permutation_names[0]
    non_identity_names = permutation_names[1:]
    summaries: Dict[str, Any] = {}
    for branch, accumulator in accumulators.items():
        summaries[branch] = accumulator.summaries()
        summaries[branch]["order_sensitivity"] = accumulator.order_sensitivity(
            identity_name, non_identity_names
        ).as_dict()
        for condition in all_names:
            summaries[branch][condition]["delta_vs_c0"] = (
                summaries[branch][condition]["mean"]
                - summaries[branch][identity_name]["mean"]
            )
    return {
        "protocol": {
            "seed": seed,
            "episodes": len(raw_rows),
            "identity": identity_name,
            "non_identity_permutations": non_identity_names,
            "frame_random_is_query_only_and_synchronous": True,
        },
        "summary": summaries,
        "episodes": raw_rows,
    }


def save_diagnostics(result: Mapping[str, Any], output_dir: str | Path) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "order_diagnostics.json"
    csv_path = output_dir / "order_diagnostics_episodes.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    rows = list(result.get("episodes", []))
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


__all__ = [
    "compute_episode_loss",
    "calibrate_global_alpha",
    "create_grad_scaler",
    "cuda_amp_enabled",
    "diagnose_order",
    "episode_targets",
    "evaluate",
    "evaluate_adaptive_fusion",
    "PseudoValidationResult",
    "pseudo_validate_episode_alpha",
    "save_diagnostics",
    "save_transport_heatmaps",
    "train_one_epoch",
    "unpack_episode",
]
