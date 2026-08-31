"""Version-2 sub-action corpora and leakage-safe candidate selection.

Innovation 2 separates three concerns which were easy to conflate in the
original code:

* generation is text-only and provider agnostic;
* offline text QC may inspect every class, while video QC is guarded by the
  base/validation/test protocol;
* a novel class is adapted online using *support* examples only.

The module intentionally has no dependency on a particular LLM or video
encoder.  Providers, text encoders and support-cost functions are injected by
the caller.  Consequently importing this module never performs a network
request and never loads a model.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import yaml


PathLike = Union[str, Path]
VALID_K = (2, 3, 4, 5)
SCHEMA_VERSION = "2.0"


class CorpusV2Error(ValueError):
    """Raised when a v2 corpus or generation response is invalid."""


class LeakageError(CorpusV2Error):
    """Raised before an operation could consume forbidden video data."""


class ProviderError(CorpusV2Error):
    """Raised when a provider plugin cannot be loaded or parsed."""


class GenerationProvider(Protocol):
    """Minimal provider interface; implementations may wrap any LLM API."""

    def generate(
        self,
        *,
        prompt: str,
        class_name: str,
        k: int,
        sample_index: int,
        seed: int,
        temperature: float,
    ) -> Any:
        """Return K stages, or JSON containing a ``subs``/``stages`` list."""


@dataclass
class StageCandidate:
    """One ordered decomposition of an action class."""

    k: int
    subs: Tuple[str, ...]
    gen_id: str
    recipe_id: str = "default"
    active: bool = True
    qc: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.k = int(self.k)
        self.subs = tuple(str(value).strip() for value in self.subs)
        self.gen_id = str(self.gen_id).strip()
        self.recipe_id = str(self.recipe_id).strip() or "default"
        self.validate()

    def validate(self) -> None:
        if self.k not in VALID_K:
            raise CorpusV2Error(f"candidate K must be one of {VALID_K}, got {self.k}")
        if len(self.subs) != self.k:
            raise CorpusV2Error(
                f"candidate {self.gen_id!r} declares K={self.k} but has "
                f"{len(self.subs)} stages"
            )
        if not self.gen_id:
            raise CorpusV2Error("candidate gen_id must not be empty")
        if any(not value for value in self.subs):
            raise CorpusV2Error(f"candidate {self.gen_id!r} has an empty stage")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageCandidate":
        if not isinstance(value, Mapping):
            raise CorpusV2Error("candidate must be a mapping")
        subs = value.get("subs", value.get("stages"))
        if not isinstance(subs, (list, tuple)):
            raise CorpusV2Error("candidate subs must be a list")
        return cls(
            k=value.get("K", value.get("k", len(subs))),
            subs=tuple(subs),
            gen_id=value.get("gen_id", ""),
            recipe_id=value.get("recipe_id", "default"),
            active=bool(value.get("active", True)),
            qc=copy.deepcopy(dict(value.get("qc", {}))),
            metadata=copy.deepcopy(dict(value.get("metadata", {}))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "K": self.k,
            "subs": list(self.subs),
            "gen_id": self.gen_id,
            "recipe_id": self.recipe_id,
            "active": self.active,
            "qc": copy.deepcopy(self.qc),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class ClassCandidates:
    """All raw candidates for one class, including QC-disabled candidates."""

    candidates: List[StageCandidate]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, *, require_k_coverage: bool = True) -> None:
        if not self.candidates:
            raise CorpusV2Error("every class must contain at least one candidate")
        ids = [candidate.gen_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise CorpusV2Error("gen_id values must be unique within each class")
        for candidate in self.candidates:
            candidate.validate()
        observed = {candidate.k for candidate in self.candidates}
        if require_k_coverage and observed != set(VALID_K):
            raise CorpusV2Error(
                f"candidate set must cover K={VALID_K}; observed {sorted(observed)}"
            )

    @property
    def active_candidates(self) -> List[StageCandidate]:
        return [candidate for candidate in self.candidates if candidate.active]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClassCandidates":
        if not isinstance(value, Mapping):
            raise CorpusV2Error("class entry must be a mapping")
        raw = value.get("candidates")
        if not isinstance(raw, list):
            raise CorpusV2Error("class entry must contain a candidates list")
        return cls(
            candidates=[StageCandidate.from_dict(item) for item in raw],
            metadata=copy.deepcopy(dict(value.get("metadata", {}))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class CorpusV2:
    """Serializable v2 corpus whose classes may select different K online."""

    dataset: str
    classes: Dict[str, ClassCandidates]
    generation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self, *, require_k_coverage: bool = True) -> None:
        if str(self.schema_version) != SCHEMA_VERSION:
            raise CorpusV2Error(
                f"unsupported schema_version {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        if not str(self.dataset).strip():
            raise CorpusV2Error("dataset must not be empty")
        if not self.classes:
            raise CorpusV2Error("corpus must contain at least one class")
        for class_name, entry in self.classes.items():
            if not str(class_name).strip():
                raise CorpusV2Error("class names must not be empty")
            try:
                entry.validate(require_k_coverage=require_k_coverage)
            except CorpusV2Error as exc:
                raise CorpusV2Error(f"class {class_name!r}: {exc}") from exc
        # Fail at serialization time rather than silently persisting objects.
        try:
            json.dumps(self.to_dict(), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise CorpusV2Error(f"corpus metadata is not JSON serializable: {exc}") from exc

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, require_k_coverage: bool = True
    ) -> "CorpusV2":
        if not isinstance(value, Mapping):
            raise CorpusV2Error("corpus root must be a mapping")
        raw_classes = value.get("classes")
        if not isinstance(raw_classes, Mapping):
            raise CorpusV2Error("corpus root must contain a classes mapping")
        corpus = cls(
            dataset=str(value.get("dataset", "")),
            classes={
                str(name): ClassCandidates.from_dict(entry)
                for name, entry in raw_classes.items()
            },
            generation=copy.deepcopy(dict(value.get("generation", {}))),
            metadata=copy.deepcopy(dict(value.get("metadata", {}))),
            schema_version=str(value.get("schema_version", "")),
        )
        corpus.validate(require_k_coverage=require_k_coverage)
        return corpus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "generation": copy.deepcopy(self.generation),
            "metadata": copy.deepcopy(self.metadata),
            "classes": {
                name: entry.to_dict() for name, entry in self.classes.items()
            },
        }

    def clone(self) -> "CorpusV2":
        return CorpusV2.from_dict(self.to_dict(), require_k_coverage=False)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_corpus_v2(
    path: PathLike, *, require_k_coverage: bool = True
) -> CorpusV2:
    """Load JSON or YAML based on suffix and strictly validate the schema."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"v2 corpus not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix.lower() == ".json":
            value = json.load(handle)
        elif source.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(handle)
        else:
            raise CorpusV2Error("v2 corpus suffix must be .json, .yaml or .yml")
    return CorpusV2.from_dict(value, require_k_coverage=require_k_coverage)


def save_corpus_v2(corpus: CorpusV2, path: PathLike) -> Path:
    """Atomically persist a validated corpus as deterministic JSON or YAML."""

    corpus.validate(require_k_coverage=True)
    destination = Path(path)
    if destination.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise CorpusV2Error("v2 corpus suffix must be .json, .yaml or .yml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        if destination.suffix.lower() == ".json":
            json.dump(corpus.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        else:
            yaml.safe_dump(
                corpus.to_dict(), handle, allow_unicode=True, sort_keys=False
            )
    temporary.replace(destination)
    return destination


DEFAULT_PROMPT_TEMPLATE = """Decompose the visual action class {class_name!r} into exactly {k} ordered, observable stages.
Return JSON only as {{\"subs\": [\"stage 1\", ..., \"stage {k}\"]}}.
Each stage must describe visible motion or object state, not intent, emotion, or hidden mental state.
Use concise phrases compatible with: A video of action about {class_name}: <stage>."""


def build_generation_prompt(
    class_name: str, k: int, *, template: str = DEFAULT_PROMPT_TEMPLATE
) -> str:
    if int(k) not in VALID_K:
        raise CorpusV2Error(f"K must be one of {VALID_K}")
    return template.format(class_name=str(class_name), k=int(k))


def generation_manifest(
    class_names: Sequence[str],
    *,
    samples_per_k: int = 3,
    seed: int = 916,
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Dict[str, Any]:
    """Make a dry-run manifest without loading or calling a provider."""

    if samples_per_k < 1:
        raise CorpusV2Error("samples_per_k must be positive")
    jobs: List[Dict[str, Any]] = []
    for class_index, class_name in enumerate(_unique_class_names(class_names)):
        for k in VALID_K:
            for sample_index in range(samples_per_k):
                jobs.append(
                    {
                        "class_name": class_name,
                        "K": k,
                        "sample_index": sample_index,
                        "seed": _job_seed(seed, class_index, k, sample_index),
                        "prompt": build_generation_prompt(class_name, k, template=template),
                    }
                )
    return {"network_requests": 0, "jobs": jobs}


def load_provider(specification: str) -> GenerationProvider:
    """Load ``module:object``; the object may be an instance, class or factory."""

    if ":" not in specification:
        raise ProviderError(
            "provider must use 'python.module:object' syntax; no built-in network "
            "provider is selected implicitly"
        )
    module_name, object_name = specification.rsplit(":", 1)
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, object_name)
        if isinstance(value, type):
            value = value()
        elif callable(value) and not hasattr(value, "generate"):
            value = value()
    except Exception as exc:
        raise ProviderError(f"could not load provider {specification!r}: {exc}") from exc
    if not hasattr(value, "generate") or not callable(value.generate):
        raise ProviderError(
            f"provider {specification!r} must expose a callable generate(...) method"
        )
    return value


def generate_corpus_v2(
    class_names: Sequence[str],
    provider: GenerationProvider,
    *,
    dataset: str,
    samples_per_k: int = 3,
    temperature: float = 0.7,
    seed: int = 916,
    recipe_id: str = "default",
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> CorpusV2:
    """Generate text-only candidates for every K in ``VALID_K``.

    Calls are deterministic with respect to their per-job seeds if the injected
    provider honours ``seed``.  Duplicate decompositions are retained only once;
    each K must still have at least one valid response.
    """

    if samples_per_k < 1:
        raise CorpusV2Error("samples_per_k must be positive")
    if temperature < 0:
        raise CorpusV2Error("temperature must be non-negative")
    names = _unique_class_names(class_names)
    classes: Dict[str, ClassCandidates] = {}
    for class_index, class_name in enumerate(names):
        candidates: List[StageCandidate] = []
        seen: set = set()
        for k in VALID_K:
            successful_for_k = 0
            for sample_index in range(samples_per_k):
                job_seed = _job_seed(seed, class_index, k, sample_index)
                try:
                    raw = provider.generate(
                        prompt=build_generation_prompt(class_name, k, template=template),
                        class_name=class_name,
                        k=k,
                        sample_index=sample_index,
                        seed=job_seed,
                        temperature=temperature,
                    )
                except Exception as exc:
                    raise ProviderError(
                        f"provider call failed for class={class_name!r}, K={k}, "
                        f"sample={sample_index}: {exc}"
                    ) from exc
                subs = _parse_provider_response(raw, k)
                identity = tuple(normalize_text(value) for value in subs)
                if identity in seen:
                    continue
                seen.add(identity)
                successful_for_k += 1
                candidates.append(
                    StageCandidate(
                        k=k,
                        subs=tuple(subs),
                        gen_id=f"{_slug(class_name)}-k{k}-s{sample_index}",
                        recipe_id=recipe_id,
                        metadata={"seed": job_seed, "sample_index": sample_index},
                    )
                )
            if successful_for_k == 0:
                raise ProviderError(
                    f"provider produced no unique valid K={k} candidate for {class_name!r}"
                )
        classes[class_name] = ClassCandidates(candidates)
    corpus = CorpusV2(
        dataset=dataset,
        classes=classes,
        generation={
            "provider": provider.__class__.__module__ + "." + provider.__class__.__name__,
            "recipe_id": recipe_id,
            "temperature": temperature,
            "seed": seed,
            "samples_per_k": samples_per_k,
            "prompt_template": template,
        },
    )
    corpus.validate()
    return corpus


def _parse_provider_response(raw: Any, expected_k: int) -> Tuple[str, ...]:
    if isinstance(raw, str):
        text = raw.strip()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "provider string response must be JSON containing a subs/stages list"
            ) from exc
    if isinstance(raw, Mapping):
        raw = raw.get("subs", raw.get("stages"))
    if not isinstance(raw, (list, tuple)):
        raise ProviderError("provider response must be a stage list or JSON object")
    values = tuple(str(value).strip() for value in raw)
    if len(values) != expected_k or any(not value for value in values):
        raise ProviderError(
            f"provider must return exactly {expected_k} non-empty stages, got {len(values)}"
        )
    return values


@dataclass(frozen=True)
class TextQCConfig:
    min_chars: int = 3
    max_chars: int = 160
    redundancy_threshold: float = 0.85
    banned_terms: Tuple[str, ...] = (
        "intend",
        "intends",
        "want to",
        "decide",
        "believe",
        "feel happy",
        "feel sad",
        "意图",
        "想要",
        "决定",
        "认为",
        "感到",
    )

    def __post_init__(self) -> None:
        if self.min_chars < 1 or self.max_chars < self.min_chars:
            raise CorpusV2Error("invalid text QC length bounds")
        if not 0.0 <= self.redundancy_threshold <= 1.0:
            raise CorpusV2Error("redundancy_threshold must be in [0, 1]")


@dataclass
class TextQCReport:
    classes_checked: List[str]
    candidate_results: Dict[str, Dict[str, List[str]]]

    @property
    def accepted(self) -> int:
        return sum(
            not reasons
            for by_candidate in self.candidate_results.values()
            for reasons in by_candidate.values()
        )

    @property
    def rejected(self) -> int:
        return sum(
            bool(reasons)
            for by_candidate in self.candidate_results.values()
            for reasons in by_candidate.values()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": "all_classes_text_only",
            "classes_checked": list(self.classes_checked),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "candidate_results": copy.deepcopy(self.candidate_results),
        }


def run_text_qc(
    corpus: CorpusV2, config: Optional[TextQCConfig] = None
) -> Tuple[CorpusV2, TextQCReport]:
    """Apply pure-text QC to every class and mark rejected candidates inactive."""

    config = config or TextQCConfig()
    checked = sorted(corpus.classes)
    result = corpus.clone()
    records: Dict[str, Dict[str, List[str]]] = {}
    for class_name in checked:
        records[class_name] = {}
        for candidate in result.classes[class_name].candidates:
            reasons = text_qc_reasons(candidate, config)
            candidate.qc["text"] = {
                "accepted": not reasons,
                "reasons": list(reasons),
            }
            candidate.active = candidate.active and not reasons
            records[class_name][candidate.gen_id] = reasons
    return result, TextQCReport(checked, records)


def text_qc_reasons(candidate: StageCandidate, config: TextQCConfig) -> List[str]:
    reasons: List[str] = []
    normalized = [normalize_text(stage) for stage in candidate.subs]
    for index, (stage, clean) in enumerate(zip(candidate.subs, normalized)):
        if len(stage.strip()) < config.min_chars:
            reasons.append(f"stage_{index}:too_short")
        if len(stage.strip()) > config.max_chars:
            reasons.append(f"stage_{index}:too_long")
        if not clean:
            reasons.append(f"stage_{index}:no_lexical_content")
        for term in config.banned_terms:
            if normalize_text(term) in clean:
                reasons.append(f"stage_{index}:banned_term:{term}")
    if len(normalized) != len(set(normalized)):
        reasons.append("duplicate_stage")
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            similarity = token_jaccard(normalized[left], normalized[right])
            if similarity >= config.redundancy_threshold:
                reasons.append(
                    f"redundant_stages:{left},{right}:{similarity:.3f}"
                )
    return sorted(set(reasons))


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", value.lower()))


def _text_tokens(value: str) -> set:
    return set(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", value.lower()))


def token_jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = _text_tokens(left), _text_tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


_BASE_NAMES = {"base", "train", "training"}
_VAL_NAMES = {"val", "valid", "validation", "development", "dev"}
_TEST_NAMES = {"test", "novel", "evaluation", "eval"}


def canonical_split(value: str) -> str:
    split = str(value).strip().lower()
    if split in _BASE_NAMES:
        return "base"
    if split in _VAL_NAMES:
        return "validation"
    if split in _TEST_NAMES:
        return "test"
    raise LeakageError(f"unknown split {value!r}; refusing video access by default")


def assert_video_qc_access(
    class_names: Iterable[str],
    split_by_class: Mapping[str, str],
    *,
    purpose: str,
) -> None:
    """Fail closed according to the Innovation-2 video-use protocol.

    ``candidate_filter`` permits base videos only. ``recipe_selection`` permits
    validation videos only. Test/novel video is forbidden for both.
    """

    if purpose not in {"candidate_filter", "recipe_selection"}:
        raise LeakageError(f"unknown video QC purpose {purpose!r}")
    expected = "base" if purpose == "candidate_filter" else "validation"
    for class_name in class_names:
        if class_name not in split_by_class:
            raise LeakageError(
                f"class {class_name!r} has no declared split; refusing video access"
            )
        actual = canonical_split(split_by_class[class_name])
        if actual == "test":
            raise LeakageError(
                f"test/novel video for {class_name!r} must never be used by offline QC"
            )
        if actual != expected:
            raise LeakageError(
                f"{purpose} may use {expected} classes only; {class_name!r} is {actual}"
            )


@dataclass
class VideoQCReport:
    classes_checked: List[str]
    retained_gen_ids: Dict[str, List[str]]
    score_direction: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_base_video_qc(
    corpus: CorpusV2,
    scores_by_class: Mapping[str, Mapping[str, float]],
    split_by_class: Mapping[str, str],
    *,
    top_n: int = 2,
    lower_is_better: bool = True,
) -> Tuple[CorpusV2, VideoQCReport]:
    """Filter only base-class candidates using precomputed video-fit scores.

    The function consumes scores rather than videos, allowing the feature/OT
    pipeline to live elsewhere. Passing a validation or test class is rejected
    *before* any score is applied.
    """

    if top_n < 1:
        raise CorpusV2Error("top_n must be positive")
    assert_video_qc_access(
        scores_by_class.keys(), split_by_class, purpose="candidate_filter"
    )
    result = corpus.clone()
    retained: Dict[str, List[str]] = {}
    for class_name, candidate_scores in scores_by_class.items():
        if class_name not in result.classes:
            raise CorpusV2Error(f"video QC class not present in corpus: {class_name!r}")
        active = result.classes[class_name].active_candidates
        known_ids = {candidate.gen_id for candidate in active}
        unknown = set(candidate_scores) - known_ids
        if unknown:
            raise CorpusV2Error(
                f"video QC scores contain unknown/inactive gen_id(s) for {class_name!r}: "
                f"{sorted(unknown)}"
            )
        missing = known_ids - set(candidate_scores)
        if missing:
            raise CorpusV2Error(
                f"video QC scores are incomplete for {class_name!r}: {sorted(missing)}"
            )
        for score in candidate_scores.values():
            if not math.isfinite(float(score)):
                raise CorpusV2Error("video QC scores must be finite")
        direction = 1.0 if lower_is_better else -1.0
        ranked = sorted(
            active,
            key=lambda candidate: (
                direction * float(candidate_scores[candidate.gen_id]),
                candidate.k,
                candidate.gen_id,
            ),
        )
        keep = {candidate.gen_id for candidate in ranked[:top_n]}
        retained[class_name] = sorted(keep)
        for candidate in result.classes[class_name].candidates:
            if not candidate.active:
                continue
            accepted = candidate.gen_id in keep
            candidate.active = accepted
            candidate.qc["base_video"] = {
                "accepted": accepted,
                "score": float(candidate_scores[candidate.gen_id]),
                "lower_is_better": lower_is_better,
            }
    return result, VideoQCReport(
        sorted(scores_by_class),
        retained,
        "lower_is_better" if lower_is_better else "higher_is_better",
    )


@dataclass(frozen=True)
class RecipeSelection:
    recipe_id: str
    mean_score: float
    per_class_scores: Dict[str, float]


def select_recipe_on_validation(
    scores_by_class: Mapping[str, Mapping[str, float]],
    split_by_class: Mapping[str, str],
    *,
    lower_is_better: bool = False,
    tie_tolerance: float = 1e-12,
) -> RecipeSelection:
    """Select one generation recipe using validation-class video scores only."""

    if not scores_by_class:
        raise CorpusV2Error("validation recipe scores must not be empty")
    assert_video_qc_access(
        scores_by_class.keys(), split_by_class, purpose="recipe_selection"
    )
    recipe_ids = None
    for class_name, scores in scores_by_class.items():
        if not scores:
            raise CorpusV2Error(f"no recipe scores for {class_name!r}")
        if not all(math.isfinite(float(value)) for value in scores.values()):
            raise CorpusV2Error("recipe scores must be finite")
        ids = set(scores)
        recipe_ids = ids if recipe_ids is None else recipe_ids & ids
    if not recipe_ids:
        raise CorpusV2Error("validation classes have no common recipe_id")
    aggregate = {
        recipe: mean(
            float(scores_by_class[class_name][recipe])
            for class_name in scores_by_class
        )
        for recipe in recipe_ids
    }
    best_value = (
        min(aggregate.values()) if lower_is_better else max(aggregate.values())
    )
    tied = [
        recipe
        for recipe, value in aggregate.items()
        if abs(value - best_value) <= tie_tolerance
    ]
    chosen = min(tied)
    return RecipeSelection(
        chosen,
        aggregate[chosen],
        {
            class_name: float(scores_by_class[class_name][chosen])
            for class_name in scores_by_class
        },
    )


@dataclass(frozen=True)
class CandidateSelection:
    class_name: str
    candidate: StageCandidate
    mean_support_cost: float
    support_count: int


def select_candidate_from_support(
    class_name: str,
    candidates: Sequence[StageCandidate],
    support_items: Iterable[Any],
    cost_fn: Callable[[Any, StageCandidate], Any],
    *,
    tie_tolerance: float = 1e-8,
) -> CandidateSelection:
    """Choose a candidate from support examples only; ties prefer smaller K.

    There is deliberately no query argument. ``cost_fn`` should return the OT
    transport cost (lower is better) for one support item and one candidate.
    """

    support = list(support_items)
    if not support:
        raise CorpusV2Error("online candidate selection requires support examples")
    active = [candidate for candidate in candidates if candidate.active]
    if not active:
        raise CorpusV2Error(f"class {class_name!r} has no active candidates")
    costs: Dict[str, float] = {}
    for candidate in active:
        values = [_finite_scalar(cost_fn(item, candidate)) for item in support]
        costs[candidate.gen_id] = mean(values)
    minimum = min(costs.values())
    tied = [
        candidate
        for candidate in active
        if abs(costs[candidate.gen_id] - minimum) <= tie_tolerance
    ]
    chosen = min(tied, key=lambda candidate: (candidate.k, candidate.gen_id))
    return CandidateSelection(
        class_name, chosen, costs[chosen.gen_id], len(support)
    )


def select_episode_candidates(
    corpus: CorpusV2,
    support_by_class: Mapping[str, Iterable[Any]],
    cost_fn: Callable[[Any, StageCandidate], Any],
    *,
    tie_tolerance: float = 1e-8,
) -> Dict[str, CandidateSelection]:
    """Online ragged-K selection for every class represented by the support set."""

    output: Dict[str, CandidateSelection] = {}
    for class_name, support_items in support_by_class.items():
        if class_name not in corpus.classes:
            raise CorpusV2Error(f"support class missing from corpus: {class_name!r}")
        output[class_name] = select_candidate_from_support(
            class_name,
            corpus.classes[class_name].candidates,
            support_items,
            cost_fn,
            tie_tolerance=tie_tolerance,
        )
    return output


@dataclass(frozen=True)
class FrozenCandidate:
    candidate: StageCandidate
    encoded_stages: Any


@dataclass(frozen=True)
class FrozenCandidateBank:
    """Test-time text-only precomputation; it contains no video or labels."""

    corpus_fingerprint: str
    by_class: Dict[str, Tuple[FrozenCandidate, ...]]

    def ragged_k(self) -> Dict[str, Tuple[int, ...]]:
        return {
            name: tuple(item.candidate.k for item in candidates)
            for name, candidates in self.by_class.items()
        }


def freeze_candidate_bank(
    corpus: CorpusV2, encode_stages: Callable[[Tuple[str, ...]], Any]
) -> FrozenCandidateBank:
    """Precompute active candidate text features without accepting video inputs."""

    frozen: Dict[str, Tuple[FrozenCandidate, ...]] = {}
    for class_name, entry in corpus.classes.items():
        frozen[class_name] = tuple(
            FrozenCandidate(candidate, encode_stages(candidate.subs))
            for candidate in entry.active_candidates
        )
    return FrozenCandidateBank(corpus.fingerprint(), frozen)


def select_frozen_from_support(
    class_name: str,
    candidates: Sequence[FrozenCandidate],
    support_items: Iterable[Any],
    cost_fn: Callable[[Any, Any], Any],
    *,
    tie_tolerance: float = 1e-8,
) -> CandidateSelection:
    """Select from a frozen text bank using support videos and no query data."""

    by_id = {item.candidate.gen_id: item for item in candidates}

    def wrapped(item: Any, candidate: StageCandidate) -> Any:
        return cost_fn(item, by_id[candidate.gen_id].encoded_stages)

    return select_candidate_from_support(
        class_name,
        [item.candidate for item in candidates],
        support_items,
        wrapped,
        tie_tolerance=tie_tolerance,
    )


def _finite_scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if not math.isfinite(result):
        raise CorpusV2Error("support-fit costs must be finite scalars")
    return result


def _unique_class_names(class_names: Sequence[str]) -> List[str]:
    names = [str(name).strip() for name in class_names]
    if not names or any(not name for name in names):
        raise CorpusV2Error("class names must be non-empty")
    if len(names) != len(set(names)):
        raise CorpusV2Error("class names must be unique")
    return names


def _job_seed(seed: int, class_index: int, k: int, sample_index: int) -> int:
    return int(seed) + class_index * 10_000 + k * 100 + sample_index


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if normalized:
        return normalized
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


__all__ = [
    "CorpusV2Error",
    "LeakageError",
    "ProviderError",
    "GenerationProvider",
    "StageCandidate",
    "ClassCandidates",
    "CorpusV2",
    "VALID_K",
    "SCHEMA_VERSION",
    "load_corpus_v2",
    "save_corpus_v2",
    "DEFAULT_PROMPT_TEMPLATE",
    "build_generation_prompt",
    "generation_manifest",
    "load_provider",
    "generate_corpus_v2",
    "TextQCConfig",
    "TextQCReport",
    "run_text_qc",
    "text_qc_reasons",
    "normalize_text",
    "token_jaccard",
    "canonical_split",
    "assert_video_qc_access",
    "VideoQCReport",
    "apply_base_video_qc",
    "RecipeSelection",
    "select_recipe_on_validation",
    "CandidateSelection",
    "select_candidate_from_support",
    "select_episode_candidates",
    "FrozenCandidate",
    "FrozenCandidateBank",
    "freeze_candidate_bank",
    "select_frozen_from_support",
]
