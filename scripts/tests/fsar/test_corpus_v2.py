from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from fsar.corpus_v2 import (
    CandidateSelection,
    ClassCandidates,
    CorpusV2,
    CorpusV2Error,
    LeakageError,
    ProviderError,
    StageCandidate,
    TextQCConfig,
    apply_base_video_qc,
    freeze_candidate_bank,
    generate_corpus_v2,
    generation_manifest,
    load_corpus_v2,
    load_provider,
    run_text_qc,
    save_corpus_v2,
    select_candidate_from_support,
    select_episode_candidates,
    select_frozen_from_support,
    select_recipe_on_validation,
)


def candidate(k: int, suffix: str, *, active: bool = True) -> StageCandidate:
    return StageCandidate(
        k=k,
        subs=tuple(f"visible motion {suffix} phase {index}" for index in range(k)),
        gen_id=f"{suffix}-k{k}",
        active=active,
    )


def corpus_two_classes() -> CorpusV2:
    return CorpusV2(
        dataset="toy",
        classes={
            name: ClassCandidates([candidate(k, name) for k in (2, 3, 4, 5)])
            for name in ("base_action", "novel_action")
        },
        generation={"seed": 7},
    )


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_v2_schema_round_trip_preserves_variable_k_and_qc(
    tmp_path: Path, suffix: str
) -> None:
    corpus = corpus_two_classes()
    corpus.classes["base_action"].candidates[0].qc = {"text": {"accepted": True}}
    path = tmp_path / f"corpus{suffix}"
    save_corpus_v2(corpus, path)
    loaded = load_corpus_v2(path)

    assert loaded.fingerprint() == corpus.fingerprint()
    assert [item.k for item in loaded.classes["base_action"].candidates] == [2, 3, 4, 5]
    assert loaded.classes["base_action"].candidates[0].qc["text"]["accepted"]


def test_schema_requires_all_k_and_exact_stage_count() -> None:
    incomplete = CorpusV2(
        dataset="toy", classes={"a": ClassCandidates([candidate(2, "a")])}
    )
    with pytest.raises(CorpusV2Error, match="cover K"):
        incomplete.validate()
    with pytest.raises(CorpusV2Error, match="declares K"):
        StageCandidate(k=3, subs=("one", "two"), gen_id="bad")


class FakeProvider:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, **job):
        self.calls.append(job)
        return {
            "subs": [
                f"observable {job['class_name']} step {index} sample {job['sample_index']}"
                for index in range(job["k"])
            ]
        }


def test_provider_agnostic_generation_and_network_free_manifest() -> None:
    provider = FakeProvider()
    manifest = generation_manifest(["pick up"], samples_per_k=2, seed=11)
    assert manifest["network_requests"] == 0
    assert len(manifest["jobs"]) == 8
    assert provider.calls == []

    corpus = generate_corpus_v2(
        ["pick up"], provider, dataset="toy", samples_per_k=2, seed=11
    )
    assert len(provider.calls) == 8
    assert len(corpus.classes["pick up"].candidates) == 8
    assert {value.k for value in corpus.classes["pick up"].candidates} == {2, 3, 4, 5}
    assert all("prompt" in call and "seed" in call for call in provider.calls)


def test_provider_failures_are_explicit_and_no_implicit_provider() -> None:
    with pytest.raises(ProviderError, match="module:object"):
        load_provider("some-provider-without-object")

    class BadProvider:
        def generate(self, **_):
            return "not JSON"

    with pytest.raises(ProviderError, match="must be JSON"):
        generate_corpus_v2(["a"], BadProvider(), dataset="toy", samples_per_k=1)


def test_text_qc_covers_every_class_and_keeps_rejection_auditable() -> None:
    corpus = corpus_two_classes()
    bad = corpus.classes["novel_action"].candidates[0]
    bad.subs = ("intends to win", "intends to win")
    filtered, report = run_text_qc(corpus, TextQCConfig())

    assert report.classes_checked == ["base_action", "novel_action"]
    assert report.rejected == 1
    rejected = filtered.classes["novel_action"].candidates[0]
    assert not rejected.active
    assert rejected.qc["text"]["reasons"]
    # Raw candidate remains in the serialized schema for reproducibility.
    filtered.validate(require_k_coverage=True)


def test_base_video_qc_rejects_validation_and_test_before_filtering() -> None:
    corpus = corpus_two_classes()
    base_scores = {
        item.gen_id: float(item.k) for item in corpus.classes["base_action"].candidates
    }
    filtered, report = apply_base_video_qc(
        corpus,
        {"base_action": base_scores},
        {"base_action": "train", "novel_action": "test"},
        top_n=1,
    )
    assert [item.k for item in filtered.classes["base_action"].active_candidates] == [2]
    assert report.classes_checked == ["base_action"]

    novel_scores = {
        item.gen_id: 0.0 for item in corpus.classes["novel_action"].candidates
    }
    with pytest.raises(LeakageError, match="must never"):
        apply_base_video_qc(
            corpus,
            {"novel_action": novel_scores},
            {"novel_action": "test"},
        )
    with pytest.raises(LeakageError, match="base classes only"):
        apply_base_video_qc(
            corpus,
            {"novel_action": novel_scores},
            {"novel_action": "validation"},
        )


def test_validation_video_selects_recipe_only_and_never_accepts_test() -> None:
    scores = {
        "val_a": {"recipe_a": 0.7, "recipe_b": 0.8},
        "val_b": {"recipe_a": 0.9, "recipe_b": 0.8},
    }
    # Mean tie is deterministic by recipe id.
    choice = select_recipe_on_validation(
        scores, {"val_a": "val", "val_b": "validation"}
    )
    assert choice.recipe_id == "recipe_a"
    with pytest.raises(LeakageError, match="must never"):
        select_recipe_on_validation(
            {"held_out": {"recipe_a": 1.0}}, {"held_out": "test"}
        )


def test_support_only_online_selection_tie_prefers_smaller_k_and_is_ragged() -> None:
    corpus = corpus_two_classes()

    def same_cost(_support, _candidate):
        return 1.0

    result = select_candidate_from_support(
        "novel_action",
        corpus.classes["novel_action"].candidates,
        ["support-video"],
        same_cost,
    )
    assert isinstance(result, CandidateSelection)
    assert result.candidate.k == 2
    assert "query" not in inspect.signature(select_candidate_from_support).parameters

    def class_dependent_cost(support_value, value):
        target_k = support_value
        return abs(value.k - target_k)

    selections = select_episode_candidates(
        corpus,
        {"base_action": [3], "novel_action": [5]},
        class_dependent_cost,
    )
    assert selections["base_action"].candidate.k == 3
    assert selections["novel_action"].candidate.k == 5
    # Results are a ragged list of stages, not a padded/stacked tensor.
    assert [len(value.candidate.subs) for value in selections.values()] == [3, 5]


def test_frozen_bank_precomputes_text_only_and_selects_with_support() -> None:
    corpus = corpus_two_classes()
    encoded_inputs = []

    def encoder(stages):
        encoded_inputs.append(stages)
        return {"K": len(stages)}

    bank = freeze_candidate_bank(corpus, encoder)
    assert len(encoded_inputs) == 8
    assert all(isinstance(value, tuple) for value in encoded_inputs)
    assert bank.ragged_k()["novel_action"] == (2, 3, 4, 5)

    selected = select_frozen_from_support(
        "novel_action",
        bank.by_class["novel_action"],
        ["support-only"],
        lambda _support, encoded: abs(encoded["K"] - 4),
    )
    assert selected.candidate.k == 4
    assert "query" not in inspect.signature(select_frozen_from_support).parameters


def test_video_score_map_must_be_complete_and_finite() -> None:
    corpus = corpus_two_classes()
    with pytest.raises(CorpusV2Error, match="incomplete"):
        apply_base_video_qc(
            corpus,
            {"base_action": {"base_action-k2": 0.0}},
            {"base_action": "base"},
        )


def test_json_schema_uses_uppercase_k_and_ordered_subs(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    save_corpus_v2(corpus_two_classes(), path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    item = raw["classes"]["base_action"]["candidates"][0]
    assert item["K"] == 2
    assert item["subs"] == [
        "visible motion base_action phase 0",
        "visible motion base_action phase 1",
    ]
