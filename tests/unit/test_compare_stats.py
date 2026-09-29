import pytest

from atlasforge.compare.slices import (
    MIN_SLICE_N,
    analyse_slices,
    group_counts,
    slice_value,
    verdict,
)
from atlasforge.compare.stats import is_binary, mcnemar_exact, paired_bootstrap
from atlasforge.eval.dataset import Example


def example(
    i: int = 0, *, reference: str | None = "a b", lang: str | None = None, **meta: object
) -> Example:
    return Example(id=str(i), reference=reference, lang=lang, meta=meta)  # type: ignore[arg-type]


class TestPairedBootstrap:
    def test_identical_inputs_give_zero_width_interval(self) -> None:
        r = paired_bootstrap([0.2, 0.5, 0.9], [0.2, 0.5, 0.9])
        assert (r.mean, r.low, r.high) == (0.0, 0.0, 0.0)

    def test_constant_shift_is_recovered_exactly(self) -> None:
        r = paired_bootstrap([0.0] * 20, [1.0] * 20)
        assert (r.mean, r.low, r.high, r.n) == (1.0, 1.0, 1.0, 20)

    def test_deterministic_for_a_seed(self) -> None:
        base, cand = [0, 1, 0, 1, 1, 0, 0, 1] * 5, [1, 1, 0, 1, 0, 0, 1, 1] * 5
        assert paired_bootstrap(base, cand, seed=3) == paired_bootstrap(base, cand, seed=3)

    def test_different_seed_changes_interval_but_not_mean(self) -> None:
        base, cand = [0, 1, 0, 1, 1, 0, 0, 1] * 5, [1, 1, 0, 1, 0, 0, 1, 1] * 5
        a, b = paired_bootstrap(base, cand, seed=1), paired_bootstrap(base, cand, seed=2)
        assert a.mean == b.mean
        assert (a.low, a.high) != (b.low, b.high)

    def test_interval_contains_mean_and_is_ordered(self) -> None:
        base = [i % 3 / 3 for i in range(60)]
        cand = [(i * 7 % 5) / 5 for i in range(60)]
        r = paired_bootstrap(base, cand)
        assert r.low <= r.mean <= r.high

    def test_mixed_signal_interval_spans_zero(self) -> None:
        base = [0.0, 1.0] * 20
        cand = [1.0, 0.0] * 20
        r = paired_bootstrap(base, cand)
        assert r.low < 0 < r.high

    def test_clear_improvement_excludes_zero(self) -> None:
        base = [0.0] * 30 + [1.0] * 30
        cand = [1.0] * 30 + [1.0] * 30
        r = paired_bootstrap(base, cand)
        assert r.low > 0

    def test_small_batches_do_not_change_correctness(self) -> None:
        # n large enough that the resample matrix is split into row batches
        big = [float(i % 2) for i in range(5000)]
        r = paired_bootstrap(big, [1.0 - v for v in big], n_boot=50, seed=0)
        assert r.n == 5000
        assert r.low <= r.mean <= r.high

    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            (([1.0], [1.0, 2.0]), {}),
            (([], []), {}),
            (([1.0], [1.0]), {"n_boot": 0}),
            (([1.0], [1.0]), {"confidence": 1.0}),
            (([1.0], [1.0]), {"confidence": 0.0}),
        ],
    )
    def test_invalid_input(
        self, args: tuple[list[float], list[float]], kwargs: dict[str, float]
    ) -> None:
        with pytest.raises(ValueError, match=r"same length|no paired|n_boot|confidence"):
            paired_bootstrap(*args, **kwargs)  # type: ignore[arg-type]


class TestMcNemar:
    def test_hand_computed_p_value(self) -> None:
        # 0 base-only, 10 candidate-only: p = 2 * 1 / 2**10
        r = mcnemar_exact([0.0] * 10, [1.0] * 10)
        assert (r.base_only, r.candidate_only) == (0, 10)
        assert r.p_value == pytest.approx(2 / 1024)

    def test_balanced_discordance_is_not_significant(self) -> None:
        r = mcnemar_exact([1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0])
        assert r.p_value == 1.0

    def test_no_discordant_pairs(self) -> None:
        r = mcnemar_exact([1.0, 0.0], [1.0, 0.0])
        assert (r.base_only, r.candidate_only, r.p_value) == (0, 0, 1.0)

    def test_p_value_capped_at_one(self) -> None:
        assert mcnemar_exact([1.0, 0.0], [0.0, 1.0]).p_value == 1.0

    def test_large_counts_do_not_overflow(self) -> None:
        r = mcnemar_exact([1.0] * 3000 + [0.0] * 2000, [0.0] * 3000 + [1.0] * 2000)
        assert 0.0 <= r.p_value <= 1.0

    def test_rejects_non_binary(self) -> None:
        with pytest.raises(ValueError, match="0/1"):
            mcnemar_exact([0.5], [1.0])

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            mcnemar_exact([1.0], [1.0, 0.0])

    def test_is_binary(self) -> None:
        assert is_binary([0.0, 1.0, 1.0])
        assert not is_binary([0.0, 0.5])
        assert not is_binary([])


class TestSliceValue:
    def test_lang(self) -> None:
        assert slice_value(example(lang="yo"), "lang") == "yo"
        assert slice_value(example(), "lang") == "(none)"

    @pytest.mark.parametrize(
        ("words", "bucket"),
        [(1, "short"), (5, "short"), (6, "medium"), (15, "medium"), (16, "long")],
    )
    def test_length_buckets(self, words: int, bucket: str) -> None:
        ref = " ".join(["w"] * words)
        assert slice_value(example(reference=ref), "length").startswith(bucket)

    def test_has_number(self) -> None:
        assert slice_value(example(reference="cost 50 naira"), "has_number") == "yes"
        assert slice_value(example(reference="no digits"), "has_number") == "no"

    def test_meta_field_and_missing(self) -> None:
        assert slice_value(example(domain="agri"), "domain") == "agri"
        assert slice_value(example(), "domain") == "(missing)"

    def test_non_string_meta_is_stringified(self) -> None:
        assert slice_value(example(year=2026), "year") == "2026"

    def test_group_counts(self) -> None:
        exs = [example(i, domain="a") for i in range(3)] + [example(9, domain="b")]
        assert dict(group_counts(exs, "domain")) == {"a": 3, "b": 1}


class TestVerdict:
    @pytest.mark.parametrize(
        ("low", "high", "higher", "expected"),
        [
            (0.1, 0.5, True, "improved"),
            (-0.5, -0.1, True, "regressed"),
            (-0.1, 0.1, True, "no clear change"),
            (0.1, 0.5, False, "regressed"),
            (-0.5, -0.1, False, "improved"),
            (0.0, 0.5, True, "no clear change"),
        ],
    )
    def test_direction(self, low: float, high: float, higher: bool, expected: str) -> None:
        assert verdict(low, high, higher_is_better=higher) == expected


class TestAnalyseSlices:
    def test_small_slice_is_insufficient_never_a_claim(self) -> None:
        exs = [example(i, g="tiny") for i in range(MIN_SLICE_N - 1)]
        (result,) = analyse_slices("g", exs, [0.0] * len(exs), [1.0] * len(exs))
        assert result.status == "insufficient data"
        assert (result.delta, result.low, result.high) == (None, None, None)
        assert result.n == MIN_SLICE_N - 1
        assert result.candidate_mean == 1.0  # still visible, just not claimed

    def test_threshold_is_inclusive(self) -> None:
        exs = [example(i, g="x") for i in range(MIN_SLICE_N)]
        (result,) = analyse_slices("g", exs, [0.0] * MIN_SLICE_N, [1.0] * MIN_SLICE_N)
        assert result.status == "improved"

    def test_improved_and_regressed_slices_ordered_worst_first(self) -> None:
        exs = [example(i, g="up") for i in range(40)] + [
            example(100 + i, g="down") for i in range(40)
        ]
        base = [0.0] * 40 + [1.0] * 40
        cand = [1.0] * 40 + [0.0] * 40
        results = analyse_slices("g", exs, base, cand)
        assert [r.value for r in results] == ["down", "up"]
        assert [r.status for r in results] == ["regressed", "improved"]

    def test_lower_is_better_flips_direction_and_order(self) -> None:
        exs = [example(i, g="a") for i in range(40)] + [example(100 + i, g="b") for i in range(40)]
        base = [0.5] * 80
        cand = [0.1] * 40 + [0.9] * 40  # a: error fell (good), b: error rose (bad)
        results = analyse_slices("g", exs, base, cand, higher_is_better=False)
        assert [r.value for r in results] == ["b", "a"]
        assert [r.status for r in results] == ["regressed", "improved"]

    def test_insufficient_slices_sort_last(self) -> None:
        exs = [example(i, g="big") for i in range(40)] + [
            example(100 + i, g="small") for i in range(3)
        ]
        results = analyse_slices("g", exs, [0.0] * 43, [1.0] * 43)
        assert [r.value for r in results] == ["big", "small"]

    def test_custom_min_n(self) -> None:
        exs = [example(i, g="x") for i in range(5)]
        (result,) = analyse_slices("g", exs, [0.0] * 5, [1.0] * 5, min_n=5)
        assert result.status == "improved"
