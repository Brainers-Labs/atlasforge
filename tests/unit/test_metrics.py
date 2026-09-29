import pytest

from atlasforge.eval import metrics as m


class TestExactMatch:
    def test_equal(self) -> None:
        assert m.exact_match("a", "a") == 1.0

    def test_different(self) -> None:
        assert m.exact_match("a", "b") == 0.0


class TestChrf:
    def test_identical_is_100(self) -> None:
        assert m.chrf("ina kwana", "ina kwana") == pytest.approx(100.0)

    def test_disjoint_is_0(self) -> None:
        assert m.chrf("xyz", "ina kwana") == 0.0

    def test_partial_overlap_is_between(self) -> None:
        assert 0 < m.chrf("ina kwana lafiya", "ina kwana") < 100

    def test_both_empty_is_100_and_one_empty_is_0(self) -> None:
        assert m.chrf("", "") == 100.0
        assert m.chrf("", "abc") == 0.0
        assert m.chrf("abc", "") == 0.0

    def test_chrfpp_differs_from_chrf_on_word_order(self) -> None:
        a, b = "kwana ina lafiya", "ina kwana lafiya"
        assert m.chrf(a, b, word_order=2) != m.chrf(a, b)

    def test_corpus(self) -> None:
        assert m.corpus_chrf(["ina kwana"], ["ina kwana"]) == pytest.approx(100.0)

    def test_corpus_all_predictions_empty_scores_zero_without_crashing(self) -> None:
        assert m.corpus_chrf(["", ""], ["ina kwana", "sannu"]) == 0.0


class TestWerCer:
    def test_wer_one_of_three_words_wrong(self) -> None:
        assert m.wer("a b d", "a b c") == pytest.approx(1 / 3)

    def test_cer_one_of_three_chars_wrong(self) -> None:
        assert m.cer("abd", "abc") == pytest.approx(1 / 3)

    def test_perfect_is_zero(self) -> None:
        assert m.wer("a b c", "a b c") == 0.0

    def test_empty_prediction_is_all_deletions(self) -> None:
        assert m.wer("", "a b c") == 1.0

    def test_empty_reference_rule(self) -> None:
        assert m.wer("", "") == 0.0
        assert m.wer("x", "") == 1.0
        assert m.cer("", "") == 0.0
        assert m.cer("x", "") == 1.0

    def test_wer_can_exceed_one_with_insertions(self) -> None:
        assert m.wer("a b c d e", "a") > 1.0

    def test_corpus_wer_counts_errors_over_all_reference_words(self) -> None:
        # ref words: 2 + 2 = 4. Errors: 0 + (1 substitution + 1 deletion) = 2  ->  0.5
        assert m.corpus_wer(["a b", "c"], ["a b", "d e"]) == pytest.approx(0.5)

    def test_pooled_differs_from_mean_of_rates(self) -> None:
        preds, refs = ["x", "a b c d e f g h i j"], ["a", "a b c d e f g h i j"]
        pooled = m.corpus_wer(preds, refs)
        mean_of_rates = (m.wer(preds[0], refs[0]) + m.wer(preds[1], refs[1])) / 2
        assert pooled == pytest.approx(1 / 11)
        assert mean_of_rates == pytest.approx(0.5)

    def test_corpus_excludes_empty_references(self) -> None:
        assert m.corpus_wer(["a b", "zzz"], ["a b", ""]) == 0.0

    def test_corpus_all_empty_references_is_none(self) -> None:
        assert m.corpus_wer(["a"], [""]) is None
        assert m.corpus_cer(["a"], [""]) is None

    def test_corpus_cer(self) -> None:
        assert m.corpus_cer(["abd"], ["abc"]) == pytest.approx(1 / 3)


class TestExtractLabel:
    LABELS = frozenset({"positive", "negative", "very positive"})

    def test_finds_label(self) -> None:
        assert m.extract_label("the answer is positive today", self.LABELS) == "positive"

    def test_earliest_wins(self) -> None:
        assert m.extract_label("negative not positive", self.LABELS) == "negative"

    def test_longest_wins_at_same_position(self) -> None:
        assert m.extract_label("very positive indeed", self.LABELS) == "very positive"

    def test_whole_word_only(self) -> None:
        assert m.extract_label("positively", self.LABELS) is None

    def test_none_when_absent(self) -> None:
        assert m.extract_label("no idea", self.LABELS) is None

    def test_exact_answer(self) -> None:
        assert m.extract_label("positive", self.LABELS) == "positive"

    def test_ignores_empty_label(self) -> None:
        assert m.extract_label("anything", {""}) is None


class TestMacroF1:
    def test_hand_computed(self) -> None:
        # a: tp1 fp0 fn1 -> 2/3.  b: tp2 fp1 fn0 -> 4/5.  mean = (2/3 + 4/5) / 2
        got = m.macro_f1(["a", "b", "b", "b"], ["a", "a", "b", "b"])
        assert got == pytest.approx((2 / 3 + 4 / 5) / 2)

    def test_perfect(self) -> None:
        assert m.macro_f1(["a", "b"], ["a", "b"]) == 1.0

    def test_none_prediction_is_a_miss_without_false_positive(self) -> None:
        # a: tp0 fp0 fn1 -> 0.  b: tp1 fp0 fn0 -> 1.  mean 0.5
        assert m.macro_f1([None, "b"], ["a", "b"]) == pytest.approx(0.5)

    def test_all_wrong(self) -> None:
        assert m.macro_f1(["b", "a"], ["a", "b"]) == 0.0


class TestPercentile:
    def test_median_interpolates(self) -> None:
        assert m.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_p95(self) -> None:
        assert m.percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)

    def test_unsorted_input(self) -> None:
        assert m.percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)

    def test_single_value(self) -> None:
        assert m.percentile([7], 95) == 7

    def test_empty_is_none(self) -> None:
        assert m.percentile([], 50) is None

    def test_extremes(self) -> None:
        assert m.percentile([1, 2, 3], 0) == 1
        assert m.percentile([1, 2, 3], 100) == 3


class TestAlignmentChecks:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: m.corpus_chrf(["a"], []),
            lambda: m.corpus_wer(["a"], []),
            lambda: m.macro_f1(["a"], []),
            lambda: m.corpus_chrf([], []),
        ],
    )
    def test_misaligned_or_empty_raises(self, call: object) -> None:
        with pytest.raises(ValueError, match=r"differ in length|no examples"):
            call()  # type: ignore[operator]
