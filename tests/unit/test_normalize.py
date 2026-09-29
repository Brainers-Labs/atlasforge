"""Normalisation is what makes scores meaningful, so it is tested hardest.

Non-ASCII characters are written as escapes so the intent survives any editor:
  e-underdot U+1EB9  o-underdot U+1ECD  s-underdot U+1E63  n-dot-above U+1E45
  Hausa: b-hook U+0253  d-hook U+0257  k-hook U+0199  y-hook U+01B4
  tone marks: grave U+0300  acute U+0301  macron U+0304
"""

import unicodedata

import pytest

from atlasforge.errors import ConfigError
from atlasforge.eval.normalize import (
    NormalizeConfig,
    normalize,
    strip_tones,
    tone_aware,
    tone_insensitive,
)

E_DOT = "ẹ"
O_DOT = "ọ"
S_DOT = "ṣ"


class TestUnicodeForms:
    def test_precomposed_and_combining_are_equal(self) -> None:
        composed = "ẹ́"  # e-underdot + acute (precomposed base)
        combining = "ẹ́"  # e + dot below + acute
        assert normalize(composed) == normalize(combining)

    def test_output_is_nfc(self) -> None:
        out = normalize("ẹ́", tone_insensitive())
        assert out == unicodedata.normalize("NFC", out)

    def test_tone_aware_keeps_tone_marks(self) -> None:
        assert "́" in _nfd(normalize("wá", tone_aware()))


class TestToneStripping:
    def test_strips_tones_but_keeps_underdot(self) -> None:
        text = f"{O_DOT}̀r{O_DOT}̀"  # o-underdot + grave, r, o-underdot + grave
        assert normalize(text, tone_insensitive()) == f"{O_DOT}r{O_DOT}"

    def test_yoruba_sentence(self) -> None:
        text = "Ṣé o wà dáadáa?"
        assert normalize(text, tone_insensitive("yo")) == f"{S_DOT}e o wa daadaa"

    def test_keeps_igbo_n_dot_above(self) -> None:
        assert normalize("ṅu", tone_insensitive("ig")) == "ṅu"

    def test_strips_macron_and_grave_together(self) -> None:
        assert strip_tones("āè") == "ae"

    def test_syllabic_nasal_tone_removed(self) -> None:
        assert normalize("ń", tone_insensitive("yo")) == "n"  # n with acute

    def test_strip_is_noop_on_plain_ascii(self) -> None:
        assert strip_tones("hello world") == "hello world"


class TestHausa:
    def test_hooked_letters_preserved(self) -> None:
        text = "ɓarawo ƙasa ɗan ƴa"
        assert normalize(text, tone_insensitive("ha")) == text

    def test_uppercase_hooked_letters_lowercase_correctly(self) -> None:
        assert normalize("Ɗan Ɓara", tone_aware("ha")) == "ɗan ɓara"

    def test_leading_apostrophe_kept_for_hausa(self) -> None:
        assert normalize("’Yar", tone_aware("ha")) == "'yar"

    def test_leading_apostrophe_dropped_elsewhere(self) -> None:
        assert normalize("'Yar", tone_aware("en")) == "yar"

    def test_orphan_apostrophe_dropped_for_hausa(self) -> None:
        assert normalize("a ' b", tone_aware("ha")) == "a b"


class TestPunctuationAndSpace:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a,b", "a b"),
            ("well-known!", "well known"),
            ("  a \t b\n", "a b"),
            ("", ""),
            ("   ", ""),
            ("Hello, World.", "hello world"),
            ("50% off", "50 off"),
        ],
    )
    def test_cases(self, raw: str, expected: str) -> None:
        assert normalize(raw) == expected

    def test_internal_apostrophe_kept(self) -> None:
        assert normalize("Don’t stop") == "don't stop"

    def test_punctuation_keep_mode(self) -> None:
        cfg = NormalizeConfig(punctuation="keep")
        assert normalize("Hello, World!", cfg) == "hello, world!"

    def test_lowercase_off(self) -> None:
        cfg = NormalizeConfig(lowercase=False)
        assert normalize("Hello", cfg) == "Hello"

    def test_dotted_capital_i_stays_composed(self) -> None:
        out = normalize("İ", NormalizeConfig())
        assert out == unicodedata.normalize("NFC", out)


class TestIdempotence:
    SAMPLES = (
        "Ṣé o wà dáadáa?",
        "’Yar ɓarawo, ƙasa!",
        "a''b '' c",
        "Don’t -- stop...",
        "ẹ́ ẹ́",
        "",
    )

    @pytest.mark.parametrize("sample", SAMPLES)
    @pytest.mark.parametrize("tones", ["keep", "strip"])
    @pytest.mark.parametrize("lang", [None, "ha", "yo", "ig", "en"])
    def test_idempotent(self, sample: str, tones: str, lang: str | None) -> None:
        cfg = NormalizeConfig(lang=lang, tones=tones)  # type: ignore[arg-type]
        once = normalize(sample, cfg)
        assert normalize(once, cfg) == once


class TestConfigValidation:
    def test_bad_tones(self) -> None:
        with pytest.raises(ConfigError):
            NormalizeConfig(tones="nope")  # type: ignore[arg-type]

    def test_bad_punctuation(self) -> None:
        with pytest.raises(ConfigError):
            NormalizeConfig(punctuation="nope")  # type: ignore[arg-type]

    def test_bad_lang(self) -> None:
        with pytest.raises(ConfigError):
            NormalizeConfig(lang="fr")  # type: ignore[arg-type]

    def test_default_config_is_tone_aware(self) -> None:
        assert NormalizeConfig().tones == "keep"


def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)
