"""errors, types, security helpers."""

import pytest

from atlasforge.errors import (
    AtlasForgeError,
    BackendError,
    BackendHTTPError,
    BackendTimeout,
    ConfigError,
    DatasetError,
)
from atlasforge.security import mask_secret
from atlasforge.types import LANGS, GenParams, parse_lang


class TestErrors:
    def test_format_with_hint(self) -> None:
        err = AtlasForgeError("boom", hint="try again")
        assert err.format() == "boom\n  -> try again"
        assert str(err) == "boom"

    def test_format_without_hint(self) -> None:
        assert AtlasForgeError("boom").format() == "boom"

    def test_dataset_error_line_number(self) -> None:
        err = DatasetError("missing 'reference'", line=7)
        assert str(err) == "line 7: missing 'reference'"
        assert err.line == 7

    def test_http_error_fields(self) -> None:
        err = BackendHTTPError("bad", status_code=429, request_id="r1", body_excerpt="slow down")
        assert (err.status_code, err.request_id, err.body_excerpt) == (429, "r1", "slow down")
        assert isinstance(err, BackendError)

    def test_timeout_is_not_builtin(self) -> None:
        assert not issubclass(BackendTimeout, TimeoutError)
        assert issubclass(BackendTimeout, BackendError)


class TestLang:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("ha", "ha"), (" YO ", "yo"), ("Igbo", "ig"), ("english", "en"), ("HAUSA", "ha")],
    )
    def test_parse(self, raw: str, expected: str) -> None:
        assert parse_lang(raw) == expected

    def test_parse_rejects_unknown_with_hint(self) -> None:
        with pytest.raises(ConfigError) as info:
            parse_lang("fr")
        assert info.value.hint is not None
        assert "ha" in info.value.hint

    def test_langs_constant(self) -> None:
        assert LANGS == ("ha", "yo", "ig", "en")


class TestGenParams:
    def test_defaults_match_model_card(self) -> None:
        p = GenParams()
        assert (p.temperature, p.repetition_penalty, p.max_new_tokens) == (0.1, 1.12, 1000)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature": -0.1},
            {"repetition_penalty": 0},
            {"max_new_tokens": 0},
            {"top_p": 0},
            {"top_p": 1.5},
        ],
    )
    def test_validation(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ConfigError):
            GenParams(**kwargs)  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            GenParams().temperature = 1.0  # type: ignore[misc]


class TestMaskSecret:
    def test_hf_style(self) -> None:
        assert mask_secret("hf_abcdefghijkl") == "hf_****ijkl"

    def test_short_reveals_nothing(self) -> None:
        assert mask_secret("abc") == "****"
        assert mask_secret("") == "****"

    def test_never_reveals_middle(self) -> None:
        secret = "hf_SECRETMIDDLEPARTxyz1"
        assert "SECRETMIDDLE" not in mask_secret(secret)

    def test_no_prefix(self) -> None:
        assert mask_secret("abcdefghijklmnop") == "****mnop"
