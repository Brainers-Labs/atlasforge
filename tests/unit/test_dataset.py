import hashlib
import json
from pathlib import Path

import pytest

from atlasforge.errors import ConfigError, DatasetError
from atlasforge.eval.dataset import load_dataset


def write(tmp_path: Path, *lines: object, name: str = "d.jsonl") -> Path:
    path = tmp_path / name
    body = "\n".join(x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in lines)
    path.write_text(body + "\n", encoding="utf-8")
    return path


class TestGeneration:
    def test_loads_input_shorthand(self, tmp_path: Path) -> None:
        path = write(tmp_path, {"id": "a", "input": "Bawo ni?", "reference": "Fine", "lang": "yo"})
        ds = load_dataset(path, "generation")
        (ex,) = ds.examples
        assert ex.id == "a"
        assert ex.messages == ({"role": "user", "content": "Bawo ni?"},)
        assert ex.lang == "yo"
        assert ex.reference == "Fine"

    def test_messages_form(self, tmp_path: Path) -> None:
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
        ds = load_dataset(write(tmp_path, {"messages": msgs}), "generation")
        assert [m["role"] for m in ds.examples[0].messages] == ["system", "user"]

    def test_reference_optional_for_generation(self, tmp_path: Path) -> None:
        ds = load_dataset(write(tmp_path, {"input": "hi"}), "generation")
        assert ds.examples[0].reference is None

    def test_unicode_survives(self, tmp_path: Path) -> None:
        text = "Ṣé o wà dáadáa?"
        ds = load_dataset(write(tmp_path, {"input": text}), "generation")
        assert ds.examples[0].messages[0]["content"] == text

    def test_meta_kept(self, tmp_path: Path) -> None:
        ds = load_dataset(write(tmp_path, {"input": "x", "meta": {"domain": "agri"}}), "generation")
        assert ds.examples[0].meta == {"domain": "agri"}

    def test_blank_lines_and_bom_ok(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_bytes(b'\xef\xbb\xbf{"input": "a"}\n\n{"input": "b"}\n')
        assert len(load_dataset(path, "generation")) == 2

    def test_crlf_ok(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_bytes(b'{"input": "a"}\r\n{"input": "b"}\r\n')
        assert len(load_dataset(path, "generation")) == 2

    def test_unicode_line_separator_inside_string_is_not_a_line_break(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_bytes('{"input": "a b"}\n'.encode())
        ds = load_dataset(path, "generation")
        assert ds.examples[0].messages[0]["content"] == "a b"


class TestIds:
    def test_auto_id_is_stable_and_content_based(self, tmp_path: Path) -> None:
        a = load_dataset(write(tmp_path, {"input": "x", "reference": "y"}), "generation")
        b = load_dataset(
            write(tmp_path, {"input": "x", "reference": "y"}, name="e.jsonl"), "generation"
        )
        assert a.examples[0].id == b.examples[0].id
        assert len(a.examples[0].id) == 12

    def test_integer_id_becomes_string(self, tmp_path: Path) -> None:
        assert (
            load_dataset(write(tmp_path, {"id": 7, "input": "x"}), "generation").examples[0].id
            == "7"
        )

    def test_duplicate_ids_rejected_with_first_line(self, tmp_path: Path) -> None:
        path = write(tmp_path, {"id": "a", "input": "x"}, {"id": "a", "input": "y"})
        with pytest.raises(DatasetError, match=r"line 2.*first seen on line 1"):
            load_dataset(path, "generation")

    def test_identical_content_without_ids_is_duplicate(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="duplicate id"):
            load_dataset(write(tmp_path, {"input": "x"}, {"input": "x"}), "generation")

    @pytest.mark.parametrize("bad", [True, "", [], 1.5])
    def test_bad_id_types(self, tmp_path: Path, bad: object) -> None:
        with pytest.raises(DatasetError, match="'id'"):
            load_dataset(write(tmp_path, {"id": bad, "input": "x"}), "generation")


class TestValidationErrors:
    def test_invalid_json_reports_line(self, tmp_path: Path) -> None:
        path = write(tmp_path, {"input": "ok"}, "{not json")
        with pytest.raises(DatasetError, match=r"line 2: invalid JSON"):
            load_dataset(path, "generation")

    def test_non_object_line(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="JSON object"):
            load_dataset(write(tmp_path, "[1, 2]"), "generation")

    def test_typo_key_rejected_with_hint(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="refrence") as info:
            load_dataset(write(tmp_path, {"input": "x", "refrence": "y"}), "generation")
        assert info.value.hint is not None
        assert "meta" in info.value.hint

    def test_input_and_messages_together(self, tmp_path: Path) -> None:
        row = {"input": "x", "messages": [{"role": "user", "content": "y"}]}
        with pytest.raises(DatasetError, match="exactly one"):
            load_dataset(write(tmp_path, row), "generation")

    def test_neither_input_nor_messages(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="exactly one"):
            load_dataset(write(tmp_path, {"reference": "y"}), "generation")

    @pytest.mark.parametrize(
        "row",
        [
            {"input": ""},
            {"input": 5},
            {"messages": []},
            {"messages": "hi"},
            {"messages": [{"role": "bot", "content": "x"}]},
            {"messages": [{"role": "user"}]},
        ],
    )
    def test_bad_inputs(self, tmp_path: Path, row: dict[str, object]) -> None:
        with pytest.raises(DatasetError):
            load_dataset(write(tmp_path, row), "generation")

    def test_bad_lang_gets_line_and_hint(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match=r"line 1") as info:
            load_dataset(write(tmp_path, {"input": "x", "lang": "fr"}), "generation")
        assert info.value.hint is not None

    def test_non_string_lang(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="'lang'"):
            load_dataset(write(tmp_path, {"input": "x", "lang": 3}), "generation")

    def test_bad_meta(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="'meta'"):
            load_dataset(write(tmp_path, {"input": "x", "meta": [1]}), "generation")

    def test_bad_reference(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="'reference'"):
            load_dataset(write(tmp_path, {"input": "x", "reference": 3}), "generation")

    def test_empty_file(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="no examples"):
            load_dataset(write(tmp_path, ""), "generation")

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="cannot read"):
            load_dataset(tmp_path / "nope.jsonl", "generation")

    def test_not_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_bytes(b'{"input": "\xff"}\n')
        with pytest.raises(DatasetError, match="UTF-8"):
            load_dataset(path, "generation")

    def test_unknown_task(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            load_dataset(write(tmp_path, {"input": "x"}), "nope")  # type: ignore[arg-type]


class TestClassification:
    def test_reference_required(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="'reference' is required"):
            load_dataset(write(tmp_path, {"input": "x"}), "classification")

    def test_ok(self, tmp_path: Path) -> None:
        ds = load_dataset(write(tmp_path, {"input": "x", "reference": "pos"}), "classification")
        assert ds.examples[0].reference == "pos"


class TestAsr:
    def test_relative_audio_resolved_against_dataset_dir(self, tmp_path: Path) -> None:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        ds = load_dataset(write(tmp_path, {"audio": "a.wav", "reference": "ina kwana"}), "asr")
        assert ds.examples[0].audio == tmp_path / "a.wav"

    def test_missing_audio_file(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="audio file not found"):
            load_dataset(write(tmp_path, {"audio": "nope.wav", "reference": "x"}), "asr")

    def test_requires_audio_and_reference(self, tmp_path: Path) -> None:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        with pytest.raises(DatasetError, match="requires a string 'audio'"):
            load_dataset(write(tmp_path, {"reference": "x"}), "asr")
        with pytest.raises(DatasetError, match="'reference' is required"):
            load_dataset(write(tmp_path, {"audio": "a.wav"}), "asr")

    def test_input_not_allowed(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="takes 'audio'"):
            load_dataset(write(tmp_path, {"audio": "a.wav", "input": "x", "reference": "x"}), "asr")

    def test_audio_not_allowed_for_text_tasks(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="only valid for task 'asr'"):
            load_dataset(write(tmp_path, {"input": "x", "audio": "a.wav"}), "generation")


class TestFingerprint:
    def test_sha256_matches_file_bytes(self, tmp_path: Path) -> None:
        path = write(tmp_path, {"input": "x"})
        assert (
            load_dataset(path, "generation").sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        )

    def test_changes_when_file_changes(self, tmp_path: Path) -> None:
        a = load_dataset(write(tmp_path, {"input": "x"}), "generation").sha256
        b = load_dataset(write(tmp_path, {"input": "y"}), "generation").sha256
        assert a != b
