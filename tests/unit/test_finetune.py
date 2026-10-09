"""Fine-tuning: config, data preparation, and training wiring (stand-in torch/trl/peft)."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from atlasforge import __version__
from atlasforge.cli import app
from atlasforge.errors import BackendError, ConfigError, ResourceError
from atlasforge.finetune import QLoRAConfig, describe, load_config, plan_run, prepare_data, train
from atlasforge.finetune.data import check_truncation, count_truncated, render_texts
from atlasforge.finetune.qlora import RUN_FILE

runner = CliRunner()


def write_jsonl(path: Path, n: int = 30, *, prefix: str = "q", reference: bool = True) -> Path:
    rows = []
    for i in range(n):
        row: dict[str, Any] = {"id": f"{prefix}{i}", "input": f"{prefix} question {i}"}
        if reference:
            row["reference"] = f"{prefix} answer {i}"
        rows.append(row)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def make_config(tmp_path: Path, **overrides: Any) -> QLoRAConfig:
    values: dict[str, Any] = {"output_dir": tmp_path / "adapter", **overrides}
    if "train_file" not in values:
        values["train_file"] = write_jsonl(tmp_path / "train.jsonl")
    return QLoRAConfig(**values)


class TestConfigValidation:
    def test_defaults_are_valid_and_conventional(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        assert (cfg.lora_r, cfg.lora_alpha, cfg.quantize) == (16, 32, "4bit")
        assert cfg.effective_batch_size == 16
        assert "q_proj" in cfg.target_modules

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"lora_r": 0}, "lora_r"),
            ({"lora_alpha": 0}, "lora_alpha"),
            ({"lora_dropout": 1.0}, "lora_dropout"),
            ({"lora_dropout": -0.1}, "lora_dropout"),
            ({"target_modules": ()}, "target_modules"),
            ({"learning_rate": 0}, "learning_rate"),
            ({"num_epochs": 0}, "num_epochs"),
            ({"batch_size": 0}, "batch_size"),
            ({"grad_accum": 0}, "grad_accum"),
            ({"max_seq_len": 32}, "max_seq_len"),
            ({"max_seq_len": 9000}, "max_seq_len"),
            ({"warmup_ratio": 1.0}, "warmup_ratio"),
            ({"logging_steps": 0}, "logging_steps"),
            ({"quantize": "8bit"}, "quantize"),
            ({"min_examples": 0}, "min_examples"),
        ],
    )
    def test_each_invalid_value_is_named(
        self, tmp_path: Path, override: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ConfigError, match=message):
            make_config(tmp_path, **override)

    def test_all_problems_reported_together(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError) as info:
            make_config(tmp_path, lora_r=0, learning_rate=0)
        assert "lora_r" in str(info.value)
        assert "learning_rate" in str(info.value)

    def test_hyperparameters_summary_for_the_model_card(self, tmp_path: Path) -> None:
        hyper = make_config(tmp_path, lora_r=8).hyperparameters()
        assert hyper["lora_r"] == 8
        assert hyper["effective_batch_size"] == 16
        assert hyper["quantize"] == "4bit"
        assert isinstance(hyper["target_modules"], str)


class TestLoadConfig:
    def test_json_with_relative_paths_resolved_against_the_config_dir(self, tmp_path: Path) -> None:
        write_jsonl(tmp_path / "train.jsonl")
        cfg_file = tmp_path / "run.json"
        cfg_file.write_text(
            json.dumps({"train_file": "train.jsonl", "output_dir": "out", "lora_r": 8})
        )
        cfg = load_config(cfg_file)
        assert cfg.train_file == tmp_path / "train.jsonl"
        assert cfg.output_dir == tmp_path / "out"
        assert cfg.lora_r == 8
        assert cfg.eval_file is None

    def test_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "run.yaml"
        cfg_file.write_text(
            "train_file: train.jsonl\noutput_dir: out\neval_file: test.jsonl\n"
            "target_modules: [q_proj, v_proj]\nquantize: none\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.eval_file == tmp_path / "test.jsonl"
        assert cfg.target_modules == ("q_proj", "v_proj")
        assert cfg.quantize == "none"

    def test_absolute_paths_are_kept(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "run.json"
        absolute = tmp_path / "elsewhere" / "t.jsonl"
        cfg_file.write_text(json.dumps({"train_file": str(absolute), "output_dir": "out"}))
        assert load_config(cfg_file).train_file == absolute

    def test_a_typo_is_rejected_not_ignored(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "run.json"
        cfg_file.write_text(json.dumps({"train_file": "a", "output_dir": "b", "learing_rate": 1}))
        with pytest.raises(ConfigError, match="learing_rate") as info:
            load_config(cfg_file)
        assert "learning_rate" in (info.value.hint or "")

    def test_required_keys(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "run.json"
        cfg_file.write_text(json.dumps({"train_file": "a"}))
        with pytest.raises(ConfigError, match="output_dir"):
            load_config(cfg_file)

    @pytest.mark.parametrize("bad", ['["a"]', '"text"', "42"])
    def test_must_be_a_mapping(self, tmp_path: Path, bad: str) -> None:
        cfg_file = tmp_path / "run.json"
        cfg_file.write_text(bad)
        with pytest.raises(ConfigError, match="mapping"):
            load_config(cfg_file)

    def test_target_modules_type_checked(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "run.json"
        cfg_file.write_text(
            json.dumps({"train_file": "a", "output_dir": "b", "target_modules": "q_proj"})
        )
        with pytest.raises(ConfigError, match="list of strings"):
            load_config(cfg_file)

    def test_wrong_value_type_is_a_config_error(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "run.json"
        cfg_file.write_text(json.dumps({"train_file": "a", "output_dir": "b", "lora_r": "sixteen"}))
        with pytest.raises((ConfigError, TypeError)):
            load_config(cfg_file)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "run.toml"
        cfg_file.write_text("x = 1")
        with pytest.raises(ConfigError, match="Unsupported") as info:
            load_config(cfg_file)
        assert ".json" in (info.value.hint or "")

    def test_unreadable_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="Cannot read"):
            load_config(tmp_path / "missing.json")

    def test_invalid_json_and_yaml(self, tmp_path: Path) -> None:
        bad_json, bad_yaml = tmp_path / "a.json", tmp_path / "a.yaml"
        bad_json.write_text("{nope")
        bad_yaml.write_text("key: [unclosed")
        with pytest.raises(ConfigError, match="not valid JSON"):
            load_config(bad_json)
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(bad_yaml)

    def test_yaml_without_pyyaml_gives_a_hint_and_json_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "yaml", None)
        yaml_file = tmp_path / "run.yaml"
        yaml_file.write_text("train_file: a\noutput_dir: b\n")
        with pytest.raises(ConfigError, match="PyYAML") as info:
            load_config(yaml_file)
        assert "finetune" in (info.value.hint or "")
        json_file = tmp_path / "run.json"
        json_file.write_text(json.dumps({"train_file": "a", "output_dir": "b"}))
        assert load_config(json_file).output_dir == tmp_path / "b"


class TestPrepareData:
    def test_builds_conversations_ending_with_the_reference(self, tmp_path: Path) -> None:
        data = prepare_data(make_config(tmp_path))
        assert len(data) == 30
        conversation = data.conversations[0]
        assert [m["role"] for m in conversation] == ["user", "assistant"]
        assert conversation[-1]["content"] == "q answer 0"
        assert len(data.sha256) == 64

    def test_messages_form_is_preserved(self, tmp_path: Path) -> None:
        rows = [
            {
                "id": str(i),
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": f"u{i}"},
                ],
                "reference": f"a{i}",
            }
            for i in range(25)
        ]
        path = tmp_path / "t.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        data = prepare_data(make_config(tmp_path, train_file=path))
        assert [m["role"] for m in data.conversations[0]] == ["system", "user", "assistant"]

    def test_validation_errors_stop_everything_with_a_pointer_to_the_validator(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"input": "x", "reference": "y"}\n{broken\n' * 15, encoding="utf-8")
        with pytest.raises(ConfigError, match="error") as info:
            prepare_data(make_config(tmp_path, train_file=bad))
        assert "dataset validate" in (info.value.hint or "")

    def test_leakage_against_the_held_out_file_is_fatal(self, tmp_path: Path) -> None:
        train = write_jsonl(tmp_path / "train.jsonl", 30, prefix="a")
        test = tmp_path / "test.jsonl"
        test.write_text(
            json.dumps({"id": "t", "input": "a question 3", "reference": "x"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="also appear") as info:
            prepare_data(make_config(tmp_path, train_file=train, eval_file=test))
        assert "--against" in (info.value.hint or "")

    def test_disjoint_held_out_file_is_fine(self, tmp_path: Path) -> None:
        test = write_jsonl(tmp_path / "test.jsonl", 5, prefix="heldout")
        assert len(prepare_data(make_config(tmp_path, eval_file=test))) == 30

    def test_every_example_needs_an_answer(self, tmp_path: Path) -> None:
        no_refs = write_jsonl(tmp_path / "noref.jsonl", 30, reference=False)
        with pytest.raises(ConfigError, match="no 'reference'"):
            prepare_data(make_config(tmp_path, train_file=no_refs))

    def test_too_few_examples_refused_with_the_reason(self, tmp_path: Path) -> None:
        small = write_jsonl(tmp_path / "small.jsonl", 5)
        with pytest.raises(ConfigError, match="at least 20") as info:
            prepare_data(make_config(tmp_path, train_file=small))
        assert "repeat" in (info.value.hint or "")

    def test_the_minimum_can_be_lowered_deliberately(self, tmp_path: Path) -> None:
        small = write_jsonl(tmp_path / "small.jsonl", 5)
        assert len(prepare_data(make_config(tmp_path, train_file=small, min_examples=3))) == 5


class TestPlan:
    def test_step_arithmetic(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path, num_epochs=3)  # 30 examples, effective batch 16
        plan = plan_run(cfg, prepare_data(cfg))
        assert plan.steps_per_epoch == 2  # ceil(30 / 16)
        assert plan.total_steps == 6
        assert plan.n_examples == 30
        assert plan.median_words > 0

    def test_describe_states_when_leakage_was_not_checked(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path)
        text = describe(cfg, prepare_data(cfg))
        assert "leakage was NOT checked" in text
        assert "QLoRA (4-bit)" in text
        assert "30 examples" in text
        assert "6 total" not in text

    def test_describe_with_held_out_file_and_plain_lora(self, tmp_path: Path) -> None:
        test = write_jsonl(tmp_path / "test.jsonl", 3, prefix="held")
        cfg = make_config(tmp_path, eval_file=test, quantize="none", revision="abc")
        text = describe(cfg, prepare_data(cfg))
        assert "NOT checked" not in text
        assert "LoRA (no quantisation)" in text
        assert "@ abc" in text


class FakeTokenizer:
    def __init__(
        self, chat_template: str | None = "tpl", lengths: dict[str, int] | None = None
    ) -> None:
        self.chat_template = chat_template
        self.pad_token: str | None = None
        self.eos_token = "</s>"
        self.lengths = lengths or {}
        self.saved_to: str | None = None

    def apply_chat_template(self, messages: list[dict[str, str]], *, tokenize: bool) -> str:
        assert tokenize is False
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, texts: list[str], *, add_special_tokens: bool) -> dict[str, list[list[int]]]:
        assert add_special_tokens is False
        return {"input_ids": [[0] * self.lengths.get(t, 10) for t in texts]}

    def save_pretrained(self, path: str) -> None:
        self.saved_to = path


class TestTextRendering:
    def test_uses_the_chat_template_for_every_conversation(self, tmp_path: Path) -> None:
        data = prepare_data(make_config(tmp_path))
        texts = render_texts(data, FakeTokenizer())
        assert len(texts) == 30
        assert texts[0] == "user:q question 0|assistant:q answer 0"

    def test_a_missing_chat_template_is_refused_not_guessed(self, tmp_path: Path) -> None:
        data = prepare_data(make_config(tmp_path))
        with pytest.raises(ConfigError, match="chat template") as info:
            render_texts(data, FakeTokenizer(chat_template=None))
        assert "planning/21" in (info.value.hint or "")

    def test_counts_examples_over_the_limit(self) -> None:
        tok = FakeTokenizer(lengths={"long": 500})
        assert count_truncated(["short", "long", "long"], tok, 100) == 2

    @pytest.mark.parametrize(
        ("truncated", "ok"), [(0, True), (10, True), (11, False), (100, False)]
    )
    def test_truncation_guard_threshold_is_ten_percent(self, truncated: int, ok: bool) -> None:
        if ok:
            check_truncation(truncated, 100, 2048)
        else:
            with pytest.raises(ConfigError, match="cut off") as info:
                check_truncation(truncated, 100, 2048)
            assert "max_seq_len" in (info.value.hint or "")

    def test_no_examples_never_divides_by_zero(self) -> None:
        check_truncation(0, 0, 2048)


# ---- training wiring, with stand-in torch / transformers / peft / trl / datasets -----------


class SFTConfigNew:
    def __init__(self, output_dir: str, *, max_length: int | None = None, dataset_text_field: str | None = None,
                 per_device_train_batch_size: int | None = None, gradient_accumulation_steps: int | None = None,
                 learning_rate: float | None = None, num_train_epochs: int | None = None, warmup_ratio: float | None = None,
                 logging_steps: int | None = None, seed: int | None = None, bf16: bool | None = None,
                 fp16: bool | None = None, report_to: str | None = None, save_strategy: str | None = None) -> None:  # fmt: skip
        self.kwargs = {k: v for k, v in locals().items() if k != "self"}


class SFTConfigOld(SFTConfigNew):
    def __init__(self, output_dir: str, *, max_seq_length: int | None = None, dataset_text_field: str | None = None,
                 per_device_train_batch_size: int | None = None, gradient_accumulation_steps: int | None = None,
                 learning_rate: float | None = None, num_train_epochs: int | None = None, warmup_ratio: float | None = None,
                 logging_steps: int | None = None, seed: int | None = None, bf16: bool | None = None,
                 fp16: bool | None = None, report_to: str | None = None, save_strategy: str | None = None) -> None:  # fmt: skip
        self.kwargs = {k: v for k, v in locals().items() if k != "self"}


class Harness:
    def __init__(self, *, cuda: bool = True, bf16: bool = True, new_trl: bool = True) -> None:
        self.events: list[str] = []
        self.tokenizer = FakeTokenizer()
        self.model_kwargs: dict[str, Any] = {}
        self.lora: dict[str, Any] = {}
        self.trainer_kwargs: dict[str, Any] = {}
        self.train_error: Exception | None = None
        self.sft_config: Any = None
        outer = self

        class Model:
            def save_pretrained(self, path: str) -> None:
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "adapter_model.safetensors").write_bytes(b"fake")
                outer.events.append("saved-adapter")

        self.model = Model()

        torch = ModuleType("torch")
        torch.float16 = "fp16"  # type: ignore[attr-defined]
        torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
        torch.cuda = SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=lambda: bf16)  # type: ignore[attr-defined]

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(name: str, **kw: Any) -> FakeTokenizer:
                outer.events.append(f"tokenizer:{name}")
                return outer.tokenizer

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(name: str, **kw: Any) -> Any:
                outer.events.append("model-loaded")
                outer.model_kwargs = {"name": name, **kw}
                return outer.model

        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
        transformers.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore[attr-defined]
        transformers.BitsAndBytesConfig = SimpleNamespace  # type: ignore[attr-defined]

        peft = ModuleType("peft")

        def prepare(model: Any) -> Any:
            outer.events.append("prepared-kbit")
            return model

        def lora_config(**kw: Any) -> Any:
            outer.lora = kw
            return SimpleNamespace(**kw)

        peft.prepare_model_for_kbit_training = prepare  # type: ignore[attr-defined]
        peft.LoraConfig = lora_config  # type: ignore[attr-defined]

        sft_config_cls = SFTConfigNew if new_trl else SFTConfigOld

        def make_trainer_class() -> Any:
            if new_trl:

                class Trainer:
                    def __init__(
                        self,
                        model: Any,
                        args: Any,
                        train_dataset: Any,
                        peft_config: Any,
                        processing_class: Any,
                    ) -> None:
                        outer.trainer_kwargs = {
                            "model": model,
                            "args": args,
                            "train_dataset": train_dataset,
                            "peft_config": peft_config,
                            "processing_class": processing_class,
                        }
                        self.model = model

                    def train(self) -> None:
                        outer.events.append("trained")
                        if outer.train_error:
                            raise outer.train_error

                return Trainer

            class OldTrainer:
                def __init__(
                    self,
                    model: Any,
                    args: Any,
                    train_dataset: Any,
                    peft_config: Any,
                    tokenizer: Any,
                ) -> None:
                    outer.trainer_kwargs = {
                        "model": model,
                        "args": args,
                        "train_dataset": train_dataset,
                        "peft_config": peft_config,
                        "tokenizer": tokenizer,
                    }
                    self.model = model

                def train(self) -> None:
                    outer.events.append("trained")

            return OldTrainer

        trl = ModuleType("trl")
        trl.SFTConfig = sft_config_cls  # type: ignore[attr-defined]
        trl.SFTTrainer = make_trainer_class()  # type: ignore[attr-defined]

        datasets = ModuleType("datasets")
        datasets.Dataset = SimpleNamespace(from_dict=lambda d: SimpleNamespace(columns=d))  # type: ignore[attr-defined]

        self.modules = {
            "torch": torch,
            "transformers": transformers,
            "peft": peft,
            "trl": trl,
            "datasets": datasets,
        }

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, module in self.modules.items():
            monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Harness:
    harness = Harness()
    harness.install(monkeypatch)
    return harness


class TestTrain:
    def test_writes_the_adapter_and_a_complete_run_record(
        self, fake: Harness, tmp_path: Path
    ) -> None:
        cfg = make_config(tmp_path, revision="rev1")
        run = train(cfg)
        out = tmp_path / "adapter"
        assert (out / "adapter_model.safetensors").is_file()
        assert fake.tokenizer.saved_to == str(out)
        record = json.loads((out / RUN_FILE).read_text(encoding="utf-8"))
        assert record["method"] == "qlora"
        assert record["base_model"] == "NCAIR1/N-ATLaS"
        assert record["base_revision"] == "rev1"
        assert record["n_train_examples"] == 30
        assert record["n_truncated"] == 0
        assert record["atlasforge_version"] == __version__
        assert record["hyperparameters"]["lora_r"] == 16
        assert len(record["dataset_sha256"]) == 64
        assert run.to_dict() == record

    def test_order_of_operations(self, fake: Harness, tmp_path: Path) -> None:
        train(make_config(tmp_path))
        assert fake.events == [
            "tokenizer:NCAIR1/N-ATLaS",
            "model-loaded",
            "prepared-kbit",
            "trained",
            "saved-adapter",
        ]

    def test_four_bit_loading_settings(self, fake: Harness, tmp_path: Path) -> None:
        train(make_config(tmp_path))
        quant = fake.model_kwargs["quantization_config"]
        assert (quant.load_in_4bit, quant.bnb_4bit_quant_type, quant.bnb_4bit_compute_dtype) == (
            True,
            "nf4",
            "bf16",
        )
        assert fake.model_kwargs["device_map"] == "auto"
        assert "torch_dtype" not in fake.model_kwargs

    def test_unquantised_lora_skips_quantisation_and_kbit_prep(
        self, fake: Harness, tmp_path: Path
    ) -> None:
        train(make_config(tmp_path, quantize="none"))
        assert "quantization_config" not in fake.model_kwargs
        assert fake.model_kwargs["torch_dtype"] == "bf16"
        assert "prepared-kbit" not in fake.events

    def test_lora_settings_are_passed_through(self, fake: Harness, tmp_path: Path) -> None:
        train(
            make_config(
                tmp_path,
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=("q_proj", "v_proj"),
            )
        )
        assert fake.lora == {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

    def test_training_text_is_the_chat_template_rendering(
        self, fake: Harness, tmp_path: Path
    ) -> None:
        train(make_config(tmp_path))
        column = fake.trainer_kwargs["train_dataset"].columns["text"]
        assert len(column) == 30
        assert column[0] == "user:q question 0|assistant:q answer 0"

    def test_new_trl_argument_names(self, fake: Harness, tmp_path: Path) -> None:
        train(make_config(tmp_path, max_seq_len=1024))
        kwargs = fake.trainer_kwargs["args"].kwargs
        assert kwargs["max_length"] == 1024
        assert "processing_class" in fake.trainer_kwargs
        assert kwargs["per_device_train_batch_size"] == 1
        assert kwargs["gradient_accumulation_steps"] == 16
        assert kwargs["dataset_text_field"] == "text"
        assert kwargs["report_to"] == "none"
        assert kwargs["output_dir"].endswith("checkpoints")

    def test_old_trl_argument_names(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        harness = Harness(new_trl=False)
        harness.install(monkeypatch)
        train(make_config(tmp_path, max_seq_len=1024))
        assert harness.trainer_kwargs["args"].kwargs["max_seq_length"] == 1024
        assert "tokenizer" in harness.trainer_kwargs
        assert "processing_class" not in harness.trainer_kwargs

    @pytest.mark.parametrize(("bf16", "expected"), [(True, (True, False)), (False, (False, True))])
    def test_precision_follows_hardware_support(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        bf16: bool,
        expected: tuple[bool, bool],
    ) -> None:
        harness = Harness(bf16=bf16)
        harness.install(monkeypatch)
        train(make_config(tmp_path))
        kwargs = harness.trainer_kwargs["args"].kwargs
        assert (kwargs["bf16"], kwargs["fp16"]) == expected

    def test_a_missing_pad_token_is_set_from_eos(self, fake: Harness, tmp_path: Path) -> None:
        train(make_config(tmp_path))
        assert fake.tokenizer.pad_token == "</s>"


class TestTrainGuards:
    def test_four_bit_without_a_gpu_is_refused_before_loading_anything(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        harness = Harness(cuda=False)
        harness.install(monkeypatch)
        with pytest.raises(ResourceError, match="NVIDIA") as info:
            train(make_config(tmp_path))
        assert "Mac" in (info.value.hint or "")
        assert harness.events == []

    def test_bad_data_fails_before_any_heavy_import(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)  # would raise the extras error if reached
        small = write_jsonl(tmp_path / "small.jsonl", 3)
        with pytest.raises(ConfigError, match="at least"):
            train(make_config(tmp_path, train_file=small))

    def test_missing_extras_give_the_install_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ConfigError) as info:
            train(make_config(tmp_path))
        assert info.value.hint == 'pip install "brainers-atlasforge[finetune]"'

    def test_heavily_truncated_data_is_refused_and_nothing_is_saved(
        self, fake: Harness, tmp_path: Path
    ) -> None:
        class EverythingIsLong(FakeTokenizer):
            def __call__(
                self, texts: list[str], *, add_special_tokens: bool
            ) -> dict[str, list[list[int]]]:
                return {"input_ids": [[0] * 99999 for _ in texts]}

        fake.tokenizer = EverythingIsLong()
        with pytest.raises(ConfigError, match="cut off"):
            train(make_config(tmp_path))
        assert not (tmp_path / "adapter" / RUN_FILE).exists()
        assert "model-loaded" not in fake.events

    def test_a_few_truncated_examples_are_recorded_not_fatal(
        self, fake: Harness, tmp_path: Path
    ) -> None:
        long_text = "user:q question 0|assistant:q answer 0"
        fake.tokenizer = FakeTokenizer(lengths={long_text: 5000})
        run = train(make_config(tmp_path, max_seq_len=2048))
        assert run.n_truncated == 1

    def test_no_chat_template_stops_training(self, fake: Harness, tmp_path: Path) -> None:
        fake.tokenizer = FakeTokenizer(chat_template=None)
        with pytest.raises(ConfigError, match="chat template"):
            train(make_config(tmp_path))
        assert "model-loaded" not in fake.events

    def test_out_of_memory_is_explained(self, fake: Harness, tmp_path: Path) -> None:
        class OutOfMemoryError(RuntimeError):
            pass

        fake.train_error = OutOfMemoryError("CUDA out of memory while training on private text")
        with pytest.raises(ResourceError, match="memory") as info:
            train(make_config(tmp_path))
        assert "private" not in info.value.format()
        assert "max_seq_len" in (info.value.hint or "")
        assert not (tmp_path / "adapter" / RUN_FILE).exists()

    def test_other_failures_never_echo_their_message(self, fake: Harness, tmp_path: Path) -> None:
        fake.train_error = ValueError("secret training row: q answer 7")
        with pytest.raises(BackendError) as info:
            train(make_config(tmp_path))
        assert "secret" not in info.value.format()
        assert "ValueError" in str(info.value)

    def test_gated_base_model_points_at_the_licence(self, fake: Harness, tmp_path: Path) -> None:
        fake.train_error = type("GatedRepoError", (Exception,), {})("x")
        with pytest.raises(BackendError) as info:
            train(make_config(tmp_path))
        assert "HF_TOKEN" in (info.value.hint or "")


class TestCli:
    def test_dry_run_needs_no_gpu_and_trains_nothing(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "run.json"
        write_jsonl(tmp_path / "train.jsonl")
        cfg_file.write_text(json.dumps({"train_file": "train.jsonl", "output_dir": "out"}))
        result = runner.invoke(app, ["finetune", str(cfg_file), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "30 examples" in result.output
        assert "Dry run: nothing was trained." in result.output
        assert not (tmp_path / "out").exists()

    def test_dry_run_surfaces_data_problems(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "run.json"
        write_jsonl(tmp_path / "train.jsonl", 3)
        cfg_file.write_text(json.dumps({"train_file": "train.jsonl", "output_dir": "out"}))
        result = runner.invoke(app, ["finetune", str(cfg_file), "--dry-run"])
        assert isinstance(result.exception, ConfigError)

    def test_finetune_trains_and_prints_next_steps(self, fake: Harness, tmp_path: Path) -> None:
        cfg_file = tmp_path / "run.json"
        write_jsonl(tmp_path / "train.jsonl")
        cfg_file.write_text(json.dumps({"train_file": "train.jsonl", "output_dir": "out"}))
        result = runner.invoke(app, ["finetune", str(cfg_file)])
        assert result.exit_code == 0, result.output
        assert "Adapter saved to" in result.output
        assert "--adapter" in result.output
        assert (tmp_path / "out" / RUN_FILE).is_file()

    def test_adapter_flag_is_rejected_for_server_backends(self) -> None:
        result = runner.invoke(
            app, ["run", "hi", "--base-url", "http://127.0.0.1:1/v1", "--adapter", "x"]
        )
        assert isinstance(result.exception, ConfigError)
        assert "local backend" in str(result.exception)
