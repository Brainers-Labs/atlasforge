import io
import math
import shutil
import struct
import subprocess
import wave
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import pytest

from atlasforge.asr import (
    decode_audio,
    duration_s,
    merge_texts,
    pcm_to_float32,
    pcm_to_wav,
    plan_windows,
    transcribe_long,
)
from atlasforge.asr.audio import SAMPLE_RATE
from atlasforge.asr.chunking import LongAudioBackend
from atlasforge.errors import AudioError, ConfigError, ResourceError
from atlasforge.types import (
    AudioInput,
    BackendInfo,
    Generation,
    GenParams,
    Lang,
    Message,
    Transcript,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


def sine_wav(path: Path, seconds: float = 1.0, *, rate: int = 44_100, channels: int = 2) -> Path:
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate))) * channels
        for i in range(int(rate * seconds))
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return path


class TestPlanWindows:
    def test_short_audio_is_one_window(self) -> None:
        assert plan_windows(5 * SAMPLE_RATE) == [(0, 5 * SAMPLE_RATE)]

    def test_exactly_one_window_long(self) -> None:
        n = 28 * SAMPLE_RATE
        assert plan_windows(n) == [(0, n)]

    def test_sixty_seconds_hand_computed(self) -> None:
        s = SAMPLE_RATE
        # window 28 s, overlap 2 s -> step 26 s
        assert plan_windows(60 * s) == [(0, 28 * s), (26 * s, 54 * s), (52 * s, 60 * s)]

    @pytest.mark.parametrize("seconds", [28.1, 30, 55.5, 56, 57, 120, 3599])
    def test_windows_cover_everything_with_the_requested_overlap(self, seconds: float) -> None:
        n = int(seconds * SAMPLE_RATE)
        windows = plan_windows(n)
        assert windows[0][0] == 0
        assert windows[-1][1] == n
        for (_, prev_end), (start, _) in pairwise(windows):
            assert prev_end - start == 2 * SAMPLE_RATE  # exact overlap
        for start, end in windows:
            assert 0 < end - start <= 28 * SAMPLE_RATE
        assert windows[-1][1] - windows[-1][0] >= 2 * SAMPLE_RATE or len(windows) == 1

    def test_never_exceeds_the_model_limit(self) -> None:
        assert all(
            e - s <= 30 * SAMPLE_RATE
            for s, e in plan_windows(500 * SAMPLE_RATE, window_s=30, overlap_s=1)
        )

    def test_zero_overlap(self) -> None:
        s = SAMPLE_RATE
        assert plan_windows(60 * s, window_s=30, overlap_s=0) == [(0, 30 * s), (30 * s, 60 * s)]

    @pytest.mark.parametrize(
        ("window", "overlap"), [(31, 2), (0, 0), (-1, 0), (10, 10), (10, -1), (10, 11)]
    )
    def test_invalid_parameters(self, window: float, overlap: float) -> None:
        with pytest.raises(ConfigError):
            plan_windows(SAMPLE_RATE, window_s=window, overlap_s=overlap)

    def test_no_samples(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            plan_windows(0)


class TestMergeTexts:
    def test_removes_duplicated_seam_words(self) -> None:
        assert merge_texts(["ina kwana lafiya", "kwana lafiya yau"]) == "ina kwana lafiya yau"

    def test_case_and_punctuation_ignored_at_the_seam(self) -> None:
        assert merge_texts(["Ina kwana.", "ina kwana yau"]) == "Ina kwana. yau"

    def test_tone_marks_ignored_at_the_seam(self) -> None:
        assert merge_texts(["o wà dáadáa", "wa daadaa lo"]) == "o wà dáadáa lo"

    def test_single_word_overlap_is_kept_not_deleted(self) -> None:
        assert merge_texts(["a b", "b c"]) == "a b b c"

    def test_no_overlap_concatenates(self) -> None:
        assert merge_texts(["one two", "three four"]) == "one two three four"

    def test_longest_overlap_wins(self) -> None:
        assert merge_texts(["x a b a b", "a b a b y"]) == "x a b a b y"

    def test_empty_parts_are_skipped(self) -> None:
        assert merge_texts(["", "hello there friend", "  ", "there friend now"]) == (
            "hello there friend now"
        )

    def test_single_and_empty_inputs(self) -> None:
        assert merge_texts(["only one"]) == "only one"
        assert merge_texts([]) == ""

    def test_three_way_merge(self) -> None:
        parts = ["a b c d", "c d e f", "e f g h"]
        assert merge_texts(parts) == "a b c d e f g h"

    def test_seam_words_beyond_the_cap_are_not_merged(self) -> None:
        words = [f"w{i}" for i in range(20)]
        # 20 identical words at the seam exceed the 15-word cap: only the cap is considered,
        # and the shifted comparison finds no match, so nothing is dropped.
        assert len(merge_texts([" ".join(words), " ".join(words)]).split()) >= 20


class FakeBackend:
    """Scripted ASR backend that records the duration of every WAV it receives."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)
        self.durations: list[float] = []

    def generate(self, messages: Sequence[Message], params: GenParams | None = None) -> Generation:
        raise NotImplementedError

    def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript:
        assert isinstance(audio, bytes)
        with wave.open(io.BytesIO(audio), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16_000)
            self.durations.append(wav.getnframes() / wav.getframerate())
        return Transcript(text=self.texts[len(self.durations) - 1], lang=lang, latency_ms=10.0)

    def info(self) -> BackendInfo:
        return BackendInfo(backend="fake", model="fake")

    def close(self) -> None:
        return None


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * SAMPLE_RATE)


class TestTranscribeLong:
    def test_long_audio_is_windowed_and_merged(self) -> None:
        backend = FakeBackend(["ina kwana lafiya", "kwana lafiya yau da", "yau da zuwa gida"])
        result = transcribe_long(backend, b"ignored", "ha", decoder=lambda _a: silence(60))
        assert backend.durations == [28.0, 28.0, 8.0]
        assert result.text == "ina kwana lafiya yau da zuwa gida"
        assert result.lang == "ha"
        assert [(c.start_s, c.end_s) for c in result.chunks] == [
            (0.0, 28.0),
            (26.0, 54.0),
            (52.0, 60.0),
        ]
        assert result.latency_ms == 30.0

    def test_short_audio_is_a_single_call(self) -> None:
        backend = FakeBackend(["sannu"])
        result = transcribe_long(backend, b"x", "ha", decoder=lambda _a: silence(5))
        assert backend.durations == [5.0]
        assert result.text == "sannu"
        assert len(result.chunks) == 1

    def test_chunk_text_is_stripped(self) -> None:
        backend = FakeBackend(["  padded  "])
        result = transcribe_long(backend, b"x", "yo", decoder=lambda _a: silence(1))
        assert result.chunks[0].text == "padded"

    def test_custom_window_and_overlap(self) -> None:
        backend = FakeBackend(["a", "b", "c"])
        transcribe_long(
            backend, b"x", "ig", window_s=10, overlap_s=0, decoder=lambda _a: silence(25)
        )
        assert backend.durations == [10.0, 10.0, 5.0]

    def test_backend_receives_at_most_the_model_limit(self) -> None:
        backend = FakeBackend([f"t{i}" for i in range(40)])
        transcribe_long(backend, b"x", "en", decoder=lambda _a: silence(600))
        assert max(backend.durations) <= 30.0


class TestPcmHelpers:
    def test_wav_roundtrip(self) -> None:
        pcm = silence(0.5)
        with wave.open(io.BytesIO(pcm_to_wav(pcm)), "rb") as wav:
            assert wav.getframerate() == 16_000
            assert wav.readframes(wav.getnframes()) == pcm

    def test_duration(self) -> None:
        assert duration_s(silence(2.5)) == pytest.approx(2.5)

    def test_float32_range(self) -> None:
        samples = pcm_to_float32(struct.pack("<hhh", 0, 32767, -32768))
        assert samples.dtype.name == "float32"
        assert samples[0] == 0.0
        assert samples[1] == pytest.approx(32767 / 32768)
        assert samples[2] == -1.0


class TestDecodeErrors:
    def test_missing_ffmpeg_gives_install_hint(self) -> None:
        with pytest.raises(ResourceError) as info:
            decode_audio(b"x", which=lambda _n: None)
        assert "winget" in (info.value.hint or "")

    @needs_ffmpeg
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(AudioError, match="not found") as info:
            decode_audio(tmp_path / "nope.wav")
        assert info.value.hint is not None

    @needs_ffmpeg
    def test_garbage_bytes(self) -> None:
        with pytest.raises(AudioError, match="could not decode") as info:
            decode_audio(b"this is definitely not audio" * 10)
        assert info.value.hint is not None

    @needs_ffmpeg
    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        with pytest.raises(AudioError):
            decode_audio(empty)


@needs_ffmpeg
class TestDecodeWithRealFfmpeg:
    def test_stereo_44k_wav_becomes_16k_mono(self, tmp_path: Path) -> None:
        pcm = decode_audio(sine_wav(tmp_path / "a.wav", 1.0))
        assert duration_s(pcm) == pytest.approx(1.0, abs=0.05)
        assert len(pcm) % 2 == 0

    def test_accepts_raw_bytes(self, tmp_path: Path) -> None:
        pcm = decode_audio(sine_wav(tmp_path / "a.wav", 0.5).read_bytes())
        assert duration_s(pcm) == pytest.approx(0.5, abs=0.05)

    def test_decoded_audio_is_not_silence(self, tmp_path: Path) -> None:
        samples = pcm_to_float32(decode_audio(sine_wav(tmp_path / "a.wav", 0.5)))
        assert float(abs(samples).max()) > 0.2

    def test_whatsapp_style_ogg_opus(self, tmp_path: Path) -> None:
        wav = sine_wav(tmp_path / "a.wav", 1.0)
        ogg = tmp_path / "note.ogg"
        made = subprocess.run(
            [
                str(shutil.which("ffmpeg")),
                "-v",
                "error",
                "-y",
                "-i",
                str(wav),
                "-c:a",
                "libopus",
                str(ogg),
            ],
            capture_output=True,
            check=False,
        )
        if made.returncode != 0:
            pytest.skip("this ffmpeg build cannot encode opus")
        pcm = decode_audio(ogg)
        assert duration_s(pcm) == pytest.approx(1.0, abs=0.15)

    def test_end_to_end_long_file_through_transcribe_long(self, tmp_path: Path) -> None:
        wav = sine_wav(tmp_path / "long.wav", 65.0, rate=16_000, channels=1)
        backend = FakeBackend([f"t{i}" for i in range(5)])
        result = transcribe_long(backend, wav, "ha")
        assert len(backend.durations) == 3
        assert max(backend.durations) <= 28.0 + 1e-6
        assert result.chunks[-1].end_s == pytest.approx(65.0, abs=0.05)


class TestLongAudioBackend:
    def test_transcribe_splits_long_audio_and_merges(self) -> None:
        inner = FakeBackend(["ina kwana lafiya", "kwana lafiya yau da", "yau da zuwa gida"])
        wrapped = LongAudioBackend(inner, decoder=lambda _a: silence(60))
        result = wrapped.transcribe(b"x", "ha")
        assert inner.durations == [28.0, 28.0, 8.0]
        assert result.text == "ina kwana lafiya yau da zuwa gida"
        assert len(result.chunks) == 3

    def test_window_settings_are_forwarded(self) -> None:
        inner = FakeBackend(["a", "b", "c"])
        wrapped = LongAudioBackend(inner, window_s=10, overlap_s=0, decoder=lambda _a: silence(25))
        assert wrapped.transcribe(b"x", "ig").text == "a b c"
        assert inner.durations == [10.0, 10.0, 5.0]

    def test_everything_else_is_delegated(self) -> None:
        wrapped = LongAudioBackend(FakeBackend([]), decoder=lambda _a: silence(1))
        assert wrapped.info().model == "fake"
        with pytest.raises(NotImplementedError):
            wrapped.generate([{"role": "user", "content": "hi"}])
        wrapped.close()

    def test_invalid_window_settings_fail_at_construction(self) -> None:
        with pytest.raises(ConfigError):
            LongAudioBackend(FakeBackend([]), window_s=45)
