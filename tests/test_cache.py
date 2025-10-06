import os
import json
import tempfile
from pathlib import Path
import types
import sys


def create_dummy_audio(path: str, seconds: float = 0.1):
    # Minimal valid WAV header with silence; avoid extra deps
    import wave
    import struct

    framerate = 16000
    nframes = int(seconds * framerate)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        silence_frame = struct.pack('<h', 0)
        for _ in range(nframes):
            wf.writeframes(silence_frame)


def test_pipeline_cache_roundtrip(tmp_path: Path, monkeypatch):
    # Use isolated cache dir
    cache_dir = tmp_path / ".tw_cache"
    monkeypatch.setenv("TW_CACHE_DIR", str(cache_dir))

    # Create dummy audio file
    audio_path = tmp_path / "dummy.wav"
    create_dummy_audio(str(audio_path), seconds=0.05)

    # Provide a minimal fake torch to satisfy imports without installing torch
    fake_torch = types.ModuleType("torch")
    class _FakeCuda:
        @staticmethod
        def is_available():
            return False
    fake_torch.cuda = _FakeCuda()
    sys.modules.setdefault("torch", fake_torch)

    # Import after stubbing torch
    from vocalis.core.audio_pipeline import AudioProcessingPipeline

    # Stub heavy methods to avoid real model work
    pipeline = AudioProcessingPipeline()
    # Pre-create a lightweight fake diarizer to stop load_diarizer from importing heavy deps
    class _FakeDiarizer:
        def __init__(self):
            self.segmentation_model = "seg"
            self.embedding_model = "emb"
            self.num_speakers = 2
            self.threshold = 0.5

    pipeline.diarizer = _FakeDiarizer()

    def fake_transcribe(audio_path: str, task: str = "transcribe", return_timestamps: bool = True):
        return {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0},
            ],
        }

    def fake_diarize(audio_path: str, num_speakers: int = 2):
        return [
            {"speaker": "Speaker 0", "start": 0.0, "end": 0.5, "score": 1.0},
            {"speaker": "Speaker 1", "start": 0.5, "end": 1.0, "score": 1.0},
        ]

    pipeline.transcribe = fake_transcribe  # type: ignore
    pipeline.diarize = fake_diarize  # type: ignore

    # First run: should compute and write cache
    res1 = pipeline.process_audio(
        audio_path=str(audio_path),
        task="transcribe",
        segmentation_model="seg",
        embedding_model="emb",
        num_speakers=2,
        threshold=0.5,
        use_cache=True,
        force_reprocess=False,
    )

    assert "error" not in res1
    assert res1.get("text") == "hello world"
    assert res1.get("merged_segments")
    assert not res1.get("from_cache", False)

    # Second run: should hit cache
    res2 = pipeline.process_audio(
        audio_path=str(audio_path),
        task="transcribe",
        segmentation_model="seg",
        embedding_model="emb",
        num_speakers=2,
        threshold=0.5,
        use_cache=True,
        force_reprocess=False,
    )

    assert "error" not in res2
    assert res2.get("text") == "hello world"
    assert res2.get("from_cache") is True

    # Force reprocess should bypass cache
    res3 = pipeline.process_audio(
        audio_path=str(audio_path),
        task="transcribe",
        segmentation_model="seg",
        embedding_model="emb",
        num_speakers=2,
        threshold=0.5,
        use_cache=True,
        force_reprocess=True,
    )

    assert "error" not in res3
    assert res3.get("from_cache", False) is False


