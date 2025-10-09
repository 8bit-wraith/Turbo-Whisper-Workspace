#!/usr/bin/env python3
"""
Marine-Based Voice Activity Detection (VAD)

Uses Marine Algorithm's jitter-based salience for robust VAD:
- Low jitter + high salience = VOICE
- High jitter or low salience = SILENCE/NOISE
- O(1) per sample - real-time capable
- Self-adapting to noise floor
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Optional import for demo
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class VadState(Enum):
    """Voice activity states"""
    SILENCE = 0
    VOICE = 1
    UNCERTAIN = 2


@dataclass
class VadSegment:
    """Voice activity segment"""
    start_time: float
    end_time: float
    state: VadState
    confidence: float
    mean_salience: float
    peak_count: int


class MarineVAD:
    """
    Voice Activity Detection using Marine Algorithm

    The Marine algorithm detects stable patterns (low jitter).
    Speech has stable fundamental frequencies = low jitter = high salience.
    Noise/silence has random patterns = high jitter = low salience.
    """

    def __init__(
        self,
        theta_c: float = 0.015,
        alpha: float = 0.15,
        salience_threshold: float = 2.0,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.2,
        hangover_frames: int = 10
    ):
        """
        Args:
            theta_c: Energy threshold (fraction of RMS)
            alpha: EMA smoothing factor
            salience_threshold: Minimum salience for voice activity
            min_speech_duration: Minimum speech segment length (seconds)
            min_silence_duration: Minimum silence segment length (seconds)
            hangover_frames: Frames to keep voice active after last peak
        """
        self.theta_c = theta_c
        self.alpha = alpha
        self.salience_threshold = salience_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.hangover_frames = hangover_frames

        # Marine state
        self.last_peak_time: Optional[float] = None
        self.ema_period = 0.0
        self.ema_amplitude = 0.0

        # VAD state
        self.current_state = VadState.SILENCE
        self.hangover_counter = 0
        self.segment_start_time = 0.0
        self.segment_scores: List[float] = []
        self.segment_peaks = 0

        # Adaptive threshold
        self.salience_history: List[float] = []
        self.max_history = 1000

    def reset(self):
        """Reset VAD state for new audio"""
        self.last_peak_time = None
        self.ema_period = 0.0
        self.ema_amplitude = 0.0
        self.current_state = VadState.SILENCE
        self.hangover_counter = 0
        self.segment_start_time = 0.0
        self.segment_scores = []
        self.segment_peaks = 0
        self.salience_history = []

    def _is_peak(self, prev: float, current: float, next: float) -> bool:
        """Check if current sample is a local maximum"""
        return prev < current > next

    def _update_ema(self, current: float, ema: float) -> float:
        """Update exponential moving average"""
        return (1 - self.alpha) * ema + self.alpha * current

    def _adapt_threshold(self):
        """Adapt salience threshold based on recent history"""
        if len(self.salience_history) < 100:
            return self.salience_threshold

        # Use median + 1.5*MAD for robust threshold
        recent = self.salience_history[-500:]
        median = np.median(recent)
        mad = np.median(np.abs(np.array(recent) - median))

        # Adaptive threshold: median + 1.5*MAD
        adaptive_threshold = median + 1.5 * mad

        # Don't go too low
        return max(adaptive_threshold, self.salience_threshold * 0.5)

    def process_frame(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_offset: float = 0.0
    ) -> Tuple[VadState, float]:
        """
        Process audio frame and return VAD decision

        Args:
            audio: Audio samples (mono)
            sample_rate: Sample rate in Hz
            frame_offset: Time offset of this frame (seconds)

        Returns:
            (state, confidence) tuple
        """
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Calculate RMS for adaptive threshold
        rms = np.sqrt(np.mean(audio ** 2))
        energy_threshold = self.theta_c * rms

        fs = float(sample_rate)
        frame_scores = []

        # Process audio using Marine algorithm
        for n in range(1, len(audio) - 1):
            x = audio[n]

            # Pre-gating: skip low-energy samples
            if abs(x) < energy_threshold:
                continue

            # Peak detection
            if self._is_peak(audio[n-1], x, audio[n+1]):
                current_time = frame_offset + (n / fs)
                current_amplitude = abs(x)

                # Calculate salience using jitter
                if self.last_peak_time is not None:
                    period = current_time - self.last_peak_time
                    jp = abs(period - self.ema_period)  # Period jitter
                    ja = abs(current_amplitude - self.ema_amplitude)  # Amplitude jitter

                    # Update EMAs
                    self.ema_period = self._update_ema(period, self.ema_period)
                    self.ema_amplitude = self._update_ema(current_amplitude, self.ema_amplitude)

                    # Marine salience: low jitter = high salience
                    jitter = jp + ja + 1e-6
                    score = current_amplitude / jitter

                    frame_scores.append(score)
                    self.salience_history.append(score)

                    # Limit history size
                    if len(self.salience_history) > self.max_history:
                        self.salience_history = self.salience_history[-self.max_history:]

                    self.segment_peaks += 1
                else:
                    # First peak - initialize EMAs
                    # Speech typically has ~100-150ms pitch period
                    self.ema_period = 0.01  # 10ms initial (100Hz fundamental)
                    self.ema_amplitude = current_amplitude

                self.last_peak_time = current_time

        # Make VAD decision based on salience scores
        if frame_scores:
            mean_score = np.mean(frame_scores)
            max_score = np.max(frame_scores)
            self.segment_scores.extend(frame_scores)

            # Get adaptive threshold
            threshold = self._adapt_threshold()

            # Voice detection criteria:
            # 1. High mean salience (stable patterns)
            # 2. Multiple peaks (speech has regular structure)
            if mean_score > threshold and len(frame_scores) >= 2:
                confidence = min(mean_score / (threshold * 2), 1.0)
                self.hangover_counter = self.hangover_frames
                return VadState.VOICE, confidence

            # Uncertain state
            elif mean_score > threshold * 0.5:
                confidence = mean_score / threshold
                if self.hangover_counter > 0:
                    self.hangover_counter -= 1
                    return VadState.VOICE, confidence * 0.7
                return VadState.UNCERTAIN, confidence

        # No significant peaks or low salience = silence
        if self.hangover_counter > 0:
            self.hangover_counter -= 1
            return VadState.VOICE, 0.3  # Hangover period

        return VadState.SILENCE, 0.0

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_size: int = 512,
        hop_size: int = 256
    ) -> List[VadSegment]:
        """
        Process entire audio file and return VAD segments

        Args:
            audio: Audio samples
            sample_rate: Sample rate
            frame_size: Frame size in samples
            hop_size: Hop size in samples (overlap)

        Returns:
            List of VAD segments
        """
        self.reset()

        segments: List[VadSegment] = []
        current_segment_state = VadState.SILENCE
        current_segment_start = 0.0
        current_segment_scores = []
        current_segment_peaks = 0

        # Process in frames with overlap
        num_frames = (len(audio) - frame_size) // hop_size + 1

        for i in range(num_frames):
            start_idx = i * hop_size
            end_idx = start_idx + frame_size

            if end_idx > len(audio):
                break

            frame = audio[start_idx:end_idx]
            frame_time = start_idx / sample_rate

            state, confidence = self.process_frame(frame, sample_rate, frame_time)

            # State change detection
            if state != current_segment_state:
                # End current segment
                if current_segment_scores:
                    segment_duration = frame_time - current_segment_start

                    # Apply minimum duration filters
                    is_valid = False
                    if current_segment_state == VadState.VOICE:
                        is_valid = segment_duration >= self.min_speech_duration
                    elif current_segment_state == VadState.SILENCE:
                        is_valid = segment_duration >= self.min_silence_duration
                    else:
                        is_valid = True  # Always include uncertain segments

                    if is_valid:
                        segments.append(VadSegment(
                            start_time=current_segment_start,
                            end_time=frame_time,
                            state=current_segment_state,
                            confidence=np.mean(current_segment_scores) if current_segment_scores else 0.0,
                            mean_salience=np.mean(current_segment_scores) if current_segment_scores else 0.0,
                            peak_count=current_segment_peaks
                        ))

                # Start new segment
                current_segment_state = state
                current_segment_start = frame_time
                current_segment_scores = [confidence]
                current_segment_peaks = self.segment_peaks
                self.segment_peaks = 0
            else:
                # Continue current segment
                current_segment_scores.append(confidence)

        # Close final segment
        if current_segment_scores:
            segments.append(VadSegment(
                start_time=current_segment_start,
                end_time=len(audio) / sample_rate,
                state=current_segment_state,
                confidence=np.mean(current_segment_scores),
                mean_salience=np.mean(current_segment_scores),
                peak_count=current_segment_peaks
            ))

        return segments

    def get_voice_segments(self, segments: List[VadSegment]) -> List[Tuple[float, float]]:
        """
        Extract just the voice segments as (start, end) tuples

        Args:
            segments: List of VAD segments

        Returns:
            List of (start_time, end_time) for voice segments
        """
        return [
            (seg.start_time, seg.end_time)
            for seg in segments
            if seg.state == VadState.VOICE
        ]


def demo_marine_vad():
    """Demonstrate Marine VAD on sample audio"""
    from pathlib import Path

    if not SOUNDFILE_AVAILABLE:
        print("⚠️ soundfile not installed - using synthetic audio")
        # Generate synthetic audio instead
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2)
        audio = np.sin(2 * np.pi * 100 * t) * 0.3
        audio_file = "synthetic"
    else:
        print("🌊 MARINE VAD DEMONSTRATION")
        print("=" * 60)

        # Find test audio
        test_files = [
            "/aidata/Turbo-Whisper-Workspace/test_scarlett.wav",
            "/aidata/ayeverse/Marine-Sense/test_scarlett.wav",
        ]

        audio_file = None
        for f in test_files:
            if Path(f).exists():
                audio_file = f
                break

        if not audio_file:
            print("❌ No test audio found!")
            return

        print(f"📁 Processing: {audio_file}")

        # Load audio
        audio, sr = sf.read(audio_file)

    # Create VAD
    vad = MarineVAD(
        salience_threshold=1.5,
        min_speech_duration=0.1,
        min_silence_duration=0.2
    )

    print(f"🎵 Audio: {len(audio)/sr:.2f}s @ {sr}Hz")
    print(f"⚙️  Processing with Marine VAD...")

    # Process
    segments = vad.process_audio(audio, sr)

    # Display results
    print(f"\n📊 VAD RESULTS:")
    print(f"   Total segments: {len(segments)}")

    voice_segments = [s for s in segments if s.state == VadState.VOICE]
    silence_segments = [s for s in segments if s.state == VadState.SILENCE]

    print(f"   Voice segments: {len(voice_segments)}")
    print(f"   Silence segments: {len(silence_segments)}")

    total_voice_time = sum(s.end_time - s.start_time for s in voice_segments)
    total_duration = audio.shape[0] / sr

    print(f"\n⏱️  Duration Analysis:")
    print(f"   Total: {total_duration:.2f}s")
    print(f"   Voice: {total_voice_time:.2f}s ({total_voice_time/total_duration*100:.1f}%)")
    print(f"   Silence: {total_duration - total_voice_time:.2f}s ({(1-total_voice_time/total_duration)*100:.1f}%)")

    print(f"\n🎯 Voice Segments:")
    for i, seg in enumerate(voice_segments[:10], 1):  # Show first 10
        duration = seg.end_time - seg.start_time
        print(f"   {i}. {seg.start_time:.2f}s - {seg.end_time:.2f}s "
              f"({duration:.2f}s) | confidence: {seg.confidence:.2f} | "
              f"salience: {seg.mean_salience:.1f} | peaks: {seg.peak_count}")

    if len(voice_segments) > 10:
        print(f"   ... and {len(voice_segments) - 10} more")

    print("\n✅ Marine VAD complete!")
    print("\nKey Advantages:")
    print("  • O(1) complexity - real-time capable")
    print("  • Self-adapting to noise floor")
    print("  • Robust to varying speaking styles")
    print("  • No machine learning required")


if __name__ == "__main__":
    demo_marine_vad()
