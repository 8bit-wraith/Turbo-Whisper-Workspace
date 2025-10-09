#!/usr/bin/env python3
"""
Marine-Based Sound Source Localization

Uses Marine Algorithm's precise peak detection for Time Difference of Arrival (TDOA)
triangulation across multiple microphones.

Key Insight: Marine peaks are time-precise. The delta between peak times
on different channels gives us TDOA for source localization.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class MicrophonePosition:
    """3D position of a microphone"""
    x: float
    y: float
    z: float
    channel: int
    name: str = ""

    def distance_to(self, other: 'MicrophonePosition') -> float:
        """Calculate Euclidean distance to another mic"""
        return np.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )


@dataclass
class SoundSource:
    """Detected sound source location"""
    x: float
    y: float
    z: float
    confidence: float
    timestamp: float
    salience: float
    contributing_channels: List[int]


class MarineAlgorithm:
    """
    Simple Marine Algorithm for per-channel peak detection
    (Lightweight version for localization)
    """

    def __init__(self, theta_c: float = 0.01, alpha: float = 0.1):
        self.theta_c = theta_c
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.last_peak_time: Optional[float] = None
        self.ema_period = 0.0
        self.ema_amplitude = 0.0

    def process(self, signal: np.ndarray, sample_rate: int) -> Dict:
        """Process signal and return peak times and salience scores"""
        peaks = []
        scores = []
        times = []

        # Convert to mono if needed
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

        fs = float(sample_rate)

        # Main processing loop - O(1) per sample
        for n in range(1, len(signal) - 1):
            x = signal[n]

            # Pre-gating
            if abs(x) < self.theta_c:
                continue

            # Peak detection: local maximum
            if signal[n-1] < x > signal[n+1]:
                current_time = n / fs
                current_amplitude = abs(x)

                # Jitter-based salience
                if self.last_peak_time is not None:
                    period = current_time - self.last_peak_time
                    jp = abs(period - self.ema_period)
                    ja = abs(current_amplitude - self.ema_amplitude)

                    self.ema_period = (1 - self.alpha) * self.ema_period + self.alpha * period
                    self.ema_amplitude = (1 - self.alpha) * self.ema_amplitude + self.alpha * current_amplitude

                    # Marine salience: low jitter = high salience
                    jitter = jp + ja + 1e-6
                    score = current_amplitude / jitter

                    peaks.append(n)
                    scores.append(score)
                    times.append(current_time)
                else:
                    # First peak - seed EMAs
                    self.ema_period = 0.1
                    self.ema_amplitude = current_amplitude

                self.last_peak_time = current_time

        return {
            'peaks': peaks,
            'scores': scores,
            'times': times,
            'max_salience': max(scores) if scores else 0,
            'mean_salience': np.mean(scores) if scores else 0
        }


class MarineLocalization:
    """
    Multi-channel sound source localization using Marine TDOA

    For each sound event:
    1. Detect peaks on all channels using Marine
    2. Match peaks across channels (similar time, high correlation)
    3. Calculate TDOA (Time Difference of Arrival) for each mic pair
    4. Triangulate source position from TDOAs
    5. Weight by Marine salience scores
    """

    def __init__(
        self,
        mic_positions: List[MicrophonePosition],
        speed_of_sound: float = 343.0,  # m/s at 20°C
        max_tdoa_window: float = 0.05,  # 50ms max TDOA window
        min_salience: float = 1.0
    ):
        """
        Args:
            mic_positions: List of microphone 3D positions
            speed_of_sound: Speed of sound in m/s
            max_tdoa_window: Maximum TDOA window for peak matching (seconds)
            min_salience: Minimum salience for valid peaks
        """
        self.mic_positions = mic_positions
        self.speed_of_sound = speed_of_sound
        self.max_tdoa_window = max_tdoa_window
        self.min_salience = min_salience

        # Create Marine algorithms for each channel
        self.marine_channels = {
            mic.channel: MarineAlgorithm()
            for mic in mic_positions
        }

        # Validate mic setup
        if len(mic_positions) < 3:
            raise ValueError("Need at least 3 microphones for 3D localization")

    def _match_peaks_across_channels(
        self,
        channel_results: Dict[int, Dict]
    ) -> List[Dict]:
        """
        Match peaks across channels based on temporal proximity and salience

        Returns:
            List of matched peak groups
        """
        # Get reference channel (highest mean salience)
        ref_channel = max(
            channel_results.keys(),
            key=lambda ch: channel_results[ch]['mean_salience']
        )

        ref_times = channel_results[ref_channel]['times']
        ref_scores = channel_results[ref_channel]['scores']

        matched_groups = []

        # For each peak in reference channel
        for ref_idx, (ref_time, ref_salience) in enumerate(zip(ref_times, ref_scores)):
            if ref_salience < self.min_salience:
                continue

            peak_group = {
                'reference_time': ref_time,
                'reference_channel': ref_channel,
                'reference_salience': ref_salience,
                'peaks': {ref_channel: ref_time}
            }

            # Find matching peaks in other channels
            for ch, results in channel_results.items():
                if ch == ref_channel:
                    continue

                # Find peaks within TDOA window
                ch_times = np.array(results['times'])
                ch_scores = np.array(results['scores'])

                # Peaks within window
                time_diffs = np.abs(ch_times - ref_time)
                within_window = time_diffs < self.max_tdoa_window

                if np.any(within_window):
                    # Find closest peak with sufficient salience
                    candidates = np.where(
                        within_window & (ch_scores >= self.min_salience)
                    )[0]

                    if len(candidates) > 0:
                        # Choose highest salience
                        best_idx = candidates[np.argmax(ch_scores[candidates])]
                        peak_group['peaks'][ch] = ch_times[best_idx]

            # Only keep if we have at least 3 channels
            if len(peak_group['peaks']) >= 3:
                matched_groups.append(peak_group)

        return matched_groups

    def _triangulate_tdoa(
        self,
        tdoas: Dict[Tuple[int, int], float],
        peak_group: Dict
    ) -> Optional[SoundSource]:
        """
        Triangulate sound source position from TDOAs using least squares

        Args:
            tdoas: Dict mapping (mic_i, mic_j) -> TDOA
            peak_group: Matched peak group

        Returns:
            SoundSource or None if triangulation fails
        """
        # Get microphone positions for contributing channels
        contributing_mics = [
            mic for mic in self.mic_positions
            if mic.channel in peak_group['peaks']
        ]

        if len(contributing_mics) < 3:
            return None

        # Use reference mic as origin
        ref_mic = contributing_mics[0]

        # Build system of hyperbolic equations
        # Each TDOA defines a hyperboloid
        # Intersection of hyperboloids = source position

        # Simplified approach: Grid search over 3D space
        # For production: Use iterative solvers (Gauss-Newton, etc.)

        # Define search grid (assuming mics within 10m cube)
        grid_resolution = 0.1  # 10cm resolution
        x_range = np.arange(-5, 5, grid_resolution)
        y_range = np.arange(-5, 5, grid_resolution)
        z_range = np.arange(-5, 5, grid_resolution)

        # Simplified 2D search for demo (x,y plane at z=mic height)
        avg_z = np.mean([mic.z for mic in contributing_mics])

        best_error = float('inf')
        best_position = None

        # Grid search in x,y plane
        for x in x_range[::5]:  # Coarse grid for speed
            for y in y_range[::5]:
                error = 0.0

                # Calculate predicted TDOAs for this position
                for i, mic_i in enumerate(contributing_mics):
                    for mic_j in contributing_mics[i+1:]:
                        # Distance from source to each mic
                        dist_i = np.sqrt(
                            (x - mic_i.x)**2 +
                            (y - mic_i.y)**2 +
                            (avg_z - mic_i.z)**2
                        )
                        dist_j = np.sqrt(
                            (x - mic_j.x)**2 +
                            (y - mic_j.y)**2 +
                            (avg_z - mic_j.z)**2
                        )

                        # Predicted TDOA
                        predicted_tdoa = (dist_i - dist_j) / self.speed_of_sound

                        # Actual TDOA from peaks
                        actual_tdoa = (
                            peak_group['peaks'][mic_i.channel] -
                            peak_group['peaks'][mic_j.channel]
                        )

                        # Accumulate error
                        error += (predicted_tdoa - actual_tdoa) ** 2

                if error < best_error:
                    best_error = error
                    best_position = (x, y, avg_z)

        if best_position is None:
            return None

        # Calculate confidence based on error and salience
        # Lower error + higher salience = higher confidence
        rmse = np.sqrt(best_error / len(contributing_mics))
        error_confidence = np.exp(-rmse * 10)  # Exponential decay
        salience_confidence = min(peak_group['reference_salience'] / 10, 1.0)
        confidence = error_confidence * salience_confidence

        return SoundSource(
            x=best_position[0],
            y=best_position[1],
            z=best_position[2],
            confidence=confidence,
            timestamp=peak_group['reference_time'],
            salience=peak_group['reference_salience'],
            contributing_channels=[mic.channel for mic in contributing_mics]
        )

    def process_multichannel_audio(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[SoundSource]:
        """
        Process multi-channel audio and localize sound sources

        Args:
            audio: Multi-channel audio (samples x channels)
            sample_rate: Sample rate in Hz

        Returns:
            List of detected sound sources with positions
        """
        if len(audio.shape) != 2:
            raise ValueError("Audio must be multi-channel (samples x channels)")

        num_channels = audio.shape[1]
        print(f"🎵 Processing {num_channels} channels @ {sample_rate}Hz")

        # Process each channel with Marine
        channel_results = {}

        for mic in self.mic_positions:
            if mic.channel >= num_channels:
                print(f"⚠️  Mic channel {mic.channel} not in audio (only {num_channels} channels)")
                continue

            print(f"🌊 Marine processing channel {mic.channel} ({mic.name})...")
            channel_audio = audio[:, mic.channel]

            self.marine_channels[mic.channel].reset()
            result = self.marine_channels[mic.channel].process(channel_audio, sample_rate)

            channel_results[mic.channel] = result
            print(f"   Found {len(result['peaks'])} peaks, max salience: {result['max_salience']:.2f}")

        # Match peaks across channels
        print(f"\n🔗 Matching peaks across channels...")
        matched_groups = self._match_peaks_across_channels(channel_results)
        print(f"   Found {len(matched_groups)} matched peak groups")

        # Triangulate each matched group
        print(f"\n📍 Triangulating source positions...")
        sources = []

        for group_idx, peak_group in enumerate(matched_groups):
            # Calculate TDOAs for all mic pairs
            tdoas = {}
            for mic_i in self.mic_positions:
                for mic_j in self.mic_positions:
                    if mic_i.channel >= mic_j.channel:
                        continue

                    if (mic_i.channel in peak_group['peaks'] and
                        mic_j.channel in peak_group['peaks']):
                        tdoa = (
                            peak_group['peaks'][mic_i.channel] -
                            peak_group['peaks'][mic_j.channel]
                        )
                        tdoas[(mic_i.channel, mic_j.channel)] = tdoa

            # Triangulate
            source = self._triangulate_tdoa(tdoas, peak_group)
            if source:
                sources.append(source)
                print(f"   {group_idx+1}. Position: ({source.x:.2f}, {source.y:.2f}, {source.z:.2f}) "
                      f"| confidence: {source.confidence:.2f} | salience: {source.salience:.1f}")

        print(f"\n✅ Found {len(sources)} sound sources")
        return sources


def demo_marine_localization():
    """Demonstrate Marine sound source localization"""
    print("🎯 MARINE SOUND SOURCE LOCALIZATION")
    print("=" * 70)

    # Define mic array geometry (example: square array on table)
    mic_positions = [
        MicrophonePosition(x=-0.5, y=0.5, z=0.0, channel=0, name="Front-Left"),
        MicrophonePosition(x=0.5, y=0.5, z=0.0, channel=1, name="Front-Right"),
        MicrophonePosition(x=-0.5, y=-0.5, z=0.0, channel=2, name="Rear-Left"),
        MicrophonePosition(x=0.5, y=-0.5, z=0.0, channel=3, name="Rear-Right"),
    ]

    print("🎤 Microphone Array Configuration:")
    for mic in mic_positions:
        print(f"   {mic.name} (Ch {mic.channel}): ({mic.x:.1f}, {mic.y:.1f}, {mic.z:.1f})m")

    # Create localizer
    localizer = MarineLocalization(
        mic_positions=mic_positions,
        speed_of_sound=343.0,
        max_tdoa_window=0.02,  # 20ms
        min_salience=1.0
    )

    # Generate synthetic multi-channel audio
    # Simulate source at position (1.0, 0.0, 0.0)
    print("\n🔊 Simulating sound source at (1.0, 0.0, 0.0)...")

    sample_rate = 48000
    duration = 1.0  # 1 second
    num_samples = int(sample_rate * duration)

    # Generate impulse at t=0.5s
    source_audio = np.zeros(num_samples)
    source_audio[num_samples // 2] = 1.0

    # Simulate propagation to each mic
    source_pos = np.array([1.0, 0.0, 0.0])
    multi_channel = np.zeros((num_samples, len(mic_positions)))

    for i, mic in enumerate(mic_positions):
        mic_pos = np.array([mic.x, mic.y, mic.z])
        distance = np.linalg.norm(source_pos - mic_pos)

        # Time delay
        delay_samples = int((distance / 343.0) * sample_rate)
        delay_samples = min(delay_samples, num_samples - 1)

        # Amplitude attenuation (inverse square law)
        attenuation = 1.0 / (distance ** 2 + 0.1)

        # Place delayed impulse
        if delay_samples < num_samples:
            multi_channel[num_samples // 2 + delay_samples, i] = attenuation

        print(f"   {mic.name}: delay={delay_samples/sample_rate*1000:.2f}ms, "
              f"distance={distance:.2f}m, attenuation={attenuation:.3f}")

    # Add some noise for realism
    multi_channel += np.random.randn(*multi_channel.shape) * 0.01

    print("\n🎯 Localizing...")
    sources = localizer.process_multichannel_audio(multi_channel, sample_rate)

    print("\n📊 LOCALIZATION RESULTS:")
    print(f"   Actual source: (1.0, 0.0, 0.0)")

    if sources:
        for i, source in enumerate(sources, 1):
            error = np.sqrt(
                (source.x - 1.0)**2 +
                (source.y - 0.0)**2 +
                (source.z - 0.0)**2
            )
            print(f"   Detected source {i}: ({source.x:.2f}, {source.y:.2f}, {source.z:.2f}) "
                  f"| error: {error:.2f}m | confidence: {source.confidence:.2f}")
    else:
        print("   ❌ No sources detected")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo_marine_localization()
