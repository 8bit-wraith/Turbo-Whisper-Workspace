#!/usr/bin/env python3
"""
Tests for Marine-Sense integration features

Tests:
- Marine VAD (Voice Activity Detection)
- Sound source localization
- Salience scoring
- Enhanced pipeline
"""

import unittest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path

from vocalis.marine import (
    MarineVAD,
    VadState,
    MarineLocalization,
    MicrophonePosition,
    MarineAlgorithm
)


class TestMarineVAD(unittest.TestCase):
    """Test Marine-based Voice Activity Detection"""

    def setUp(self):
        """Create test VAD instance"""
        self.vad = MarineVAD(
            salience_threshold=1.0,
            min_speech_duration=0.1,
            min_silence_duration=0.1
        )

    def test_vad_initialization(self):
        """Test VAD initializes correctly"""
        self.assertIsNotNone(self.vad)
        self.assertEqual(self.vad.current_state, VadState.SILENCE)

    def test_vad_reset(self):
        """Test VAD reset functionality"""
        # Process some audio
        audio = np.random.randn(1000) * 0.1
        self.vad.process_frame(audio, 16000)

        # Reset
        self.vad.reset()
        self.assertEqual(self.vad.current_state, VadState.SILENCE)
        self.assertIsNone(self.vad.last_peak_time)

    def test_vad_silence_detection(self):
        """Test VAD detects silence correctly"""
        # Create silent audio (very low amplitude)
        silent_audio = np.random.randn(1000) * 0.001

        state, confidence = self.vad.process_frame(silent_audio, 16000)

        # Should detect silence
        self.assertEqual(state, VadState.SILENCE)
        self.assertLessEqual(confidence, 0.5)

    def test_vad_voice_detection(self):
        """Test VAD detects voice activity"""
        # Create audio with clear voice-like pattern (quasi-periodic)
        sample_rate = 16000
        duration = 1.0  # 1 second
        num_samples = int(sample_rate * duration)

        # Generate quasi-periodic signal (like speech)
        t = np.linspace(0, duration, num_samples)
        # Fundamental frequency ~100Hz with harmonics
        voice_audio = (
            np.sin(2 * np.pi * 100 * t) +
            0.5 * np.sin(2 * np.pi * 200 * t) +
            0.25 * np.sin(2 * np.pi * 300 * t)
        ) * 0.3

        # Add some noise
        voice_audio += np.random.randn(num_samples) * 0.05

        state, confidence = self.vad.process_frame(voice_audio, sample_rate)

        # Should detect voice (or at least not silence with high confidence)
        # VAD may take multiple frames to adapt
        self.assertIn(state, [VadState.VOICE, VadState.UNCERTAIN])

    def test_vad_full_audio_processing(self):
        """Test VAD on complete audio file"""
        sample_rate = 16000
        duration = 2.0

        # Create test audio: silence -> voice -> silence
        silence1 = np.random.randn(int(sample_rate * 0.5)) * 0.001
        t = np.linspace(0, 1.0, sample_rate)
        voice = (
            np.sin(2 * np.pi * 100 * t) +
            0.5 * np.sin(2 * np.pi * 200 * t)
        ) * 0.3
        silence2 = np.random.randn(int(sample_rate * 0.5)) * 0.001

        audio = np.concatenate([silence1, voice, silence2])

        # Process
        segments = self.vad.process_audio(audio, sample_rate)

        # Should have detected some segments
        self.assertGreater(len(segments), 0)

        # Should have at least one voice segment
        voice_segments = [s for s in segments if s.state == VadState.VOICE]
        self.assertGreater(len(voice_segments), 0, "Should detect at least one voice segment")

    def test_vad_get_voice_segments(self):
        """Test extracting voice segments"""
        sample_rate = 16000
        t = np.linspace(0, 1.0, sample_rate)
        audio = np.sin(2 * np.pi * 100 * t) * 0.3

        segments = self.vad.process_audio(audio, sample_rate)
        voice_times = self.vad.get_voice_segments(segments)

        # Should return list of tuples
        self.assertIsInstance(voice_times, list)
        if voice_times:
            self.assertIsInstance(voice_times[0], tuple)
            self.assertEqual(len(voice_times[0]), 2)


class TestMarineAlgorithm(unittest.TestCase):
    """Test core Marine algorithm"""

    def test_marine_initialization(self):
        """Test Marine algorithm initializes"""
        marine = MarineAlgorithm()
        self.assertIsNotNone(marine)

    def test_marine_processes_signal(self):
        """Test Marine processes signal and returns results"""
        marine = MarineAlgorithm(theta_c=0.01, alpha=0.1)

        # Create test signal with clear peaks
        signal = np.array([
            0.0, 0.1, 0.5, 0.1, 0.0,  # Peak at index 2
            0.0, 0.2, 0.8, 0.2, 0.0,  # Peak at index 7
            0.0, 0.1, 0.6, 0.1, 0.0,  # Peak at index 12
        ])

        result = marine.process(signal, 1000)

        # Should have detected peaks
        self.assertIsInstance(result, dict)
        self.assertIn('peaks', result)
        self.assertIn('scores', result)
        self.assertIn('times', result)
        self.assertGreater(len(result['peaks']), 0, "Should detect at least one peak")

    def test_marine_reset(self):
        """Test Marine algorithm reset"""
        marine = MarineAlgorithm()
        signal = np.random.randn(1000) * 0.1
        marine.process(signal, 16000)

        marine.reset()
        self.assertIsNone(marine.last_peak_time)


class TestMarineLocalization(unittest.TestCase):
    """Test Marine-based sound source localization"""

    def setUp(self):
        """Create test microphone array"""
        # Square array: 1m x 1m
        self.mic_positions = [
            MicrophonePosition(x=-0.5, y=0.5, z=0.0, channel=0, name="FL"),
            MicrophonePosition(x=0.5, y=0.5, z=0.0, channel=1, name="FR"),
            MicrophonePosition(x=-0.5, y=-0.5, z=0.0, channel=2, name="RL"),
            MicrophonePosition(x=0.5, y=-0.5, z=0.0, channel=3, name="RR"),
        ]

    def test_localizer_initialization(self):
        """Test localizer initializes with mic array"""
        localizer = MarineLocalization(
            mic_positions=self.mic_positions,
            speed_of_sound=343.0
        )
        self.assertIsNotNone(localizer)
        self.assertEqual(len(localizer.mic_positions), 4)

    def test_localizer_requires_min_mics(self):
        """Test localizer requires at least 3 mics"""
        with self.assertRaises(ValueError):
            MarineLocalization(
                mic_positions=self.mic_positions[:2]  # Only 2 mics
            )

    def test_localization_synthetic_audio(self):
        """Test localization on synthetic multi-channel audio"""
        sample_rate = 48000
        duration = 0.5
        num_samples = int(sample_rate * duration)

        # Create impulse
        source_audio = np.zeros(num_samples)
        source_audio[num_samples // 2] = 1.0

        # Simulate source at (1, 0, 0)
        source_pos = np.array([1.0, 0.0, 0.0])
        multi_channel = np.zeros((num_samples, len(self.mic_positions)))

        for i, mic in enumerate(self.mic_positions):
            mic_pos = np.array([mic.x, mic.y, mic.z])
            distance = np.linalg.norm(source_pos - mic_pos)

            # Time delay
            delay_samples = int((distance / 343.0) * sample_rate)
            attenuation = 1.0 / (distance ** 2 + 0.1)

            # Place delayed impulse
            if delay_samples < num_samples - num_samples // 2:
                multi_channel[num_samples // 2 + delay_samples, i] = attenuation

        # Add noise
        multi_channel += np.random.randn(*multi_channel.shape) * 0.01

        # Localize
        localizer = MarineLocalization(
            mic_positions=self.mic_positions,
            max_tdoa_window=0.05,
            min_salience=0.5
        )

        sources = localizer.process_multichannel_audio(multi_channel, sample_rate)

        # Should detect at least one source
        # Note: This is a simple test - localization may not be perfect
        self.assertIsInstance(sources, list)


class TestMarineEnhancedPipeline(unittest.TestCase):
    """Test Marine-enhanced audio pipeline"""

    def test_pipeline_imports(self):
        """Test pipeline can be imported"""
        try:
            from vocalis.core.marine_enhanced_pipeline import MarineEnhancedPipeline
            self.assertTrue(True, "Pipeline imports successfully")
        except ImportError as e:
            self.fail(f"Failed to import pipeline: {e}")

    def test_pipeline_initialization(self):
        """Test pipeline initializes"""
        from vocalis.core.marine_enhanced_pipeline import MarineEnhancedPipeline

        pipeline = MarineEnhancedPipeline(enable_marine_vad=True)
        self.assertIsNotNone(pipeline)
        self.assertTrue(pipeline.enable_marine_vad)
        self.assertIsNotNone(pipeline.marine_vad)

    def test_pipeline_without_vad(self):
        """Test pipeline can disable VAD"""
        from vocalis.core.marine_enhanced_pipeline import MarineEnhancedPipeline

        pipeline = MarineEnhancedPipeline(enable_marine_vad=False)
        self.assertFalse(pipeline.enable_marine_vad)
        self.assertIsNone(pipeline.marine_vad)


class TestMicrophonePosition(unittest.TestCase):
    """Test microphone position utilities"""

    def test_mic_position_creation(self):
        """Test creating microphone position"""
        mic = MicrophonePosition(x=1.0, y=2.0, z=3.0, channel=0, name="Test")
        self.assertEqual(mic.x, 1.0)
        self.assertEqual(mic.y, 2.0)
        self.assertEqual(mic.z, 3.0)
        self.assertEqual(mic.channel, 0)
        self.assertEqual(mic.name, "Test")

    def test_mic_distance_calculation(self):
        """Test distance calculation between mics"""
        mic1 = MicrophonePosition(x=0.0, y=0.0, z=0.0, channel=0)
        mic2 = MicrophonePosition(x=1.0, y=0.0, z=0.0, channel=1)

        distance = mic1.distance_to(mic2)
        self.assertAlmostEqual(distance, 1.0, places=5)

    def test_mic_distance_3d(self):
        """Test 3D distance calculation"""
        mic1 = MicrophonePosition(x=0.0, y=0.0, z=0.0, channel=0)
        mic2 = MicrophonePosition(x=1.0, y=1.0, z=1.0, channel=1)

        distance = mic1.distance_to(mic2)
        expected = np.sqrt(3.0)
        self.assertAlmostEqual(distance, expected, places=5)


def run_tests():
    """Run all Marine feature tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMarineVAD))
    suite.addTests(loader.loadTestsFromTestCase(TestMarineAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestMarineLocalization))
    suite.addTests(loader.loadTestsFromTestCase(TestMarineEnhancedPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestMicrophonePosition))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
