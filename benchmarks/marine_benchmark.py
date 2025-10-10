#!/usr/bin/env python3
"""
Marine Algorithm Benchmark - Python Edition

Matches the Rust benchmark for fair comparison.
Let's see how Python's interpreted bytecode stacks up! 🐍
"""

import sys
import time
import json
import platform
import numpy as np
from typing import Dict, List

# Import our Marine implementation
sys.path.insert(0, '/aidata/Turbo-Whisper-Workspace')
from vocalis.marine.marine_localization import MarineAlgorithm


class BenchmarkResult:
    """Benchmark result container"""

    def __init__(
        self,
        duration_ms: float,
        samples_processed: int,
        samples_per_sec: float,
        realtime_factor: float,
        peak_count: int,
        max_salience: float
    ):
        self.duration_ms = duration_ms
        self.samples_processed = samples_processed
        self.samples_per_sec = samples_per_sec
        self.realtime_factor = realtime_factor
        self.peak_count = peak_count
        self.max_salience = max_salience

    def print_result(self, label: str):
        """Print benchmark result"""
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print(f"║  {label:<57}║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print(f"  Duration:        {self.duration_ms:.3f} ms")
        print(f"  Samples:         {self.samples_processed}")
        print(f"  Throughput:      {self.samples_per_sec / 1_000_000:.2f} Msamples/sec")
        print(f"  Real-time factor: {self.realtime_factor:.1f}x")
        print(f"  Peaks detected:  {self.peak_count}")
        print(f"  Max salience:    {self.max_salience:.2f}")
        print(f"  Memory/sample:   ~{np.dtype(np.float32).itemsize * 2} bytes")

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            'duration_ms': float(self.duration_ms),
            'throughput_msamples': float(self.samples_per_sec / 1_000_000),
            'realtime_factor': float(self.realtime_factor),
            'peak_count': int(self.peak_count),
            'max_salience': float(self.max_salience)
        }


def generate_test_audio(sample_rate: int, duration_secs: float) -> np.ndarray:
    """
    Generate synthetic test audio

    Must match the Rust implementation exactly for fair comparison!
    """
    num_samples = int(sample_rate * duration_secs)
    t = np.linspace(0, duration_secs, num_samples, dtype=np.float32)

    # Quasi-periodic signal (speech-like)
    # Fundamental at 100Hz with harmonics
    fundamental = np.sin(2 * np.pi * 100 * t)
    harmonic2 = 0.5 * np.sin(2 * np.pi * 200 * t)
    harmonic3 = 0.25 * np.sin(2 * np.pi * 300 * t)

    audio = (fundamental + harmonic2 + harmonic3) * 0.3

    # Add slight noise
    noise = (np.random.rand(num_samples).astype(np.float32) - 0.5) * 0.05
    audio += noise

    return audio


def benchmark_marine(audio: np.ndarray, sample_rate: int, label: str) -> BenchmarkResult:
    """
    Benchmark Marine algorithm

    Matches Rust implementation methodology
    """
    marine = MarineAlgorithm(theta_c=0.015, alpha=0.15)

    # Warm-up run (not timed) - let JIT warm up if possible
    marine.process(audio, sample_rate)
    marine.reset()

    # Timed run
    start_time = time.perf_counter()
    result = marine.process(audio, sample_rate)
    end_time = time.perf_counter()

    duration = end_time - start_time
    duration_ms = duration * 1000.0
    samples_processed = len(audio)
    samples_per_sec = samples_processed / duration

    # Real-time factor
    audio_duration_secs = samples_processed / sample_rate
    realtime_factor = audio_duration_secs / duration

    return BenchmarkResult(
        duration_ms=duration_ms,
        samples_processed=samples_processed,
        samples_per_sec=samples_per_sec,
        realtime_factor=realtime_factor,
        peak_count=len(result['peaks']),
        max_salience=result['max_salience']
    )


def run_benchmark_suite():
    """Run comprehensive benchmark suite"""
    print("🐍 PYTHON MARINE ALGORITHM BENCHMARK")
    print("═══════════════════════════════════════════════════════════")
    print(f"Python:      {platform.python_version()}")
    print(f"NumPy:       {np.__version__}")
    print(f"Platform:    {platform.platform()}")
    print(f"Processor:   {platform.processor()}")
    print()

    sample_rate = 16000

    # Test 1: Short audio (1 second)
    print("Test 1: Short audio (1s @ 16kHz)")
    audio_1s = generate_test_audio(sample_rate, 1.0)
    result_1s = benchmark_marine(audio_1s, sample_rate, "1 Second Audio")
    result_1s.print_result("1 SECOND AUDIO")

    # Test 2: Medium audio (10 seconds)
    print("\nTest 2: Medium audio (10s @ 16kHz)")
    audio_10s = generate_test_audio(sample_rate, 10.0)
    result_10s = benchmark_marine(audio_10s, sample_rate, "10 Second Audio")
    result_10s.print_result("10 SECOND AUDIO")

    # Test 3: Long audio (60 seconds)
    print("\nTest 3: Long audio (60s @ 16kHz)")
    audio_60s = generate_test_audio(sample_rate, 60.0)
    result_60s = benchmark_marine(audio_60s, sample_rate, "60 Second Audio")
    result_60s.print_result("60 SECOND AUDIO")

    # Test 4: High sample rate (48kHz)
    print("\nTest 4: High sample rate (10s @ 48kHz)")
    audio_48k = generate_test_audio(48000, 10.0)
    result_48k = benchmark_marine(audio_48k, 48000, "48kHz Audio")
    result_48k.print_result("48kHz AUDIO")

    # Summary
    results = [result_1s, result_10s, result_60s, result_48k]
    avg_throughput = np.mean([r.samples_per_sec for r in results])
    avg_rt_factor = np.mean([r.realtime_factor for r in results])

    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║  BENCHMARK SUMMARY                                        ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  Average throughput: {avg_throughput / 1_000_000:.2f} Msamples/sec")
    print(f"  Average RT factor:  {avg_rt_factor:.1f}x")

    # Export results as JSON
    print("\n💾 Exporting results to python_benchmark_results.json")
    json_results = {
        'language': 'Python',
        'version': platform.python_version(),
        'numpy_version': np.__version__,
        'platform': platform.platform(),
        'tests': [
            {'name': '1s @ 16kHz', **result_1s.to_dict()},
            {'name': '10s @ 16kHz', **result_10s.to_dict()},
            {'name': '60s @ 16kHz', **result_60s.to_dict()},
            {'name': '10s @ 48kHz', **result_48k.to_dict()},
        ]
    }

    with open('python_benchmark_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print("✅ Benchmark complete!")

    return json_results


if __name__ == '__main__':
    # Try to set higher priority (best effort)
    try:
        import os
        os.nice(-10)
    except (PermissionError, AttributeError):
        pass

    run_benchmark_suite()
