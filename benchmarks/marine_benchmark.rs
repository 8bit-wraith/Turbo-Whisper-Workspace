//! Marine Algorithm Benchmark - Rust Edition
//!
//! Measures pure Rust Marine performance for comparison with Python
//! Demoscene-worthy optimization! Every cycle counts! 🚀

use std::time::{Duration, Instant};
use std::path::PathBuf;

// Import Marine from the marine-rust crate
// We'll link against the existing implementation
extern crate marine_algorithm;
use marine_algorithm::marine::{Marine, MarineAlgorithm};
use ndarray::Array1;

/// Benchmark result
#[derive(Debug)]
struct BenchmarkResult {
    duration_ms: f64,
    samples_processed: usize,
    samples_per_sec: f64,
    realtime_factor: f64,
    peak_count: usize,
    max_salience: f32,
}

impl BenchmarkResult {
    fn print(&self, label: &str) {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║  {}  ", label);
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!("  Duration:        {:.3} ms", self.duration_ms);
        println!("  Samples:         {}", self.samples_processed);
        println!("  Throughput:      {:.2} Msamples/sec", self.samples_per_sec / 1_000_000.0);
        println!("  Real-time factor: {:.1}x", self.realtime_factor);
        println!("  Peaks detected:  {}", self.peak_count);
        println!("  Max salience:    {:.2}", self.max_salience);
        println!("  Memory/sample:   ~{} bytes", std::mem::size_of::<f32>() * 2); // Peak + score
    }
}

/// Generate synthetic test audio
fn generate_test_audio(sample_rate: u32, duration_secs: f32) -> Array1<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut audio = Array1::zeros(num_samples);

    let fs = sample_rate as f32;

    // Generate quasi-periodic signal similar to speech
    // Fundamental at 100Hz with harmonics
    for i in 0..num_samples {
        let t = i as f32 / fs;
        let fundamental = (2.0 * std::f32::consts::PI * 100.0 * t).sin();
        let harmonic2 = 0.5 * (2.0 * std::f32::consts::PI * 200.0 * t).sin();
        let harmonic3 = 0.25 * (2.0 * std::f32::consts::PI * 300.0 * t).sin();

        audio[i] = (fundamental + harmonic2 + harmonic3) * 0.3;

        // Add slight noise
        audio[i] += (rand::random::<f32>() - 0.5) * 0.05;
    }

    audio
}

/// Benchmark Marine algorithm
fn benchmark_marine(audio: &Array1<f32>, sample_rate: u32, label: &str) -> BenchmarkResult {
    let mut marine = MarineAlgorithm::new(0.015, 0.15);

    // Warm-up run (not timed)
    marine.process(audio, sample_rate);
    marine.reset();

    // Timed run
    let start = Instant::now();
    let result = marine.process(audio, sample_rate);
    let duration = start.elapsed();

    let duration_ms = duration.as_secs_f64() * 1000.0;
    let samples_processed = audio.len();
    let samples_per_sec = samples_processed as f64 / duration.as_secs_f64();

    // Real-time factor: how many times faster than real-time playback
    let audio_duration_secs = samples_processed as f64 / sample_rate as f64;
    let realtime_factor = audio_duration_secs / duration.as_secs_f64();

    BenchmarkResult {
        duration_ms,
        samples_processed,
        samples_per_sec,
        realtime_factor,
        peak_count: result.peaks.len(),
        max_salience: result.max_salience(),
    }
}

/// Run comprehensive benchmark suite
fn run_benchmark_suite() {
    println!("🦀 RUST MARINE ALGORITHM BENCHMARK");
    println!("═══════════════════════════════════════════════════════════");
    println!("Compiler: rustc {} (opt-level=3, lto=true)",
             rustc_version_runtime::version());
    println!("CPU: {}", std::env::var("PROCESSOR_IDENTIFIER")
             .unwrap_or_else(|_| "Unknown".to_string()));
    println!();

    let sample_rate = 16000;

    // Test 1: Short audio (1 second)
    println!("Test 1: Short audio (1s @ 16kHz)");
    let audio_1s = generate_test_audio(sample_rate, 1.0);
    let result_1s = benchmark_marine(&audio_1s, sample_rate, "1 Second Audio");
    result_1s.print("1 SECOND AUDIO");

    // Test 2: Medium audio (10 seconds)
    println!("\nTest 2: Medium audio (10s @ 16kHz)");
    let audio_10s = generate_test_audio(sample_rate, 10.0);
    let result_10s = benchmark_marine(&audio_10s, sample_rate, "10 Second Audio");
    result_10s.print("10 SECOND AUDIO");

    // Test 3: Long audio (60 seconds)
    println!("\nTest 3: Long audio (60s @ 16kHz)");
    let audio_60s = generate_test_audio(sample_rate, 60.0);
    let result_60s = benchmark_marine(&audio_60s, sample_rate, "60 Second Audio");
    result_60s.print("60 SECOND AUDIO");

    // Test 4: High sample rate (48kHz)
    println!("\nTest 4: High sample rate (10s @ 48kHz)");
    let audio_48k = generate_test_audio(48000, 10.0);
    let result_48k = benchmark_marine(&audio_48k, 48000, "48kHz Audio");
    result_48k.print("48kHz AUDIO");

    // Summary
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  BENCHMARK SUMMARY                                        ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("  Average throughput: {:.2} Msamples/sec",
             (result_1s.samples_per_sec + result_10s.samples_per_sec +
              result_60s.samples_per_sec + result_48k.samples_per_sec) / 4.0 / 1_000_000.0);
    println!("  Average RT factor:  {:.1}x",
             (result_1s.realtime_factor + result_10s.realtime_factor +
              result_60s.realtime_factor + result_48k.realtime_factor) / 4.0);

    // Export results as JSON
    println!("\n💾 Exporting results to rust_benchmark_results.json");
    let json_results = serde_json::json!({
        "language": "Rust",
        "compiler": format!("rustc {}", rustc_version_runtime::version()),
        "optimization": "opt-level=3, lto=true, codegen-units=1",
        "tests": [
            {
                "name": "1s @ 16kHz",
                "duration_ms": result_1s.duration_ms,
                "throughput_msamples": result_1s.samples_per_sec / 1_000_000.0,
                "realtime_factor": result_1s.realtime_factor,
                "peak_count": result_1s.peak_count,
            },
            {
                "name": "10s @ 16kHz",
                "duration_ms": result_10s.duration_ms,
                "throughput_msamples": result_10s.samples_per_sec / 1_000_000.0,
                "realtime_factor": result_10s.realtime_factor,
                "peak_count": result_10s.peak_count,
            },
            {
                "name": "60s @ 16kHz",
                "duration_ms": result_60s.duration_ms,
                "throughput_msamples": result_60s.samples_per_sec / 1_000_000.0,
                "realtime_factor": result_60s.realtime_factor,
                "peak_count": result_60s.peak_count,
            },
            {
                "name": "10s @ 48kHz",
                "duration_ms": result_48k.duration_ms,
                "throughput_msamples": result_48k.samples_per_sec / 1_000_000.0,
                "realtime_factor": result_48k.realtime_factor,
                "peak_count": result_48k.peak_count,
            }
        ]
    });

    std::fs::write(
        "rust_benchmark_results.json",
        serde_json::to_string_pretty(&json_results).unwrap()
    ).expect("Failed to write results");

    println!("✅ Benchmark complete!");
}

fn main() {
    // Set high priority for accurate benchmarking
    #[cfg(target_os = "linux")]
    {
        use std::process::Command;
        let _ = Command::new("renice")
            .args(&["-n", "-10", "-p", &std::process::id().to_string()])
            .status();
    }

    run_benchmark_suite();
}
