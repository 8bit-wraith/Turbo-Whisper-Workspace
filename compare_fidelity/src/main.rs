use anyhow::*;
use clap::Parser;
use hound::{WavWriter, WavSpec, SampleFormat};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::Serialize;
use std::{fs::File, path::PathBuf};

// Marine-Sense integration! 🌊
use marine_algorithm::marine::{Marine, MarineAlgorithm};
use ndarray::Array1;

/// Compare two audio files; optionally restore ultrasonic 'shimmer' with golden-ratio shift.
/// Now with Marine-Sense consciousness detection! 🌊
#[derive(Parser, Debug)]
#[command(name="compare_fidelity")]
struct Args {
    /// Reference (e.g., original WAV)
    ref_file: PathBuf,
    /// Test (e.g., MP3 to analyze or restore)
    test_file: PathBuf,

    /// JSON report output
    #[arg(long, default_value="report.json")]
    out_json: PathBuf,

    /// If set, run golden-ratio ultrasonic resynthesis and write 192k WAV.
    #[arg(long)]
    restore: bool,

    /// Restored WAV path (only if --restore)
    #[arg(long, default_value="restored_192k.wav")]
    out_wav: PathBuf,

    /// Blend gain for shifted band (linear). 0.1 ≈ -20 dB
    #[arg(long, default_value_t = 0.1)]
    blend_gain: f32,

    /// Frame size for STFT (power of two)
    #[arg(long, default_value_t = 2048)]
    frame: usize,

    /// Enable Marine salience analysis
    #[arg(long)]
    marine: bool,
}

#[derive(Serialize, Clone, Copy, Default)]
struct Metrics {
    crest_ratio: f32,
    rms_ratio: f32,
    spectral_centroid_shift: f32,
    hf_energy_ratio: f32,
    // Marine-Sense integration! 🌊
    salience_mean: f32,
    salience_max: f32,
    jitter_mean: f32,
    peak_count_ratio: f32,
    // TIO coherence (future)
    tio_coherence_mean: f32,
    tio_drift_var: f32,
}

#[derive(Clone)]
struct Audio {
    sr: u32,
    mono: Vec<f32>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🌊 Golden-Ratio Ultrasonic Restoration");
    println!("   with Marine-Sense Consciousness Detection");
    println!("═══════════════════════════════════════════\n");

    let a_ref = decode_any(&args.ref_file)?;
    let a_tst = decode_any(&args.test_file)?;

    println!("📊 Analyzing reference: {} Hz, {} samples",
             a_ref.sr, a_ref.mono.len());
    println!("📊 Analyzing test:      {} Hz, {} samples\n",
             a_tst.sr, a_tst.mono.len());

    // Basic metrics (short, fast)
    let ref_m = analyze(&a_ref.mono, a_ref.sr as f32);
    let tst_m = analyze(&a_tst.mono, a_tst.sr as f32);

    // Marine salience analysis! 🌊
    let (ref_marine, tst_marine) = if args.marine {
        println!("🌊 Running Marine salience analysis...");
        let ref_sal = analyze_marine(&a_ref.mono, a_ref.sr);
        let tst_sal = analyze_marine(&a_tst.mono, a_tst.sr);
        println!("   Ref: {:.2} mean salience, {} peaks", ref_sal.mean, ref_sal.peak_count);
        println!("   Tst: {:.2} mean salience, {} peaks\n", tst_sal.mean, tst_sal.peak_count);
        (Some(ref_sal), Some(tst_sal))
    } else {
        (None, None)
    };

    let report = Metrics {
        crest_ratio: tst_m.crest / safed(ref_m.crest),
        rms_ratio: tst_m.rms / safed(ref_m.rms),
        spectral_centroid_shift: tst_m.centroid_hz - ref_m.centroid_hz,
        hf_energy_ratio: tst_m.hf_prop / safed(ref_m.hf_prop),
        salience_mean: tst_marine.as_ref().map(|m| m.mean).unwrap_or(0.0),
        salience_max: tst_marine.as_ref().map(|m| m.max).unwrap_or(0.0),
        jitter_mean: tst_marine.as_ref().map(|m| m.jitter).unwrap_or(0.0),
        peak_count_ratio: if let (Some(r), Some(t)) = (&ref_marine, &tst_marine) {
            t.peak_count as f32 / safed(r.peak_count as f32)
        } else { 0.0 },
        ..Default::default()
    };

    serde_json::to_writer_pretty(File::create(&args.out_json)?, &report)?;
    println!("✓ JSON report → {}", args.out_json.display());

    if args.restore {
        println!("\n🌊 Starting φ-ultrasonic restoration...");
        println!("   Golden ratio: 1.618033988...");
        println!("   Shifting 8-18 kHz → 12.9-29.1 kHz");
        println!("   Blend gain: {} ({:.1} dB)\n",
                 args.blend_gain, 20.0 * args.blend_gain.log10());

        let restored = restore_phi_ultrasonic(&a_tst, args.frame, args.blend_gain)?;
        write_wav_192k(&args.out_wav, &restored)?;
        println!("✓ Restored WAV (192 kHz) → {}", args.out_wav.display());

        // Analyze restored with Marine if enabled
        if args.marine {
            println!("\n🌊 Analyzing restored audio with Marine...");
            let restored_marine = analyze_marine(&restored.mono, restored.sr);
            println!("   Restored: {:.2} mean salience, {} peaks",
                     restored_marine.mean, restored_marine.peak_count);
        }
    }

    println!("\n✅ Complete!");
    Ok(())
}

fn decode_any(path: &PathBuf) -> Result<Audio> {
    use symphonia::core::{
        audio::{AudioBufferRef, Signal},
        codecs::DecoderOptions,
        formats::FormatOptions,
        io::MediaSourceStream,
        meta::MetadataOptions,
        probe::Hint,
    };

    let src = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format.default_track().ok_or_else(|| anyhow!("no default track"))?;
    let dec_opts = DecoderOptions { verify: false, ..Default::default() };
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

    let mut sr: u32 = track.codec_params.sample_rate.ok_or_else(|| anyhow!("unknown sr"))?;
    let mut mono = Vec::<f32>::new();

    loop {
        match format.next_packet() {
            std::result::Result::Ok(pkt) => {
                match decoder.decode(&pkt) {
                    std::result::Result::Ok(decoded) => {
                        sr = match decoded {
                            AudioBufferRef::F32(buf) => {
                                let rate = buf.spec().rate;
                                let ch = buf.spec().channels.count();
                                let frames = buf.frames();
                                // Convert to mono by averaging channels
                                for i in 0..frames {
                                    let mut sum = 0.0f32;
                                    for c in 0..ch {
                                        sum += buf.chan(c)[i];
                                    }
                                    mono.push(sum / ch as f32);
                                }
                                rate
                            }
                            AudioBufferRef::F64(buf) => {
                                let rate = buf.spec().rate;
                                let ch = buf.spec().channels.count();
                                let frames = buf.frames();
                                // Convert to mono by averaging channels
                                for i in 0..frames {
                                    let mut sum = 0.0f64;
                                    for c in 0..ch {
                                        sum += buf.chan(c)[i];
                                    }
                                    mono.push((sum / ch as f64) as f32);
                                }
                                rate
                            }
                            AudioBufferRef::S32(buf) => {
                                let rate = buf.spec().rate;
                                let frames = buf.frames();
                                for &s in buf.chan(0).iter().take(frames) {
                                    mono.push(s as f32 / i32::MAX as f32);
                                }
                                rate
                            }
                            AudioBufferRef::S24(buf) => {
                                let rate = buf.spec().rate;
                                let frames = buf.frames();
                                for &s in buf.chan(0).iter().take(frames) {
                                    let val = s.into_i32();
                                    mono.push(val as f32 / 8_388_607.0);
                                }
                                rate
                            }
                            AudioBufferRef::U8(buf) => {
                                let rate = buf.spec().rate;
                                let frames = buf.frames();
                                for &s in buf.chan(0).iter().take(frames) {
                                    mono.push((s as f32 - 128.0) / 128.0);
                                }
                                rate
                            }
                            AudioBufferRef::S16(buf) => {
                                let rate = buf.spec().rate;
                                let frames = buf.frames();
                                for &s in buf.chan(0).iter().take(frames) {
                                    mono.push(s as f32 / i16::MAX as f32);
                                }
                                rate
                            }
                            _ => {
                                eprintln!("Warning: Unsupported audio format, skipping packet");
                                sr
                            }
                        };
                    }
                    std::result::Result::Err(_) => break
                }
            }
            std::result::Result::Err(_) => break
        }
    }
    Ok(Audio { sr, mono })
}

fn safed(x: f32) -> f32 { if x.abs() < 1e-9 { 1e-9 } else { x } }

#[derive(Clone, Copy)]
struct Quick {
    rms: f32,
    crest: f32,
    centroid_hz: f32,
    hf_prop: f32,
}

fn analyze(x: &[f32], sr: f32) -> Quick {
    if x.is_empty() {
        return Quick { rms: 0.0, crest: 0.0, centroid_hz: 0.0, hf_prop: 0.0 };
    }
    let n = x.len().min(8192).max(128); // Handle tiny files
    let slice = &x[..n];
    let rms = (slice.iter().map(|v| v*v).sum::<f32>() / n as f32).sqrt();
    let peak = slice.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let crest = peak / safed(rms);

    // FFT magnitude
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<Complex<f32>> = slice.iter().map(|&v| Complex{re:v, im:0.0}).collect();
    fft.process(&mut buf);
    let mags: Vec<f32> = buf.iter().take(n/2).map(|c| c.norm()).collect();
    let sum = mags.iter().sum::<f32>();
    let centroid_bin = mags.iter().enumerate().map(|(i,&m)| i as f32 * m).sum::<f32>() / safed(sum);
    let centroid_hz = centroid_bin * (sr / n as f32);

    // high-frequency proportion (top 25% of bins)
    let start = (mags.len() as f32 * 0.75) as usize;
    let hf = mags[start..].iter().sum::<f32>() / safed(sum);

    Quick { rms, crest, centroid_hz, hf_prop: hf }
}

// 🌊 MARINE INTEGRATION! 🌊
#[derive(Clone, Copy, Default)]
struct MarineMetrics {
    mean: f32,
    max: f32,
    jitter: f32,
    peak_count: usize,
}

fn analyze_marine(x: &[f32], sr: u32) -> MarineMetrics {
    let arr = Array1::from_vec(x.to_vec());
    let mut marine = MarineAlgorithm::new(0.015, 0.15);
    let result = marine.process(&arr, sr);

    MarineMetrics {
        mean: result.mean_salience(),
        max: result.max_salience(),
        jitter: if result.metadata.peak_count > 0 {
            result.metadata.total_jitter / result.metadata.peak_count as f32
        } else { 0.0 },
        peak_count: result.peaks.len(),
    }
}

/// Golden-ratio ultrasonic resynthesis:
/// 1) upsample to 192k (linear)
/// 2) STFT; copy band >=8k, shift by φ, add at blend_gain
fn restore_phi_ultrasonic(a: &Audio, frame: usize, blend_gain: f32) -> Result<Audio> {
    let target_sr = 192_000u32;
    let up = upsample_linear(&a.mono, a.sr, target_sr);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(frame);
    let ifft = planner.plan_fft_inverse(frame);
    let hop = frame / 4;
    let phi = 1.618_033_988_75_f32;

    let mut out = vec![0f32; up.len()];
    let win: Vec<f32> = (0..frame).map(|i| {
        // Hamming
        0.54 - 0.46 * (std::f32::consts::TAU * i as f32 / (frame as f32 - 1.0)).cos()
    }).collect();
    let wnorm = 1.0 / win.iter().map(|v| v*v).sum::<f32>().sqrt();

    let nyq = target_sr as f32 / 2.0;
    let f_bin = target_sr as f32 / frame as f32;

    let mut i = 0usize;
    let mut frame_count = 0;
    while i + frame <= up.len() {
        // windowed frame
        let mut buf: Vec<Complex<f32>> = (0..frame)
            .map(|k| Complex{ re: up[i+k] * win[k], im: 0.0 })
            .collect();
        fft.process(&mut buf);

        // copy upper audible band (>=8k) and shift by φ
        let spec = buf.clone();
        let mut shifted = spec.clone();
        let half = frame/2;

        for k in 0..half {
            let f = k as f32 * f_bin;
            if f >= 8000.0 && f <= 18000.0 {
                let f2 = (f * phi).min(nyq - 100.0);
                let k2 = (f2 / f_bin).round() as usize;
                if k2 < half {
                    // add small amount at shifted bin
                    shifted[k2].re += spec[k].re * blend_gain;
                    shifted[k2].im += spec[k].im * blend_gain;
                }
            }
        }

        // iFFT
        let mut time: Vec<Complex<f32>> = shifted;
        ifft.process(&mut time);
        for k in 0..frame {
            out[i+k] += time[k].re * wnorm; // OLA
        }
        i += hop;
        frame_count += 1;

        // Progress indicator
        if frame_count % 1000 == 0 {
            let progress = (i as f32 / up.len() as f32) * 100.0;
            print!("\r   Processing: {:.1}%", progress);
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
    println!("\r   Processing: 100.0%   ");

    Ok(Audio { sr: target_sr, mono: out })
}

/// simple linear upsampler to target_sr
fn upsample_linear(x: &[f32], sr_in: u32, sr_out: u32) -> Vec<f32> {
    if sr_in == sr_out { return x.to_vec(); }
    let ratio = sr_out as f32 / sr_in as f32;
    let n_out = (x.len() as f32 * ratio).round() as usize;
    let mut y = vec![0f32; n_out];
    for n in 0..n_out {
        let t = n as f32 / ratio;
        let i = t.floor() as usize;
        let frac = t - i as f32;
        let a = x.get(i).copied().unwrap_or(0.0);
        let b = x.get(i+1).copied().unwrap_or(a);
        y[n] = a + (b - a) * frac;
    }
    y
}

fn write_wav_192k(path: &PathBuf, a: &Audio) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: a.sr,
        bits_per_sample: 24,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec)?;
    for &s in &a.mono {
        let clamped = (s.max(-1.0).min(1.0) * 8_388_607.0).round() as i32;
        w.write_sample(clamped)?;
    }
    w.finalize()?;
    Ok(())
}
