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

    /// Restored audio path (supports .flac or .wav)
    #[arg(long, default_value="restored_192k.flac")]
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

    /// Normalize output to prevent clipping (recommended)
    #[arg(long, default_value_t = true)]
    normalize: bool,
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
    channels: Vec<Vec<f32>>,  // Support multiple channels (stereo, mono, etc)
}

impl Audio {
    fn mono(&self) -> Vec<f32> {
        if self.channels.is_empty() {
            return Vec::new();
        }
        if self.channels.len() == 1 {
            return self.channels[0].clone();
        }
        // Average all channels to mono
        let len = self.channels[0].len();
        let mut mono = vec![0.0f32; len];
        for ch in &self.channels {
            for (i, &s) in ch.iter().enumerate() {
                mono[i] += s;
            }
        }
        for s in &mut mono {
            *s /= self.channels.len() as f32;
        }
        mono
    }

    fn channel_count(&self) -> usize {
        self.channels.len()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🌊 Golden-Ratio Ultrasonic Restoration");
    println!("   with Marine-Sense Consciousness Detection");
    println!("═══════════════════════════════════════════\n");

    let a_ref = decode_any(&args.ref_file)?;
    let a_tst = decode_any(&args.test_file)?;

    println!("📊 Analyzing reference: {} Hz, {} ch, {} samples",
             a_ref.sr, a_ref.channel_count(), a_ref.mono().len());
    println!("📊 Analyzing test:      {} Hz, {} ch, {} samples\n",
             a_tst.sr, a_tst.channel_count(), a_tst.mono().len());

    // Basic metrics (short, fast) - analyze mono downmix
    let ref_mono = a_ref.mono();
    let tst_mono = a_tst.mono();
    let ref_m = analyze(&ref_mono, a_ref.sr as f32);
    let tst_m = analyze(&tst_mono, a_tst.sr as f32);

    // Marine salience analysis! 🌊
    let (ref_marine, tst_marine) = if args.marine {
        println!("🌊 Running Marine salience analysis...");
        let ref_sal = analyze_marine(&ref_mono, a_ref.sr);
        let tst_sal = analyze_marine(&tst_mono, a_tst.sr);
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

        let mut restored = restore_phi_ultrasonic(&a_tst, args.frame, args.blend_gain)?;

        // Normalize to prevent clipping (check across all channels)
        if args.normalize {
            let mut peak = 0.0f32;
            for ch in &restored.channels {
                let ch_peak = ch.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                peak = peak.max(ch_peak);
            }
            if peak > 1.0 {
                let gain = 0.95 / peak; // Leave 5% headroom
                for ch in &mut restored.channels {
                    for s in ch {
                        *s *= gain;
                    }
                }
                println!("   Normalized by {:.2} dB to prevent clipping", 20.0 * gain.log10());
            }
        }

        write_audio(&args.out_wav, &restored)?;
        println!("✓ Restored ({} ch, 192 kHz) → {}", restored.channel_count(), args.out_wav.display());

        // Analyze restored with Marine if enabled
        if args.marine {
            println!("\n🌊 Analyzing restored audio with Marine...");
            let restored_mono = restored.mono();
            let restored_marine = analyze_marine(&restored_mono, restored.sr);
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
    let mut channels: Vec<Vec<f32>> = Vec::new();

    loop {
        match format.next_packet() {
            std::result::Result::Ok(pkt) => {
                match decoder.decode(&pkt) {
                    std::result::Result::Ok(decoded) => {
                        sr = match decoded {
                            AudioBufferRef::F32(buf) => {
                                let rate = buf.spec().rate;
                                let ch_count = buf.spec().channels.count();
                                let frames = buf.frames();
                                // Initialize channels on first packet
                                if channels.is_empty() {
                                    channels = vec![Vec::new(); ch_count];
                                }
                                // Preserve all channels
                                for c in 0..ch_count {
                                    channels[c].extend_from_slice(&buf.chan(c)[..frames]);
                                }
                                rate
                            }
                            AudioBufferRef::F64(buf) => {
                                let rate = buf.spec().rate;
                                let ch_count = buf.spec().channels.count();
                                let frames = buf.frames();
                                if channels.is_empty() {
                                    channels = vec![Vec::new(); ch_count];
                                }
                                for c in 0..ch_count {
                                    channels[c].extend(buf.chan(c).iter().take(frames).map(|&s| s as f32));
                                }
                                rate
                            }
                            AudioBufferRef::S32(buf) => {
                                let rate = buf.spec().rate;
                                let ch_count = buf.spec().channels.count();
                                let frames = buf.frames();
                                if channels.is_empty() {
                                    channels = vec![Vec::new(); ch_count];
                                }
                                for c in 0..ch_count {
                                    channels[c].extend(buf.chan(c).iter().take(frames).map(|&s| s as f32 / i32::MAX as f32));
                                }
                                rate
                            }
                            AudioBufferRef::S24(buf) => {
                                let rate = buf.spec().rate;
                                let ch_count = buf.spec().channels.count();
                                let frames = buf.frames();
                                if channels.is_empty() {
                                    channels = vec![Vec::new(); ch_count];
                                }
                                for c in 0..ch_count {
                                    channels[c].extend(buf.chan(c).iter().take(frames).map(|&s| s.into_i32() as f32 / 8_388_607.0));
                                }
                                rate
                            }
                            AudioBufferRef::U8(buf) => {
                                let rate = buf.spec().rate;
                                let ch_count = buf.spec().channels.count();
                                let frames = buf.frames();
                                if channels.is_empty() {
                                    channels = vec![Vec::new(); ch_count];
                                }
                                for c in 0..ch_count {
                                    channels[c].extend(buf.chan(c).iter().take(frames).map(|&s| (s as f32 - 128.0) / 128.0));
                                }
                                rate
                            }
                            AudioBufferRef::S16(buf) => {
                                let rate = buf.spec().rate;
                                let ch_count = buf.spec().channels.count();
                                let frames = buf.frames();
                                if channels.is_empty() {
                                    channels = vec![Vec::new(); ch_count];
                                }
                                for c in 0..ch_count {
                                    channels[c].extend(buf.chan(c).iter().take(frames).map(|&s| s as f32 / i16::MAX as f32));
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
    Ok(Audio { sr, channels })
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
/// Now processes ALL channels independently (stereo support!)
fn restore_phi_ultrasonic(a: &Audio, frame: usize, blend_gain: f32) -> Result<Audio> {
    let target_sr = 192_000u32;
    let mut restored_channels = Vec::new();

    // Process each channel independently
    for (ch_idx, channel_data) in a.channels.iter().enumerate() {
        let up = upsample_linear(channel_data, a.sr, target_sr);

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

            // Progress indicator (only for first channel to avoid spam)
            if ch_idx == 0 && frame_count % 1000 == 0 {
                let progress = (i as f32 / up.len() as f32) * 100.0;
                print!("\r   Processing: {:.1}%", progress);
                use std::io::Write;
                std::io::stdout().flush().unwrap();
            }
        }

        restored_channels.push(out);
    }
    println!("\r   Processing: 100.0%   ");

    Ok(Audio { sr: target_sr, channels: restored_channels })
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

/// Write audio to file - supports multichannel WAV and FLAC
fn write_audio(path: &PathBuf, a: &Audio) -> Result<()> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("wav");

    match ext.to_lowercase().as_str() {
        "flac" => write_flac(path, a),
        _ => write_wav(path, a), // Default to WAV
    }
}

/// Write multichannel WAV file (24-bit)
fn write_wav(path: &PathBuf, a: &Audio) -> Result<()> {
    if a.channels.is_empty() {
        return Err(anyhow!("No audio channels to write"));
    }

    let spec = WavSpec {
        channels: a.channels.len() as u16,
        sample_rate: a.sr,
        bits_per_sample: 24,
        sample_format: SampleFormat::Int,
    };

    let mut w = WavWriter::create(path, spec)?;
    let frames = a.channels[0].len();

    // Interleave channels
    for frame_idx in 0..frames {
        for ch in &a.channels {
            let sample = ch.get(frame_idx).copied().unwrap_or(0.0);
            let clamped = (sample.max(-1.0).min(1.0) * 8_388_607.0).round() as i32;
            w.write_sample(clamped)?;
        }
    }

    w.finalize()?;
    Ok(())
}

/// Write FLAC using ffmpeg (simple wrapper)
fn write_flac(path: &PathBuf, a: &Audio) -> Result<()> {
    // Write temp WAV first
    let temp_wav = path.with_extension("temp.wav");
    write_wav(&temp_wav, a)?;

    // Convert to FLAC using ffmpeg
    let status = std::process::Command::new("ffmpeg")
        .args(&[
            "-y", // Overwrite
            "-i", temp_wav.to_str().unwrap(),
            "-c:a", "flac",
            "-compression_level", "12", // Maximum compression
            path.to_str().unwrap(),
        ])
        .stderr(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .status()?;

    // Clean up temp file
    std::fs::remove_file(&temp_wav)?;

    if !status.success() {
        return Err(anyhow!("ffmpeg FLAC encoding failed"));
    }

    Ok(())
}
