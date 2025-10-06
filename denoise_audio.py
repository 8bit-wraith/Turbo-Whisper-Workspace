import os
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter

# --- Trish's Lab: Audio Science Playground ---
# Configs: tweak here for science!
INPUT_DIR = 'examples/Test1'
OUTPUT_DIR = 'examples/Test1'
INPUT_EXT = '.flac'
HIGHPASS_CUTOFFS = [100, 150, 200]  # Hz, try as many as you want

# --- Utility: Normalize audio to -1..1 range ---
def normalize_audio(y):
    max_val = np.max(np.abs(y))
    return y / max_val if max_val > 0 else y

# --- Utility: High-pass filter ---
def highpass_filter(y, sr, cutoff):
    b, a = butter(N=4, Wn=cutoff/(sr/2), btype='high', analog=False)
    filtered = lfilter(b, a, y)
    return filtered

# --- Configurable directories ---
INPUT_DIR = 'examples/Test1'
OUTPUT_DIR = 'examples/Test1'  # Can be changed if needed
INPUT_EXT = '.m4a'
FINAL_SUFFIX = '_denoised_quiet.wav'

# --- Utility: Normalize audio to -1..1 range ---
def normalize_audio(y):
    max_val = np.max(np.abs(y))
    return y / max_val if max_val > 0 else y

# --- Modular experiment pipeline ---
def process_all_files():
    inputs = [f for f in os.listdir(INPUT_DIR) if f.endswith(INPUT_EXT)]

    for in_file in inputs:
        base = os.path.splitext(in_file)[0]
        in_path = os.path.join(INPUT_DIR, in_file)
        print(f"\nProcessing {in_file} ... Trish, grab your goggles!")
        y, sr = librosa.load(in_path, sr=None, mono=True)

        # --- 1. Original (normalized) ---
        y_norm = normalize_audio(y)
        orig_wav = os.path.join(OUTPUT_DIR, f"{base}_orignorm.wav")
        sf.write(orig_wav, y_norm, sr)
        print(f"Saved: {orig_wav}")
        run_whisper(orig_wav, f"{base}_orignorm_whisper.json")

        # --- 3. Denoised ---
        y_denoised = normalize_audio(nr.reduce_noise(y=y_norm, sr=sr, y_noise=y_norm[:sr]))
        den_wav = os.path.join(OUTPUT_DIR, f"{base}_denoised.wav")
        sf.write(den_wav, y_denoised, sr)
        print(f"Saved: {den_wav}")
        run_whisper(den_wav, f"{base}_denoised_whisper.json")

        # --- 4. Denoised + Quiet ---
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y_denoised, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = np.percentile(rms, 90)
        loud_frames = rms > threshold
        mask = np.ones_like(y_denoised)
        for i, is_loud in enumerate(loud_frames):
            if is_loud:
                start = i * hop_length
                end = min(start + frame_length, len(y_denoised))
                mask[start:end] *= 0.2  # Trish says: "Shhh!"
        quiet_y = normalize_audio(y_denoised * mask)
        quiet_wav = os.path.join(OUTPUT_DIR, f"{base}_denoised_quiet.wav")
        sf.write(quiet_wav, quiet_y, sr)
        print(f"Saved: {quiet_wav}")
        run_whisper(quiet_wav, f"{base}_denoised_quiet_whisper.json")

        # --- 5. High-pass + Denoised + Quiet (for each cutoff) ---
        for cutoff in HIGHPASS_CUTOFFS:
            y_hp = normalize_audio(highpass_filter(y_norm, sr, cutoff))
            y_hp_denoised = normalize_audio(nr.reduce_noise(y=y_hp, sr=sr, y_noise=y_hp[:sr]))
            rms = librosa.feature.rms(y=y_hp_denoised, frame_length=frame_length, hop_length=hop_length)[0]
            threshold = np.percentile(rms, 90)
            loud_frames = rms > threshold
            mask = np.ones_like(y_hp_denoised)
            for i, is_loud in enumerate(loud_frames):
                if is_loud:
                    start = i * hop_length
                    end = min(start + frame_length, len(y_hp_denoised))
                    mask[start:end] *= 0.2
            y_combo = normalize_audio(y_hp_denoised * mask)
            combo_wav = os.path.join(OUTPUT_DIR, f"{base}_highpass{cutoff}_denoised_quiet.wav")
            sf.write(combo_wav, y_combo, sr)
            print(f"Saved: {combo_wav}")
            run_whisper(combo_wav, f"{base}_highpass{cutoff}_denoised_quiet_whisper.json")

    print("\nAll experiments complete! Trish, update the lab notebook!")

# --- Whisper transcription utility ---
def run_whisper(wav_path, transcript_name):
    transcript_path = os.path.join(OUTPUT_DIR, transcript_name)
    if os.path.exists(transcript_path):
        print(f"Skipping transcription for {wav_path} (already exists: {transcript_path})")
        return
    print(f"Transcribing {wav_path} with insanely-fast-whisper...")
    cmd = f"PYTORCH_ENABLE_MPS_FALLBACK=1 insanely-fast-whisper --file-name '{wav_path}' --device-id mps --transcript-path '{transcript_path}'"
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    output_json = os.path.join(OUTPUT_DIR, "output.json")
    if not os.path.exists(transcript_path) and os.path.exists(output_json):
        print(f"Trish Alert: insanely-fast-whisper wrote to output.json instead of {transcript_path}. Renaming...")
        os.rename(output_json, transcript_path)
    if ret != 0:
        print(f"Warning: Transcription failed for {wav_path} (exit code {ret})")
    else:
        print(f"Transcription complete for {wav_path}! Transcript at {transcript_path}")

if __name__ == "__main__":
    process_all_files()
