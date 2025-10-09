#!/usr/bin/env python3
"""
Marine Features Demonstration

Shows all Marine-Sense integration capabilities:
1. Marine VAD (Voice Activity Detection)
2. Sound Source Localization
3. Salience-enhanced transcription
4. Full Marine-enhanced pipeline
"""

import sys
from pathlib import Path

def demo_marine_vad():
    """Demonstrate Marine VAD"""
    print("\n" + "="*70)
    print("1️⃣  MARINE VAD (VOICE ACTIVITY DETECTION)")
    print("="*70)

    from vocalis.marine import MarineVAD
    import numpy as np

    # Create test audio: silence -> voice -> silence
    sr = 16000
    silence1 = np.random.randn(sr // 2) * 0.001  # 0.5s silence
    t = np.linspace(0, 1.0, sr)
    voice = (np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)) * 0.3  # 1s voice
    silence2 = np.random.randn(sr // 2) * 0.001  # 0.5s silence
    audio = np.concatenate([silence1, voice, silence2])

    # Create VAD
    vad = MarineVAD(salience_threshold=1.0)

    # Process
    print(f"🎵 Processing {len(audio)/sr:.1f}s test audio...")
    segments = vad.process_audio(audio, sr)

    # Results
    voice_segs = [s for s in segments if s.state.name == 'VOICE']
    silence_segs = [s for s in segments if s.state.name == 'SILENCE']

    print(f"\n✅ VAD Results:")
    print(f"   Total segments: {len(segments)}")
    print(f"   Voice segments: {len(voice_segs)}")
    print(f"   Silence segments: {len(silence_segs)}")

    total_voice = sum(s.end_time - s.start_time for s in voice_segs)
    print(f"   Voice duration: {total_voice:.2f}s")

    print(f"\n📊 Segment Details:")
    for i, seg in enumerate(segments[:5], 1):
        print(f"   {i}. {seg.start_time:.2f}s - {seg.end_time:.2f}s: "
              f"{seg.state.name:10s} | conf: {seg.confidence:.2f} | "
              f"salience: {seg.mean_salience:.1f}")


def demo_sound_localization():
    """Demonstrate sound source localization"""
    print("\n" + "="*70)
    print("2️⃣  SOUND SOURCE LOCALIZATION")
    print("="*70)

    from vocalis.marine import MarineLocalization, MicrophonePosition
    import numpy as np

    # Define mic array (1m square)
    mics = [
        MicrophonePosition(x=-0.5, y=0.5, z=0.0, channel=0, name="Front-Left"),
        MicrophonePosition(x=0.5, y=0.5, z=0.0, channel=1, name="Front-Right"),
        MicrophonePosition(x=-0.5, y=-0.5, z=0.0, channel=2, name="Rear-Left"),
        MicrophonePosition(x=0.5, y=-0.5, z=0.0, channel=3, name="Rear-Right"),
    ]

    print("🎤 Microphone Array:")
    for mic in mics:
        print(f"   {mic.name:15s} (Ch {mic.channel}): ({mic.x:5.1f}, {mic.y:5.1f}, {mic.z:5.1f})m")

    # Create synthetic 4-channel audio with source at (1, 0, 0)
    sr = 48000
    duration = 0.5
    num_samples = int(sr * duration)

    source_pos = np.array([1.0, 0.0, 0.0])
    multi_channel = np.zeros((num_samples, 4))

    print(f"\n🔊 Simulating sound source at ({source_pos[0]:.1f}, {source_pos[1]:.1f}, {source_pos[2]:.1f})m")

    # Simulate impulse propagation to each mic
    for i, mic in enumerate(mics):
        mic_pos = np.array([mic.x, mic.y, mic.z])
        distance = np.linalg.norm(source_pos - mic_pos)
        delay_samples = int((distance / 343.0) * sr)
        attenuation = 1.0 / (distance ** 2 + 0.1)

        if delay_samples < num_samples // 2:
            multi_channel[num_samples // 2 + delay_samples, i] = attenuation
            print(f"   {mic.name:15s}: distance={distance:.2f}m, delay={delay_samples/sr*1000:.2f}ms")

    # Add noise
    multi_channel += np.random.randn(*multi_channel.shape) * 0.01

    # Localize
    localizer = MarineLocalization(
        mic_positions=mics,
        max_tdoa_window=0.05,
        min_salience=0.5
    )

    print(f"\n🎯 Localizing sources...")
    sources = localizer.process_multichannel_audio(multi_channel, sr)

    print(f"\n✅ Localization Results:")
    print(f"   Actual source: (1.0, 0.0, 0.0)m")

    if sources:
        for i, src in enumerate(sources, 1):
            error = np.sqrt((src.x - 1.0)**2 + (src.y - 0.0)**2 + (src.z - 0.0)**2)
            print(f"   Detected source {i}: ({src.x:.2f}, {src.y:.2f}, {src.z:.2f})m "
                  f"| error: {error:.2f}m | confidence: {src.confidence:.2f}")
    else:
        print("   ⚠️ No sources detected (may need parameter tuning)")


def demo_salience_scoring():
    """Demonstrate Marine salience scoring"""
    print("\n" + "="*70)
    print("3️⃣  MARINE SALIENCE SCORING")
    print("="*70)

    from vocalis.marine import MarineAlgorithm
    import numpy as np

    # Create test audio with varying salience
    sr = 16000
    t = np.linspace(0, 3.0, sr * 3)

    # Low salience (random noise)
    noise = np.random.randn(sr) * 0.1

    # High salience (stable periodic pattern)
    stable = np.sin(2 * np.pi * 100 * t[:sr]) * 0.3

    # Medium salience (modulated pattern)
    modulated = np.sin(2 * np.pi * 100 * t[:sr]) * (0.2 + 0.1 * np.sin(2 * np.pi * 3 * t[:sr]))

    audio = np.concatenate([noise, stable, modulated])

    # Process with Marine
    marine = MarineAlgorithm()
    result = marine.process(audio, sr)

    print(f"🌊 Marine Algorithm Results:")
    print(f"   Total peaks: {len(result['peaks'])}")
    print(f"   Max salience: {result['max_salience']:.2f}")
    print(f"   Mean salience: {result['mean_salience']:.2f}")

    # Find top salience moments
    if result['scores']:
        threshold = np.percentile(result['scores'], 90)
        top_moments = [(t, s) for t, s in zip(result['times'], result['scores']) if s >= threshold]

        print(f"\n✨ Top Salience Moments (90th percentile: {threshold:.2f}):")
        for i, (time, salience) in enumerate(top_moments[:10], 1):
            region = "Noise" if time < 1.0 else "Stable" if time < 2.0 else "Modulated"
            print(f"   {i}. {time:.2f}s: salience={salience:.2f} ({region})")


def demo_full_pipeline():
    """Demonstrate full Marine-enhanced pipeline"""
    print("\n" + "="*70)
    print("4️⃣  MARINE-ENHANCED AUDIO PIPELINE")
    print("="*70)

    print("ℹ️  This demo requires:")
    print("   • A real audio file")
    print("   • Loaded transcription models")
    print("   • Proper environment setup")
    print("")
    print("📖 For full pipeline example, see:")
    print("   vocalis/core/marine_enhanced_pipeline.py::demo_marine_pipeline()")
    print("")
    print("Basic usage:")
    print("""
    from vocalis.core.marine_enhanced_pipeline import MarineEnhancedPipeline

    # Create enhanced pipeline
    pipeline = MarineEnhancedPipeline(enable_marine_vad=True)

    # Process with full Marine features
    result = pipeline.process_with_full_marine(
        "audio.wav",
        task="transcribe",
        enable_vad=True,
        enable_salience=True,
        num_speakers=2
    )

    # Results include:
    # - Transcription with diarization
    # - VAD segments and statistics
    # - Salience scores per segment
    # - "Consciousness moments" (key parts)
    """)


def main():
    """Run all Marine feature demos"""
    print("🌊 MARINE-SENSE INTEGRATION DEMONSTRATION")
    print("Advanced Audio Processing with O(1) Salience Detection")
    print("")

    try:
        demo_marine_vad()
    except Exception as e:
        print(f"\n❌ VAD Demo failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        demo_sound_localization()
    except Exception as e:
        print(f"\n❌ Localization Demo failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        demo_salience_scoring()
    except Exception as e:
        print(f"\n❌ Salience Demo failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        demo_full_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline Demo failed: {e}")

    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    print("")
    print("📚 For more information, see:")
    print("   • MARINE_VAD_LOCALIZATION.md - Full documentation")
    print("   • vocalis/marine/ - Implementation code")
    print("   • tests/test_marine_features.py - Unit tests")
    print("")
    print("🚀 Marine-Sense brings O(1) real-time audio intelligence!")


if __name__ == "__main__":
    main()
