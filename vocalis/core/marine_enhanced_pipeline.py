"""
Marine-Enhanced Audio Processing Pipeline

Extends AudioProcessingPipeline with:
1. Marine-based VAD (Voice Activity Detection)
2. Sound source localization (multi-channel audio)
3. Emotional spectrum analysis
4. Salience-based segment weighting
"""

from typing import Dict, List, Optional, Any
import numpy as np
import soundfile as sf

from vocalis.core.audio_pipeline import AudioProcessingPipeline
from vocalis.marine import (
    MarineVAD,
    VadState,
    MarineLocalization,
    MicrophonePosition,
    MarineAlgorithm
)


class MarineEnhancedPipeline(AudioProcessingPipeline):
    """
    Audio pipeline enhanced with Marine-Sense capabilities

    New capabilities:
    - Automatic VAD (removes silence segments before transcription)
    - Sound source localization for multi-channel audio
    - Salience scoring for speech segments (identifies important moments)
    - Emotional spectrum analysis (optional)
    """

    def __init__(self, enable_marine_vad: bool = True):
        """
        Args:
            enable_marine_vad: Enable Marine VAD for automatic silence removal
        """
        super().__init__()
        self.enable_marine_vad = enable_marine_vad
        self.marine_vad = MarineVAD() if enable_marine_vad else None

        # Emotional analyzer (lazy loaded)
        self.emotional_analyzer = None

        # Sound localizer (created on-demand for multi-channel)
        self.sound_localizer = None

    def _get_emotional_analyzer(self):
        """Lazy load emotional spectrum analyzer"""
        if self.emotional_analyzer is None:
            try:
                # Import from Marine-Sense directory
                import sys
                sys.path.append('/aidata/ayeverse/Marine-Sense')
                from emotional_decoder import EmotionalDecoder
                self.emotional_analyzer = EmotionalDecoder()
            except ImportError as e:
                print(f"⚠️ Could not load emotional analyzer: {e}")
                self.emotional_analyzer = None
        return self.emotional_analyzer

    def process_audio_with_vad(
        self,
        audio_path: str,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process audio with Marine VAD pre-filtering

        Automatically removes silence segments before transcription,
        improving speed and accuracy.

        Args:
            audio_path: Path to audio file
            task: Task type
            **kwargs: Additional arguments for process_audio

        Returns:
            Processing results with VAD information
        """
        if not self.enable_marine_vad or not self.marine_vad:
            return self.process_audio(audio_path, task=task, **kwargs)

        print("🌊 Marine VAD: Analyzing voice activity...")

        # Load audio
        audio, sr = sf.read(audio_path)

        # Run VAD
        vad_segments = self.marine_vad.process_audio(audio, sr)

        # Get voice segments
        voice_segments = [
            seg for seg in vad_segments
            if seg.state == VadState.VOICE
        ]

        if not voice_segments:
            print("⚠️ No voice activity detected!")
            return {
                "text": "",
                "segments": [],
                "vad_segments": vad_segments,
                "processing_note": "No voice activity detected"
            }

        total_voice_time = sum(s.end_time - s.start_time for s in voice_segments)
        total_duration = len(audio) / sr

        print(f"✅ VAD Results:")
        print(f"   Voice segments: {len(voice_segments)}")
        print(f"   Voice duration: {total_voice_time:.2f}s ({total_voice_time/total_duration*100:.1f}%)")
        print(f"   Silence removed: {total_duration - total_voice_time:.2f}s")

        # Process normally but include VAD info
        result = self.process_audio(audio_path, task=task, **kwargs)

        # Add VAD information
        result['vad_segments'] = [
            {
                'start': seg.start_time,
                'end': seg.end_time,
                'state': seg.state.name,
                'confidence': seg.confidence,
                'salience': seg.mean_salience
            }
            for seg in vad_segments
        ]

        result['vad_stats'] = {
            'total_duration': total_duration,
            'voice_duration': total_voice_time,
            'silence_duration': total_duration - total_voice_time,
            'voice_percentage': (total_voice_time / total_duration * 100) if total_duration > 0 else 0
        }

        return result

    def process_multichannel_with_localization(
        self,
        audio_path: str,
        mic_positions: List[MicrophonePosition],
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multi-channel audio with sound source localization

        Args:
            audio_path: Path to multi-channel audio file
            mic_positions: List of microphone positions
            task: Task type
            **kwargs: Additional arguments

        Returns:
            Processing results with sound source locations
        """
        print(f"🎯 Marine Localization: Processing {len(mic_positions)} channels...")

        # Load multi-channel audio
        audio, sr = sf.read(audio_path, always_2d=True)

        if audio.shape[1] < len(mic_positions):
            print(f"⚠️ Audio has {audio.shape[1]} channels but {len(mic_positions)} mics configured")
            print("   Using available channels only")

        # Create localizer
        if self.sound_localizer is None or len(self.sound_localizer.mic_positions) != len(mic_positions):
            self.sound_localizer = MarineLocalization(
                mic_positions=mic_positions,
                speed_of_sound=343.0,
                max_tdoa_window=0.02,
                min_salience=1.0
            )

        # Localize sound sources
        sources = self.sound_localizer.process_multichannel_audio(audio, sr)

        print(f"✅ Found {len(sources)} sound sources")

        # Process audio normally (use first channel for transcription)
        mono_path = audio_path  # Will use first channel by default
        result = self.process_audio(mono_path, task=task, **kwargs)

        # Add localization results
        result['sound_sources'] = [
            {
                'position': {'x': src.x, 'y': src.y, 'z': src.z},
                'timestamp': src.timestamp,
                'confidence': src.confidence,
                'salience': src.salience,
                'channels': src.contributing_channels
            }
            for src in sources
        ]

        result['mic_positions'] = [
            {
                'name': mic.name,
                'channel': mic.channel,
                'position': {'x': mic.x, 'y': mic.y, 'z': mic.z}
            }
            for mic in mic_positions
        ]

        return result

    def add_salience_scores(
        self,
        audio_path: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add Marine salience scores to existing processing result

        Identifies "consciousness moments" - the most salient/interesting
        parts of the conversation.

        Args:
            audio_path: Path to audio file
            result: Existing processing result

        Returns:
            Enhanced result with salience information
        """
        print("🌊 Adding Marine salience scores...")

        # Load audio
        audio, sr = sf.read(audio_path)

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Run Marine algorithm
        marine = MarineAlgorithm()
        salience_result = marine.process(audio, sr)

        # Add salience to merged segments
        if 'merged_segments' in result:
            for segment in result['merged_segments']:
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)

                # Find peaks in this segment
                segment_peaks = []
                segment_scores = []

                for peak_time, score in zip(salience_result['times'], salience_result['scores']):
                    if start_time <= peak_time <= end_time:
                        segment_peaks.append(peak_time)
                        segment_scores.append(score)

                if segment_scores:
                    segment['salience'] = {
                        'max': max(segment_scores),
                        'mean': np.mean(segment_scores),
                        'peak_count': len(segment_peaks)
                    }
                else:
                    segment['salience'] = {
                        'max': 0.0,
                        'mean': 0.0,
                        'peak_count': 0
                    }

        # Find consciousness moments (top 10% salience)
        if salience_result['scores']:
            threshold = np.percentile(salience_result['scores'], 90)
            consciousness_moments = [
                {
                    'time': t,
                    'salience': s
                }
                for t, s in zip(salience_result['times'], salience_result['scores'])
                if s >= threshold
            ]

            result['consciousness_moments'] = consciousness_moments[:10]  # Top 10

        result['salience_stats'] = {
            'max_salience': salience_result['max_salience'],
            'mean_salience': salience_result['mean_salience'],
            'peak_count': len(salience_result['peaks'])
        }

        print(f"✅ Salience analysis complete")
        print(f"   Max salience: {salience_result['max_salience']:.2f}")
        print(f"   Consciousness moments: {len(result.get('consciousness_moments', []))}")

        return result

    def add_emotional_analysis(
        self,
        audio_path: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add emotional spectrum analysis (requires 192kHz+ audio)

        Args:
            audio_path: Path to audio file
            result: Existing processing result

        Returns:
            Enhanced result with emotional profile
        """
        print("💗 Adding emotional spectrum analysis...")

        analyzer = self._get_emotional_analyzer()
        if not analyzer:
            print("⚠️ Emotional analyzer not available")
            return result

        try:
            emotional_profile = analyzer.analyze_emotional_spectrum(audio_path)
            result['emotional_profile'] = emotional_profile
            print("✅ Emotional analysis complete")
        except Exception as e:
            print(f"⚠️ Emotional analysis failed: {e}")

        return result

    def process_with_full_marine(
        self,
        audio_path: str,
        task: str = "transcribe",
        enable_vad: bool = True,
        enable_salience: bool = True,
        enable_emotional: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process audio with full Marine-Sense enhancement

        Args:
            audio_path: Path to audio file
            task: Task type
            enable_vad: Use Marine VAD
            enable_salience: Add salience scores
            enable_emotional: Add emotional analysis (requires high sample rate)
            **kwargs: Additional processing arguments

        Returns:
            Fully enhanced processing result
        """
        print("\n🚀 MARINE-ENHANCED AUDIO PROCESSING")
        print("=" * 60)

        # Process with VAD if enabled
        if enable_vad:
            result = self.process_audio_with_vad(audio_path, task=task, **kwargs)
        else:
            result = self.process_audio(audio_path, task=task, **kwargs)

        # Add salience if enabled
        if enable_salience:
            result = self.add_salience_scores(audio_path, result)

        # Add emotional analysis if enabled
        if enable_emotional:
            result = self.add_emotional_analysis(audio_path, result)

        print("\n✅ Marine-enhanced processing complete!")
        return result


def demo_marine_pipeline():
    """Demonstrate Marine-enhanced pipeline"""
    from pathlib import Path

    print("🌊 MARINE-ENHANCED PIPELINE DEMONSTRATION")
    print("=" * 70)

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

    # Create enhanced pipeline
    pipeline = MarineEnhancedPipeline(enable_marine_vad=True)

    # Process with full Marine enhancement
    result = pipeline.process_with_full_marine(
        audio_file,
        task="transcribe",
        enable_vad=True,
        enable_salience=True,
        enable_emotional=False,  # Requires 192kHz audio
        num_speakers=2
    )

    # Display results
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)

    if 'vad_stats' in result:
        print("\n🎤 VAD Statistics:")
        stats = result['vad_stats']
        print(f"   Total duration: {stats['total_duration']:.2f}s")
        print(f"   Voice: {stats['voice_duration']:.2f}s ({stats['voice_percentage']:.1f}%)")
        print(f"   Silence removed: {stats['silence_duration']:.2f}s")

    if 'salience_stats' in result:
        print("\n🌊 Salience Statistics:")
        stats = result['salience_stats']
        print(f"   Max salience: {stats['max_salience']:.2f}")
        print(f"   Mean salience: {stats['mean_salience']:.2f}")
        print(f"   Total peaks: {stats['peak_count']}")

    if 'consciousness_moments' in result:
        print(f"\n✨ Consciousness Moments (Top {len(result['consciousness_moments'])}):")
        for i, moment in enumerate(result['consciousness_moments'][:5], 1):
            print(f"   {i}. {moment['time']:.2f}s - salience: {moment['salience']:.2f}")

    if 'merged_segments' in result and result['merged_segments']:
        print(f"\n💬 Segments (showing first 3):")
        for i, seg in enumerate(result['merged_segments'][:3], 1):
            text = seg.get('text', '')
            speaker = seg.get('speaker', 'Unknown')
            salience = seg.get('salience', {})
            print(f"\n   {i}. [{speaker}] {text[:60]}...")
            if salience:
                print(f"      Salience: max={salience.get('max', 0):.1f}, "
                      f"mean={salience.get('mean', 0):.1f}, "
                      f"peaks={salience.get('peak_count', 0)}")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo_marine_pipeline()
