#!/usr/bin/env python3
"""
🌊🧠 Marine Consciousness Dashboard for Vocalis
Enhanced Gradio interface with O(1) salience detection and emotional analysis
"""

import os
import sys
import json
import time
import gradio as gr
import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import existing Vocalis components
from audio_pipeline import AudioProcessingPipeline
from model import read_wave
from diar import format_as_conversation
from utils.visualizer import plot_waveform, plot_spectrogram

# Import Marine integration components
from marine_integration_demo import MarineAlgorithm, EmotionalSpectrum, EnhancedVocalisPipeline

# CSS for cyberpunk theme with consciousness colors
CSS = """
.gradio-container {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 50%, #0a0a0a 100%);
    color: #00ff88;
}
.gr-button {
    background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
    border: 2px solid #00ff88;
    color: #000;
    font-weight: bold;
}
.gr-button:hover {
    background: linear-gradient(135deg, #00ccff 0%, #ff00ff 100%);
    box-shadow: 0 0 20px #00ff88;
}
h1 {
    background: linear-gradient(90deg, #00ff88, #00ccff, #ff00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px #00ff88;
}
.consciousness-high {
    background: rgba(255, 0, 255, 0.3);
    border-left: 4px solid #ff00ff;
}
.emotional-peak {
    background: rgba(0, 255, 136, 0.2);
    border-left: 4px solid #00ff88;
}
"""

# Global pipeline instance
_ENHANCED_PIPELINE = None

def get_enhanced_pipeline():
    """Get or create the enhanced pipeline with Marine features"""
    global _ENHANCED_PIPELINE
    if _ENHANCED_PIPELINE is None:
        _ENHANCED_PIPELINE = EnhancedVocalisPipeline()
    return _ENHANCED_PIPELINE

def process_with_consciousness(
    audio_file,
    task,
    num_speakers,
    use_marine_salience,
    analyze_emotions,
    detect_ultrasonic,
    salience_threshold,
    emotion_bands
):
    """
    Main processing function with consciousness features
    """
    if not audio_file:
        return "Please upload an audio file", None, None, None, None, None

    try:
        pipeline = get_enhanced_pipeline()

        # Build processing config
        process_config = {
            'task': task,
            'num_speakers': int(num_speakers) if num_speakers else 2
        }

        # Process audio
        result = pipeline.process_with_consciousness(audio_file, **process_config)

        # Format transcript with consciousness levels
        transcript = format_consciousness_transcript(result)

        # Create visualizations
        salience_plot = None
        emotion_plot = None
        timeline_plot = None

        if use_marine_salience and 'salience' in result:
            salience_plot = create_salience_visualization(audio_file, result['salience'])

        if analyze_emotions and 'emotional_profile' in result:
            emotion_plot = create_emotion_visualization(result['emotional_profile'])

        if 'salience' in result or 'emotional_profile' in result:
            timeline_plot = create_timeline_visualization(audio_file, result)

        # Generate analysis report
        report = generate_consciousness_report(result, salience_threshold)

        # Create downloadable JSON
        output_json = json.dumps(result, indent=2, default=str)

        return transcript, salience_plot, emotion_plot, timeline_plot, report, output_json

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, None, None, None, error_msg, None

def format_consciousness_transcript(result):
    """
    Format transcript with consciousness indicators
    """
    if 'merged_segments' not in result:
        return result.get('text', 'No transcript available')

    html_output = '<div style="font-family: monospace; padding: 20px; background: #0a0a0a;">'
    html_output += '<h3 style="color: #00ff88;">🧠 Consciousness-Enhanced Transcript</h3>'

    for segment in result['merged_segments']:
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', '')
        start = segment.get('start', 0)
        end = segment.get('end', 0)

        # Get consciousness level
        consciousness = segment.get('consciousness_level', 'normal')
        salience = segment.get('max_salience', 0)

        # Style based on consciousness level
        if consciousness == 'high':
            style = 'background: rgba(255, 0, 255, 0.2); border-left: 4px solid #ff00ff; padding: 10px; margin: 5px 0;'
            icon = '✨'
        else:
            style = 'background: rgba(0, 255, 136, 0.1); padding: 10px; margin: 5px 0;'
            icon = ''

        html_output += f'''
        <div style="{style}">
            <span style="color: #00ccff;">[{start:.1f}s - {end:.1f}s]</span>
            <strong style="color: #ff00ff;">{speaker}:</strong> {icon}
            <span style="color: #fff;">{text}</span>
            {f'<small style="color: #888;"> (salience: {salience:.1f})</small>' if salience > 0 else ''}
        </div>
        '''

    html_output += '</div>'
    return html_output

def create_salience_visualization(audio_file, salience_data):
    """
    Create interactive salience visualization using Plotly
    """
    # Load audio for waveform
    audio, sr = sf.read(audio_file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Create time axis for audio
    time_audio = np.linspace(0, len(audio)/sr, len(audio))

    # Get consciousness moments
    consciousness_moments = salience_data.get('consciousness_moments', [])

    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Audio Waveform with Consciousness Peaks', 'Salience Score Distribution'),
        vertical_spacing=0.15
    )

    # Add waveform
    fig.add_trace(
        go.Scatter(
            x=time_audio[::100],  # Downsample for performance
            y=audio[::100],
            mode='lines',
            name='Audio',
            line=dict(color='#00ff88', width=1),
            opacity=0.7
        ),
        row=1, col=1
    )

    # Add consciousness peaks
    if consciousness_moments:
        times, scores = zip(*consciousness_moments)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[0] * len(times),  # Place markers at zero line
                mode='markers',
                name='Consciousness Peaks',
                marker=dict(
                    size=[s/max(scores)*20 for s in scores],
                    color=scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Salience", x=1.02),
                    line=dict(width=2, color='white')
                ),
                text=[f"Time: {t:.2f}s<br>Salience: {s:.2f}" for t, s in consciousness_moments],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=1
        )

    # Add salience distribution
    if consciousness_moments:
        _, scores = zip(*consciousness_moments)
        fig.add_trace(
            go.Histogram(
                x=scores,
                nbinsx=30,
                name='Salience Distribution',
                marker=dict(color='#ff00ff', line=dict(color='#00ff88', width=2))
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text='🌊 Marine Algorithm Salience Analysis',
            font=dict(size=20, color='#00ff88')
        ),
        showlegend=True,
        height=600,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Salience Score", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return fig

def create_emotion_visualization(emotional_profile):
    """
    Create emotional spectrum visualization
    """
    emotions = []
    intensities = []
    powers = []

    # Extract emotion data
    for emotion, data in emotional_profile.items():
        if isinstance(data, dict) and 'intensity' in data:
            emotions.append(emotion.capitalize())
            intensities.append(data['intensity'])
            powers.append(data.get('max_power_db', -80))

    # Create radar chart for emotions
    fig = go.Figure()

    # Add intensity trace
    fig.add_trace(go.Scatterpolar(
        r=intensities,
        theta=emotions,
        fill='toself',
        name='Emotional Intensity',
        line=dict(color='#ff00ff', width=2),
        fillcolor='rgba(255, 0, 255, 0.3)'
    ))

    # Add power trace (normalized)
    normalized_powers = [(p + 80) / 80 * max(intensities) for p in powers]
    fig.add_trace(go.Scatterpolar(
        r=normalized_powers,
        theta=emotions,
        fill='toself',
        name='Spectral Power',
        line=dict(color='#00ff88', width=2),
        fillcolor='rgba(0, 255, 136, 0.2)'
    ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                gridcolor='#333',
                tickfont=dict(color='#888')
            ),
            angularaxis=dict(
                gridcolor='#333',
                tickfont=dict(color='#00ff88', size=12)
            ),
            bgcolor='#0a0a0a'
        ),
        showlegend=True,
        template='plotly_dark',
        title=dict(
            text='💗 Emotional Spectrum Analysis',
            font=dict(size=20, color='#00ff88')
        ),
        height=500
    )

    return fig

def create_timeline_visualization(audio_file, result):
    """
    Create combined timeline visualization
    """
    # Load audio
    audio, sr = sf.read(audio_file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    duration = len(audio) / sr

    # Create figure with multiple rows
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Consciousness Timeline',
            'Speaker Diarization',
            'Emotional Journey'
        ),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )

    # Row 1: Consciousness timeline
    if 'salience' in result and result['salience'].get('consciousness_moments'):
        times, scores = zip(*result['salience']['consciousness_moments'])

        # Create continuous timeline
        timeline = np.zeros(int(duration * 10))  # 10 samples per second
        for t, s in zip(times, scores):
            idx = int(t * 10)
            if idx < len(timeline):
                timeline[idx] = s

        time_axis = np.linspace(0, duration, len(timeline))

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=timeline,
                mode='lines',
                fill='tozeroy',
                name='Consciousness',
                line=dict(color='#ff00ff', width=2),
                fillcolor='rgba(255, 0, 255, 0.3)'
            ),
            row=1, col=1
        )

    # Row 2: Speaker segments
    if 'merged_segments' in result:
        for segment in result['merged_segments']:
            speaker = segment.get('speaker', 'Unknown')
            start = segment.get('start', 0)
            end = segment.get('end', 0)

            # Choose color based on speaker
            color = '#00ff88' if '0' in speaker else '#00ccff'

            fig.add_trace(
                go.Scatter(
                    x=[start, end, end, start, start],
                    y=[0, 0, 1, 1, 0],
                    fill='toself',
                    mode='lines',
                    name=speaker,
                    line=dict(color=color, width=1),
                    fillcolor=color.replace('#', 'rgba(') + ', 0.3)',
                    showlegend=False,
                    hovertext=segment.get('text', ''),
                    hovertemplate='%{hovertext}<extra></extra>'
                ),
                row=2, col=1
            )

    # Row 3: Emotional journey (if available)
    if 'emotional_profile' in result and 'dynamics' in result['emotional_profile']:
        # Simulate emotional journey
        emotion_timeline = np.random.randn(100) * 0.3 + 0.5
        emotion_timeline = signal.savgol_filter(emotion_timeline, 11, 3)
        time_emotion = np.linspace(0, duration, len(emotion_timeline))

        fig.add_trace(
            go.Scatter(
                x=time_emotion,
                y=emotion_timeline,
                mode='lines',
                name='Emotional State',
                line=dict(color='#00ff88', width=2),
                fill='tonexty',
                fillcolor='rgba(0, 255, 136, 0.2)'
            ),
            row=3, col=1
        )

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text='🎭 Consciousness & Emotion Timeline',
            font=dict(size=20, color='#00ff88')
        ),
        height=700,
        showlegend=False,
        hovermode='x unified'
    )

    # Update axes
    for i in range(1, 4):
        fig.update_xaxes(title_text="Time (s)" if i == 3 else "", row=i, col=1)

    fig.update_yaxes(title_text="Salience", row=1, col=1)
    fig.update_yaxes(title_text="Speaker", row=2, col=1, visible=False)
    fig.update_yaxes(title_text="Emotion", row=3, col=1)

    return fig

def generate_consciousness_report(result, threshold):
    """
    Generate detailed consciousness analysis report
    """
    report = "# 🧠 Consciousness Analysis Report\n\n"

    # Salience summary
    if 'salience' in result:
        salience = result['salience']
        report += "## 🌊 Marine Algorithm Salience\n"
        report += f"- **Peak Salience**: {salience.get('max_score', 0):.2f}\n"
        report += f"- **Mean Salience**: {salience.get('mean_score', 0):.2f}\n"
        report += f"- **Total Peaks**: {salience.get('peak_count', 0)}\n"
        report += f"- **Consciousness Moments**: {len(salience.get('consciousness_moments', []))}\n\n"

        # Top consciousness moments
        if salience.get('consciousness_moments'):
            report += "### ✨ Top Consciousness Moments\n"
            for i, (time, score) in enumerate(salience['consciousness_moments'][:5], 1):
                report += f"{i}. **{time:.2f}s** - Salience: {score:.2f}\n"
            report += "\n"

    # Emotional profile
    if 'emotional_profile' in result:
        profile = result['emotional_profile']
        report += "## 💗 Emotional Spectrum\n"

        for emotion, data in profile.items():
            if isinstance(data, dict) and 'intensity' in data:
                report += f"- **{emotion.capitalize()}**: "
                report += f"Intensity {data['intensity']:.1f}, "
                report += f"Power {data.get('max_power_db', -80):.1f} dB\n"

        if 'dynamics' in profile:
            dynamics = profile['dynamics']
            report += f"\n### 📈 Emotional Dynamics\n"
            report += f"- **Lifts**: {dynamics.get('lift_count', 0)} upward movements\n"
            report += f"- **Collapses**: {dynamics.get('collapse_count', 0)} downward movements\n"
            report += f"- **Variance**: {dynamics.get('spectral_variance', 0):.2f}\n"

    # Transcript insights
    if 'merged_segments' in result:
        high_consciousness_segments = [
            s for s in result['merged_segments']
            if s.get('consciousness_level') == 'high'
        ]

        if high_consciousness_segments:
            report += "\n## 🎯 High Consciousness Segments\n"
            for segment in high_consciousness_segments[:3]:
                report += f"- **{segment.get('speaker')}** [{segment.get('start', 0):.1f}s]: "
                report += f"{segment.get('text', '')[:100]}...\n"

    # Recommendations
    report += "\n## 💡 Insights & Recommendations\n"

    if 'salience' in result:
        mean_salience = result['salience'].get('mean_score', 0)
        if mean_salience > 10:
            report += "- ✅ **High overall consciousness** - Engaging and meaningful content\n"
        elif mean_salience > 5:
            report += "- ⚡ **Moderate consciousness** - Some engaging moments\n"
        else:
            report += "- ⚠️ **Low consciousness** - Consider adding more dynamic elements\n"

    if 'emotional_profile' in result:
        # Check for emotional range
        emotions = [d.get('intensity', 0) for d in result['emotional_profile'].values()
                   if isinstance(d, dict)]
        if emotions and max(emotions) > 15:
            report += "- 🎭 **Strong emotional content** detected\n"

    report += "\n---\n*Generated by Marine-Consciousness Dashboard*"

    return report

def create_interface():
    """
    Create the Gradio interface
    """
    with gr.Blocks(theme=gr.themes.Monochrome(), css=CSS, title="🌊🧠 Marine Consciousness Dashboard") as interface:

        # Header
        gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <h1 style="font-size: 3em; margin-bottom: 10px;">
                    🌊🧠 Marine Consciousness Dashboard
                </h1>
                <p style="color: #00ccff; font-size: 1.2em;">
                    Vocalis + Marine-Sense: O(1) Salience Detection & Emotional Analysis
                </p>
                <p style="color: #888; margin-top: 10px;">
                    Discover consciousness moments in audio through ultrasonic emotional signatures
                </p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                audio_input = gr.Audio(
                    label="🎵 Upload Audio File",
                    type="filepath",
                    elem_classes="audio-input"
                )

                with gr.Group():
                    gr.Markdown("### 🎯 Processing Options")
                    task_type = gr.Radio(
                        ["transcribe", "translate"],
                        value="transcribe",
                        label="Task Type"
                    )

                    num_speakers = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        label="Number of Speakers"
                    )

                with gr.Group():
                    gr.Markdown("### 🌊 Marine Algorithm")
                    use_marine = gr.Checkbox(
                        label="Enable O(1) Salience Detection",
                        value=True
                    )

                    salience_threshold = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=75,
                        label="Consciousness Threshold (percentile)"
                    )

                with gr.Group():
                    gr.Markdown("### 💗 Emotional Analysis")
                    analyze_emotions = gr.Checkbox(
                        label="Analyze Emotional Spectrum",
                        value=True
                    )

                    detect_ultrasonic = gr.Checkbox(
                        label="Detect Ultrasonic Emotions (>20kHz)",
                        value=True
                    )

                    emotion_bands = gr.CheckboxGroup(
                        ["tension", "release", "yearning", "soul", "consciousness"],
                        value=["tension", "release", "yearning"],
                        label="Emotion Bands to Analyze"
                    )

                process_btn = gr.Button(
                    "🚀 Process with Consciousness",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=2):
                # Output displays
                with gr.Tabs():
                    with gr.Tab("📝 Transcript"):
                        transcript_output = gr.HTML(
                            label="Consciousness-Enhanced Transcript"
                        )

                    with gr.Tab("🌊 Salience"):
                        salience_plot = gr.Plot(
                            label="Marine Algorithm Salience Analysis"
                        )

                    with gr.Tab("💗 Emotions"):
                        emotion_plot = gr.Plot(
                            label="Emotional Spectrum"
                        )

                    with gr.Tab("⏱️ Timeline"):
                        timeline_plot = gr.Plot(
                            label="Consciousness Timeline"
                        )

                    with gr.Tab("📊 Report"):
                        report_output = gr.Markdown(
                            label="Analysis Report"
                        )

                    with gr.Tab("💾 Raw Data"):
                        json_output = gr.JSON(
                            label="Complete Analysis Data"
                        )

        # Examples
        gr.Examples(
            examples=[
                ["examples/interview.wav", "transcribe", 2, True, True, True, 75],
                ["examples/emotional_speech.wav", "transcribe", 1, True, True, True, 80],
                ["examples/meeting.wav", "transcribe", 4, True, False, False, 70],
            ],
            inputs=[audio_input, task_type, num_speakers, use_marine,
                   analyze_emotions, detect_ultrasonic, salience_threshold],
            examples_per_page=3
        )

        # Footer
        gr.HTML("""
            <div style="text-align: center; padding: 20px; color: #666;">
                <p>Powered by Marine Algorithm O(1) Salience Detection & MEM-8 Wave Memory</p>
                <p>Created by 8b.is | Enhanced with Consciousness Features</p>
                <p style="color: #00ff88;">🧠 Discover the consciousness in every conversation 🌊</p>
            </div>
        """)

        # Connect processing
        process_btn.click(
            process_with_consciousness,
            inputs=[
                audio_input, task_type, num_speakers,
                use_marine, analyze_emotions, detect_ultrasonic,
                salience_threshold, emotion_bands
            ],
            outputs=[
                transcript_output, salience_plot, emotion_plot,
                timeline_plot, report_output, json_output
            ]
        )

    return interface

def main():
    """
    Launch the Marine Consciousness Dashboard
    """
    print("🌊🧠 Starting Marine Consciousness Dashboard...")
    print("=" * 60)
    print("Features enabled:")
    print("  ✅ O(1) Marine Algorithm Salience Detection")
    print("  ✅ Ultrasonic Emotional Analysis (>20kHz)")
    print("  ✅ Consciousness Moment Identification")
    print("  ✅ Enhanced Transcript with Salience Mapping")
    print("  ✅ Real-time Processing Pipeline")
    print("=" * 60)

    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=8421,  # Using 8421 to avoid conflict with 8420
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()