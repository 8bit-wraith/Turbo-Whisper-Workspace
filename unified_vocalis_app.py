"""
🎧⚡ Unified Vocalis Audio Workspace ⚡🎧

A comprehensive audio processing platform combining:
- Ultra-fast Whisper V3 Turbo Transcription
- Advanced Speaker Diarization
- Audio Enhancement (Trisha's Lab)
- Consciousness Analysis (Marine features)
- Security Monitoring
- Real-time Processing & Interactive Features

This unified interface brings together all Vocalis capabilities in one seamless experience.
"""

import os
import gradio as gr
import numpy as np
import soundfile as sf
import librosa
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import noisereduce as nr
from scipy.signal import butter, lfilter
from dotenv import load_dotenv

# Import Vocalis components
from audio_pipeline import AudioProcessingPipeline
from model import (
    speaker_segmentation_models, embedding2models,
    get_local_segmentation_models, get_local_embedding_models
)
from utils.audio_info import get_audio_info
from utils.visualizer import plot_waveform, plot_spectrogram, plot_pitch_track, plot_chromagram
from vocalis.core.audio_pipeline import AudioProcessingPipeline as VocalisPipeline

# Import Marine components
try:
    from marine_integration_demo import MarineAlgorithm, EmotionalSpectrum, EnhancedVocalisPipeline
    MARINE_AVAILABLE = True
except ImportError:
    MARINE_AVAILABLE = False
    print("Marine consciousness features not available")

# Import LLM helper
try:
    import llm_helper
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import common data
try:
    from common_data import COMMON_NAMES
except ImportError:
    COMMON_NAMES = []

# Load environment variables
load_dotenv()

# Global pipeline instances
_AUDIO_PIPELINE = None
_VOCALIS_PIPELINE = None
_MARINE_PIPELINE = None

def get_audio_pipeline():
    """Get or create the main audio processing pipeline"""
    global _AUDIO_PIPELINE
    if _AUDIO_PIPELINE is None:
        _AUDIO_PIPELINE = AudioProcessingPipeline()
    return _AUDIO_PIPELINE

def get_vocalis_pipeline():
    """Get or create the Vocalis pipeline"""
    global _VOCALIS_PIPELINE
    if _VOCALIS_PIPELINE is None:
        _VOCALIS_PIPELINE = VocalisPipeline()
    return _VOCALIS_PIPELINE

def get_marine_pipeline():
    """Get or create the Marine consciousness pipeline"""
    global _MARINE_PIPELINE
    if _MARINE_PIPELINE is None and MARINE_AVAILABLE:
        _MARINE_PIPELINE = EnhancedVocalisPipeline()
    return _MARINE_PIPELINE

# === Audio Enhancement Functions (Trisha's Lab) ===

def normalize_audio(y):
    """Normalize audio to -1..1 range"""
    max_val = np.max(np.abs(y))
    return y / max_val if max_val > 0 else y

def highpass_filter(y, sr, cutoff):
    """High-pass filter to remove low-frequency noise"""
    b, a = butter(N=4, Wn=cutoff/(sr/2), btype='high', analog=False)
    filtered = lfilter(b, a, y)
    return filtered

def denoise_audio_advanced(audio_path, method="standard", highpass_cutoff=None, reduce_loud_segments=False):
    """
    Advanced audio denoising with multiple methods
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        processing_steps = []

        y_processed = normalize_audio(y)
        processing_steps.append("✅ Audio normalized")

        if highpass_cutoff:
            y_processed = normalize_audio(highpass_filter(y_processed, sr, highpass_cutoff))
            processing_steps.append(f"✅ High-pass filter applied ({highpass_cutoff}Hz)")

        if method == "gentle":
            y_processed = normalize_audio(nr.reduce_noise(y=y_processed, sr=sr, stationary=False, prop_decrease=0.6))
            processing_steps.append("✅ Gentle noise reduction applied")
        elif method == "aggressive":
            y_processed = normalize_audio(nr.reduce_noise(y=y_processed, sr=sr, stationary=False, prop_decrease=0.9))
            processing_steps.append("✅ Aggressive noise reduction applied")
        else:
            noise_sample = y_processed[:sr] if len(y_processed) > sr else y_processed
            y_processed = normalize_audio(nr.reduce_noise(y=y_processed, sr=sr, y_noise=noise_sample))
            processing_steps.append("✅ Standard noise reduction applied")

        if reduce_loud_segments:
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y_processed, frame_length=frame_length, hop_length=hop_length)[0]
            threshold = np.percentile(rms, 90)
            loud_frames = rms > threshold

            mask = np.ones_like(y_processed)
            for i, is_loud in enumerate(loud_frames):
                if is_loud:
                    start = i * hop_length
                    end = min(start + frame_length, len(y_processed))
                    mask[start:end] *= 0.2

            y_processed = normalize_audio(y_processed * mask)
            processing_steps.append("✅ Dynamic volume reduction applied")

        processing_info = {
            "original_duration": len(y) / sr,
            "sample_rate": sr,
            "processing_steps": processing_steps,
            "method": method,
            "highpass_cutoff": highpass_cutoff,
            "reduce_loud_segments": reduce_loud_segments
        }

        return y_processed, sr, processing_info

    except Exception as e:
        raise e

# === Consciousness Analysis Functions ===

def create_salience_visualization(audio_file, salience_data):
    """Create interactive salience visualization"""
    try:
        audio, sr = sf.read(audio_file)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        time_audio = np.linspace(0, len(audio)/sr, len(audio))
        consciousness_moments = salience_data.get('consciousness_moments', [])

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Audio Waveform with Consciousness Peaks', 'Salience Score Distribution'),
            vertical_spacing=0.15
        )

        fig.add_trace(
            go.Scatter(
                x=time_audio[::100],
                y=audio[::100],
                mode='lines',
                name='Audio',
                line
            ),
            row=1, col=1
        )

        # Add consciousness peaks
        for moment in consciousness_moments:
            fig.add_vline(
                x=moment['time'],
                line_dash="dash",
                line_color="#ff00ff",
                annotation_text=f"Peak {moment['salience']:.1f}",
                row=1, col=1
            )

        # Salience distribution
        salience_scores = salience_data.get('scores', [])
        if salience_scores:
            fig.add_trace(
                go.Histogram(
                    x=salience_scores,
                    nbinsx=50,
                    name='Salience Distribution',
                    marker_color='#00ccff'
                ),
                row=2, col=1
            )

        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#1a1a1a',
            font_color='#cccccc'
        )

        return fig
    except Exception as e:
        return None

def create_emotion_visualization(emotional_profile):
    """Create emotion analysis visualization"""
    try:
        emotions = emotional_profile.get('emotions', [])
        timestamps = emotional_profile.get('timestamps', [])

        if not emotions or not timestamps:
            return None

        fig = go.Figure()

        # Plot emotion intensities over time
        for emotion, values in emotions.items():
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines',
                    name=emotion,
                    line=dict(width=2)
                )
            )

        fig.update_layout(
            title="Emotional Profile Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Intensity",
            height=400,
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#1a1a1a',
            font_color='#cccccc'
        )

        return fig
    except Exception as e:
        return None

# === Main Processing Functions ===

def process_audio_unified(audio_path, task, segmentation_model, embedding_model,
                         num_speakers=2, threshold=0.5, use_marine=False,
                         marine_salience=False, analyze_emotions=False):
    """Unified audio processing function"""
    try:
        if use_marine and MARINE_AVAILABLE:
            pipeline = get_marine_pipeline()
            result = pipeline.process_with_consciousness(
                audio_path,
                task=task,
                num_speakers=num_speakers,
                use_marine_salience=marine_salience,
                analyze_emotions=analyze_emotions
            )
        else:
            pipeline = get_audio_pipeline()
            result = pipeline.process_audio(
                audio_path=audio_path,
                task=task,
                segmentation_model=segmentation_model,
                embedding_model=embedding_model,
                num_speakers=num_speakers,
                threshold=threshold
            )

        return result

    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}

def generate_chat_bubbles(segments, speaker_names=None, include_timestamps=True, audio_path=None):
    """Generate interactive chat bubbles HTML with audio synchronization"""
    if not segments:
        return "<div class='chat-container'>No conversation segments found</div>"

    speaker_names = speaker_names or {}

    html = "<div class='chat-container'>"

    # Add audio player controls if audio path is provided
    if audio_path:
        html += f"""
        <div class='audio-player'>
            <audio id='main-audio-player' controls preload='metadata'>
                <source src='{audio_path}' type='audio/wav'>
                Your browser does not support the audio element.
            </audio>
            <div style='display: flex; gap: 10px; margin-top: 10px;'>
                <button onclick='playAudio()' style='background: #00ff9d; color: black; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer;'>▶️ Play</button>
                <button onclick='pauseAudio()' style='background: #ff6b6b; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer;'>⏸️ Pause</button>
                <button onclick='restartAudio()' style='background: #4ecdc4; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer;'>🔄 Restart</button>
                <span id='current-time' style='color: #00ff9d; align-self: center;'>0:00</span>
            </div>
        </div>
        """

    for i, segment in enumerate(segments):
        speaker_id = segment.get('speaker', f'Speaker {i % 2}')
        speaker_name = speaker_names.get(speaker_id, speaker_id)
        text = segment.get('text', '').strip()
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)

        if not text:
            continue

        speaker_class = f"speaker-{i % 2}"
        consciousness = segment.get('consciousness_level', 'normal')

        # Add consciousness styling
        extra_class = " consciousness-high" if consciousness == 'high' else ""

        time_display = f"{int(start_time // 60):02d}:{int(start_time % 60):02d} - {int(end_time // 60):02d}:{int(end_time % 60):02d}" if include_timestamps else ""

        # Make bubbles clickable with enhanced interactivity
        html += f"""
        <div class='chat-message {speaker_class}{extra_class}'
             data-start='{start_time}' data-end='{end_time}'
             onclick='jumpToTime({start_time}); highlightMessage(this);'
             onmouseover='showTimestamp(this)'
             onmouseout='hideTimestamp(this)'
             style='cursor: pointer; transition: all 0.3s ease;'>
            <div class='speaker-name'>{speaker_name}</div>
            <div class='message-bubble'>
                <span class='message-text'>{text}</span>
                <span class='message-actions' style='display: none; float: right; font-size: 0.8em; opacity: 0.7;'>
                    ▶️ {start_time:.1f}s
                </span>
            </div>
            {f"<div class='message-time'>{time_display}</div>" if include_timestamps else ""}
        </div>
        """

    html += "</div>"

    # Add enhanced JavaScript for audio synchronization
    html += """
    <script>
    let currentAudio = null;
    let activeMessage = null;

    function initAudioSync() {
        currentAudio = document.getElementById('main-audio-player');
        if (currentAudio) {
            currentAudio.addEventListener('timeupdate', function() {
                updateCurrentTime();
                highlightActiveMessage(currentAudio.currentTime);
            });
            currentAudio.addEventListener('loadedmetadata', function() {
                console.log('Audio loaded, duration:', currentAudio.duration);
            });
        }
    }

    function playAudio() {
        if (currentAudio) {
            currentAudio.play();
        }
    }

    function pauseAudio() {
        if (currentAudio) {
            currentAudio.pause();
        }
    }

    function restartAudio() {
        if (currentAudio) {
            currentAudio.currentTime = 0;
            currentAudio.play();
        }
    }

    function jumpToTime(time) {
        if (currentAudio) {
            currentAudio.currentTime = time;
            currentAudio.play();
        }
    }

    function updateCurrentTime() {
        if (currentAudio) {
            const time = currentAudio.currentTime;
            const minutes = Math.floor(time / 60);
            const seconds = Math.floor(time % 60);
            const timeDisplay = document.getElementById('current-time');
            if (timeDisplay) {
                timeDisplay.textContent = minutes + ':' + seconds.toString().padStart(2, '0');
            }
        }
    }

    function highlightActiveMessage(currentTime) {
        // Remove previous active highlight
        if (activeMessage) {
            activeMessage.classList.remove('active-message');
        }

        // Find and highlight current active message
        const messages = document.querySelectorAll('.chat-message');
        for (const message of messages) {
            const start = parseFloat(message.dataset.start);
            const end = parseFloat(message.dataset.end);

            if (currentTime >= start && currentTime <= end) {
                message.classList.add('active-message');
                activeMessage = message;

                // Auto-scroll to active message
                message.scrollIntoView({ behavior: 'smooth', block: 'center' });
                break;
            }
        }
    }

    function highlightMessage(element) {
        // Remove previous manual highlights
        document.querySelectorAll('.chat-message').forEach(msg => {
            msg.classList.remove('manual-highlight');
        });

        // Add manual highlight
        element.classList.add('manual-highlight');
        setTimeout(() => {
            element.classList.remove('manual-highlight');
        }, 2000);
    }

    function showTimestamp(element) {
        const actions = element.querySelector('.message-actions');
        if (actions) {
            actions.style.display = 'block';
        }
    }

    function hideTimestamp(element) {
        const actions = element.querySelector('.message-actions');
        if (actions && !element.classList.contains('active-message')) {
            actions.style.display = 'none';
        }
    }

    // Initialize when DOM is ready
    document.addEventListener('DOMContentLoaded', initAudioSync);

    // Also try immediately if DOM is already loaded
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        setTimeout(initAudioSync, 500);
    }
    </script>
    """

    return html

# === UI Theme and Styling ===

# Enhanced Cyberpunk Theme with Marine elements
unified_theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="#00ff9d",
    button_primary_background_fill_hover="#00cc7a",
    button_primary_text_color="black",
    button_primary_border_color="#00ff9d",
    block_label_background_fill="#111111",
    block_label_text_color="#00ff9d",
    block_title_text_color="#00ff9d",
    input_background_fill="#222222",
    slider_color="#00ff9d",
    body_text_color="#cccccc",
    body_background_fill="#0a0a0a",
)

# Enhanced CSS with consciousness features
enhanced_css = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');

:root {
    --cyber-green: #00ff9d;
    --cyber-cyan: #00ccff;
    --cyber-pink: #ff00ff;
    --dark-bg: #0a0a0a;
    --darker-bg: #111111;
    --dark-card: #1a1a1a;
    --text-primary: #ffffff;
    --text-secondary: #cccccc;
}

.gradio-container {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 25%, #0a0a0a 50%, #001a33 75%, #0a0a0a 100%) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

#title {
    text-align: center;
    margin-bottom: 20px;
    padding: 20px;
    background: rgba(0, 255, 157, 0.1);
    border-radius: 15px;
    border: 1px solid #00ff9d;
}

#title h1 {
    background: linear-gradient(90deg, #00ff9d 0%, #00ccff 25%, #ff00ff 50%, #00ff9d 75%, #00ccff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.5em;
    letter-spacing: 2px;
    margin: 0;
    text-shadow: 0 0 30px rgba(0, 255, 157, 0.5);
}

.status-bar {
    border-left: 4px solid #00ff9d;
    padding: 15px;
    background: rgba(0, 255, 157, 0.1);
    border-radius: 8px;
    margin: 10px 0;
}

.tabbed-content {
    min-height: 500px;
}

/* Enhanced Chat Bubbles */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 20px;
    max-height: 600px;
    height: 600px;
    overflow-y: auto;
    background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
    border-radius: 12px;
    border: 1px solid #333;
    scroll-behavior: smooth;
    box-shadow: 0 0 20px rgba(0, 255, 157, 0.1);
}

.chat-message {
    display: flex;
    flex-direction: column;
    max-width: 85%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    border-radius: 12px;
    padding: 2px;
}

.chat-message:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 255, 157, 0.2);
}

.chat-message.speaker-0 {
    align-self: flex-start;
}

.chat-message.speaker-1 {
    align-self: flex-end;
}

.speaker-name {
    font-size: 0.85em;
    margin-bottom: 4px;
    color: #00ff9d;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.message-bubble {
    padding: 12px 18px;
    border-radius: 18px;
    position: relative;
    word-break: break-word;
    font-size: 0.95em;
    line-height: 1.4;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.speaker-0 .message-bubble {
    background: linear-gradient(135deg, #333 0%, #444 100%);
    border-bottom-left-radius: 6px;
    color: #fff;
}

.speaker-1 .message-bubble {
    background: linear-gradient(135deg, #00cc7a 0%, #00ff9d 100%);
    border-bottom-right-radius: 6px;
    color: #000;
    font-weight: 500;
}

.message-time {
    font-size: 0.75em;
    margin-top: 6px;
    color: #999;
    align-self: flex-end;
    opacity: 0.8;
}

/* Consciousness highlighting */
.consciousness-high {
    animation: consciousness-glow 2s ease-in-out infinite alternate;
}

@keyframes consciousness-glow {
    0% { box-shadow: 0 0 10px rgba(255, 0, 255, 0.3); }
    100% { box-shadow: 0 0 20px rgba(255, 0, 255, 0.6); }
}

.consciousness-high .speaker-name {
    color: #ff00ff;
}

.consciousness-high .message-bubble {
    border: 1px solid rgba(255, 0, 255, 0.5);
}

/* Processing indicators */
.processing-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 15px;
    background: rgba(0, 204, 255, 0.1);
    border: 1px solid #00ccff;
    border-radius: 8px;
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.spinner {
    width: 20px;
    height: 20px;
    border: 2px solid #00ccff;
    border-top: 2px solid transparent;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Enhanced buttons */
.gr-button {
    background: linear-gradient(135deg, #00ff9d 0%, #00ccff 100%) !important;
    border: 2px solid #00ff9d !important;
    color: #000 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
}

.gr-button:hover {
    background: linear-gradient(135deg, #00ccff 0%, #ff00ff 100%) !important;
    box-shadow: 0 0 20px rgba(0, 255, 157, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Tab styling */
.gr-tabs .tab-nav {
    background: rgba(26, 26, 26, 0.8) !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 5px !important;
}

.gr-tabs .tab-nav button {
    background: transparent !important;
    color: #cccccc !important;
    border: none !important;
    padding: 12px 20px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.gr-tabs .tab-nav button.selected {
    background: linear-gradient(135deg, #00ff9d 0%, #00ccff 100%) !important;
    color: #000 !important;
    font-weight: 600 !important;
}

/* Progress bars */
.progress-container {
    width: 100%;
    height: 8px;
    background: #333;
    border-radius: 4px;
    overflow: hidden;
    margin: 10px 0;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #00ff9d 0%, #00ccff 50%, #ff00ff 100%);
    border-radius: 4px;
    transition: width 0.3s ease;
    box-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
}

/* Enhanced interactive features */
.audio-player {
    background: rgba(26, 26, 26, 0.9);
    border: 1px solid #00ff9d;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 0 10px rgba(0, 255, 157, 0.2);
}

.audio-player audio {
    width: 100%;
    filter: hue-rotate(90deg);
    border-radius: 4px;
}

.message-actions {
    transition: opacity 0.3s ease;
}

.manual-highlight {
    animation: manual-highlight-pulse 2s ease-in-out;
}

@keyframes manual-highlight-pulse {
    0%, 100% { box-shadow: 0 0 10px rgba(0, 255, 157, 0.5); }
    50% { box-shadow: 0 0 20px rgba(0, 255, 157, 0.8); }
}

/* Enhanced hover effects */
.chat-message:hover .message-actions {
    opacity: 1 !important;
}

.chat-message.active-message {
    border-left: 3px solid #00ff9d;
    background: rgba(0, 255, 157, 0.1);
}

.chat-message.active-message .message-bubble {
    box-shadow: 0 0 15px rgba(0, 255, 157, 0.4);
}

/* Responsive design */
@media (max-width: 768px) {
    .chat-message {
        max-width: 95%;
    }

    .message-bubble {
        padding: 10px 14px;
        font-size: 0.9em;
    }

    #title h1 {
        font-size: 2em;
    }

    .export-buttons {
        flex-direction: column;
    }
}
"""

# JavaScript for enhanced interactivity
enhanced_js = """
function jumpToTime(time) {
    const audio = document.querySelector('audio');
    if (audio) {
        audio.currentTime = time;
        audio.play();
    }
}

function updateProgressBar(progress) {
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        progressBar.style.width = progress + '%';
    }
}

function showProcessingIndicator(message) {
    const indicator = document.querySelector('.processing-indicator span');
    if (indicator) {
        indicator.textContent = message;
    }
}

// Auto-scroll chat to bottom
function scrollChatToBottom() {
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

// Highlight active message during playback
function highlightActiveMessage(currentTime) {
    document.querySelectorAll('.chat-message').forEach(msg => {
        const start = parseFloat(msg.dataset.start);
        const end = parseFloat(msg.dataset.end);
        msg.classList.toggle('active-message', currentTime >= start && currentTime <= end);
    });
}

// Initialize audio synchronization
document.addEventListener('DOMContentLoaded', function() {
    const audio = document.querySelector('audio');
    if (audio) {
        audio.addEventListener('timeupdate', function() {
            highlightActiveMessage(audio.currentTime);
        });
    }

    // Auto-scroll on new messages
    const observer = new MutationObserver(scrollChatToBottom);
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        observer.observe(chatContainer, { childList: true });
    }
});

// Export functionality
function exportJSON() {
    const jsonElement = document.querySelector('[data-testid="json"]');
    if (!jsonElement) {
        alert('No JSON data found');
        return;
    }

    const data = jsonElement.textContent;
    downloadFile(data, 'vocalis_results.json', 'application/json');
}

function exportCSV() {
    const jsonElement = document.querySelector('[data-testid="json"]');
    if (!jsonElement) {
        alert('No data found');
        return;
    }

    try {
        const data = JSON.parse(jsonElement.textContent);
        const segments = data.segments || [];
        const speakerNames = data.speaker_names || {};

        let csv = 'Start Time,End Time,Speaker,Text\\n';
        segments.forEach(segment => {
            const speaker = speakerNames[segment.speaker] || segment.speaker || 'Unknown';
            const text = segment.text ? segment.text.replace(/"/g, '""') : '';
            csv += `"${segment.start || 0}","${segment.end || 0}","${speaker}","${text}"\\n`;
        });

        downloadFile(csv, 'vocalis_transcript.csv', 'text/csv');
    } catch (e) {
        alert('Error generating CSV: ' + e.message);
    }
}

function exportSRT() {
    const jsonElement = document.querySelector('[data-testid="json"]');
    if (!jsonElement) {
        alert('No data found');
        return;
    }

    try {
        const data = JSON.parse(jsonElement.textContent);
        const segments = data.segments || [];
        const speakerNames = data.speaker_names || {};

        let srt = '';
        segments.forEach((segment, index) => {
            if (!segment.text) return;

            const speaker = speakerNames[segment.speaker] || segment.speaker || 'Unknown';
            const start = formatSRTTime(segment.start || 0);
            const end = formatSRTTime(segment.end || 0);

            srt += `${index + 1}\\n`;
            srt += `${start} --> ${end}\\n`;
            srt += `[${speaker}]: ${segment.text}\\n\\n`;
        });

        downloadFile(srt, 'vocalis_subtitles.srt', 'text/plain');
    } catch (e) {
        alert('Error generating SRT: ' + e.message);
    }
}

function exportTXT() {
    const jsonElement = document.querySelector('[data-testid="json"]');
    if (!jsonElement) {
        alert('No data found');
        return;
    }

    try {
        const data = JSON.parse(jsonElement.textContent);
        const segments = data.segments || [];
        const speakerNames = data.speaker_names || {};

        let txt = '';
        segments.forEach(segment => {
            if (!segment.text) return;

            const speaker = speakerNames[segment.speaker] || segment.speaker || 'Unknown';
            const start = segment.start || 0;
            const end = segment.end || 0;

            txt += `[${start.toFixed(1)}s - ${end.toFixed(1)}s] ${speaker}: ${segment.text}\\n\\n`;
        });

        downloadFile(txt, 'vocalis_transcript.txt', 'text/plain');
    } catch (e) {
        alert('Error generating TXT: ' + e.message);
    }
}

function formatSRTTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const millis = Math.floor((seconds % 1) * 1000);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')},${millis.toString().padStart(3, '0')}`;
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
"""

# === Main Gradio Interface ===

with gr.Blocks(
    theme=unified_theme,
    css=enhanced_css,
    js=enhanced_js,
    title="🎧 Unified Vocalis Audio Workspace ⚡"
) as unified_app:

    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div id="title">
                <h1>🎧⚡ Unified Vocalis ⚡🎧</h1>
                <p>Advanced Audio Processing • Speaker Diarization • Consciousness Analysis • Audio Enhancement</p>
                <div style="display: flex; gap: 10px; justify-content: center; margin-top: 10px;">
                    <span style="background: rgba(0, 255, 157, 0.2); padding: 5px 10px; border-radius: 15px; font-size: 0.8em;">Whisper V3 Turbo</span>
                    <span style="background: rgba(0, 204, 255, 0.2); padding: 5px 10px; border-radius: 15px; font-size: 0.8em;">Speaker Diarization</span>
                    <span style="background: rgba(255, 0, 255, 0.2); padding: 5px 10px; border-radius: 15px; font-size: 0.8em;">Consciousness Analysis</span>
                </div>
            </div>
            """)

    with gr.Tabs(elem_classes="tabbed-content") as main_tabs:

        # === MAIN PROCESSING TAB ===
        with gr.TabItem("🎙️ Audio Processing", elem_id="main-processing"):

            with gr.Row():
                with gr.Column(scale=2, elem_classes="audio-input-column"):

                    # Audio Input Section
                    with gr.Group():
                        gr.Markdown("### 🎵 Audio Input")
                        audio_input = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            interactive=True,
                            elem_id="audio-input"
                        )

                        # Audio Playback (hidden initially)
                        audio_playback = gr.Audio(
                            label="Audio Playback",
                            type="filepath",
                            interactive=False,
                            visible=False,
                            elem_id="audio-playback"
                        )

                    # Processing Options
                    with gr.Group():
                        gr.Markdown("### ⚙️ Processing Options")

                        with gr.Row():
                            task = gr.Radio(
                                ["transcribe", "translate"],
                                label="Task",
                                value="transcribe",
                                interactive=True
                            )

                            num_speakers = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=2,
                                step=1,
                                label="Number of Speakers",
                                interactive=True
                            )

                        # Model Selection
                        with gr.Accordion("🤖 Advanced Models", open=False):
                            local_segmentation_models = get_local_segmentation_models()
                            local_embedding_models = get_local_embedding_models()

                            segmentation_model = gr.Dropdown(
                                choices=local_segmentation_models,
                                value=local_segmentation_models[0] if local_segmentation_models else list(speaker_segmentation_models.keys())[0],
                                label="Segmentation Model"
                            )

                            embedding_model_type = gr.Dropdown(
                                choices=list(local_embedding_models.keys()) if local_embedding_models else list(embedding2models.keys()),
                                value=list(local_embedding_models.keys())[0] if local_embedding_models else list(embedding2models.keys())[0],
                                label="Embedding Model Type"
                            )

                            embedding_model_choices = list(local_embedding_models.values())[0] if local_embedding_models else list(embedding2models.values())[0]
                            embedding_model = gr.Dropdown(
                                choices=embedding_model_choices,
                                value=embedding_model_choices[0] if embedding_model_choices else None,
                                label="Embedding Model"
                            )

                            threshold = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                label="Clustering Threshold"
                            )

                        # Consciousness Features (if available)
                        if MARINE_AVAILABLE:
                            with gr.Accordion("🧠 Consciousness Analysis", open=False):
                                use_marine = gr.Checkbox(
                                    value=False,
                                    label="Enable Marine Consciousness Analysis"
                                )

                                marine_salience = gr.Checkbox(
                                    value=True,
                                    label="Detect Salience Peaks",
                                    visible=False
                                )

                                analyze_emotions = gr.Checkbox(
                                    value=False,
                                    label="Analyze Emotional Content",
                                    visible=False
                                )

                                salience_threshold = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.7,
                                    step=0.1,
                                    label="Salience Threshold",
                                    visible=False
                                )

                    # Process Button
                    btn_process = gr.Button(
                        "🚀 Process Audio",
                        variant="primary",
                        size="lg",
                        elem_id="process-btn"
                    )

                # Results Section
                with gr.Column(scale=3):

                    # Processing Status
                    processing_status = gr.HTML(
                        value="<div class='processing-indicator' style='display: none;'><div class='spinner'></div><span>Initializing...</span></div>",
                        elem_id="processing-status"
                    )

                    # Chat Container
                    chat_container = gr.HTML(
                        label="💬 Conversation",
                        elem_id="chat-bubbles-container",
                        value="<div class='chat-container'>Upload an audio file and click 'Process Audio' to begin.</div>"
                    )

                    # Results Tabs
                    with gr.Tabs():
                        with gr.TabItem("📊 Summary"):
                            output_summary = gr.Markdown(
                                value="Upload an audio file and click 'Process Audio' to see the summary."
                            )

                        with gr.TabItem("📝 Raw Text"):
                            output_raw = gr.Textbox(
                                label="Raw Transcription",
                                interactive=False,
                                lines=10
                            )

                        with gr.TabItem("📋 Details"):
                            output_json = gr.JSON(label="Detailed Results")

                            # Add export buttons
                            export_buttons = gr.HTML("""
                            <div class='export-buttons'>
                                <h4>💾 Export Results</h4>
                                <button onclick='exportJSON()' class='export-btn'>📄 JSON</button>
                                <button onclick='exportCSV()' class='export-btn'>📊 CSV</button>
                                <button onclick='exportSRT()' class='export-btn'>🎬 SRT Subtitles</button>
                                <button onclick='exportTXT()' class='export-btn'>📝 Plain Text</button>
                            </div>
                            """)

                        if MARINE_AVAILABLE:
                            with gr.TabItem("🧠 Consciousness"):
                                consciousness_output = gr.Markdown(
                                    value="Consciousness analysis will appear here when enabled."
                                )

            # Status Bar
            with gr.Row():
                status = gr.Markdown(
                    value="*System ready. Upload an audio file to begin.*",
                    elem_classes="status-bar"
                )

        # === TRISHA'S AUDIO LAB TAB ===
        with gr.TabItem("🧪 Audio Enhancement Lab"):

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    # 🧪⚡ Trisha's Audio Enhancement Laboratory

                    **Welcome to the lab!** Clean up your audio with professional-grade tools:

                    - **🎛️ Noise Reduction**: Remove background noise and hum
                    - **🔊 High-pass Filtering**: Cut out low-frequency rumble
                    - **🔇 Dynamic Volume Control**: Balance inconsistent levels
                    - **📊 Before/After Analysis**: See the improvement!

                    *Perfect for improving transcription quality! 🎯*
                    """)

                    lab_audio_input = gr.Audio(
                        label="🎵 Upload Audio for Enhancement",
                        type="filepath",
                        interactive=True
                    )

                    with gr.Row():
                        denoise_method = gr.Radio(
                            choices=["gentle", "standard", "aggressive"],
                            value="standard",
                            label="🎛️ Denoising Method"
                        )

                    with gr.Row():
                        highpass_cutoff = gr.Dropdown(
                            choices=[None, 100, 150, 200],
                            value=None,
                            label="🔊 High-pass Filter (Hz)"
                        )

                        reduce_loud = gr.Checkbox(
                            value=False,
                            label="🔇 Reduce Loud Segments"
                        )

                    btn_enhance = gr.Button("✨ Enhance Audio", variant="primary", size="lg")

                    lab_status = gr.Markdown(
                        value="*Upload audio and click 'Enhance Audio' to start!*",
                        elem_classes="status-bar"
                    )

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("🎧 Enhanced Audio"):
                            enhanced_audio_output = gr.Audio(
                                label="🎉 Your Enhanced Audio",
                                type="filepath",
                                interactive=False
                            )

                            processing_info = gr.Markdown(
                                label="📊 Processing Report"
                            )

                        with gr.TabItem("🔍 Before/After Comparison"):
                            with gr.Row():
                                original_waveform = gr.Plot(label="📊 Original Audio")
                                enhanced_waveform = gr.Plot(label="✨ Enhanced Audio")

                            comparison_info = gr.Markdown(
                                value="*Upload and process audio to see before/after comparison*"
                            )

        # === AUDIO ANALYSIS TAB ===
        with gr.TabItem("📊 Audio Analysis"):

            with gr.Row():
                with gr.Column(scale=1):
                    analysis_audio_input = gr.Audio(
                        label="🎵 Audio Input",
                        type="filepath",
                        interactive=True
                    )

                    btn_analyze = gr.Button("🔍 Analyze Audio", variant="primary")

                    audio_info_display = gr.Markdown(
                        label="Audio Information",
                        elem_id="audio-info-panel"
                    )

                    analysis_status = gr.Markdown(
                        value="*Upload an audio file and click 'Analyze Audio' to begin.*",
                        elem_classes="status-bar"
                    )

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Waveform"):
                            waveform_plot = gr.Plot(label="📊 Waveform Visualization")

                        with gr.TabItem("Spectrogram"):
                            spectrogram_plot = gr.Plot(label="🔊 Spectrogram Analysis")

                        with gr.TabItem("Pitch Track"):
                            pitch_plot = gr.Plot(label="🎵 Pitch Analysis")

                        with gr.TabItem("Chromagram"):
                            chroma_plot = gr.Plot(label="🎹 Note Distribution")

        # === CONSCIOUSNESS ANALYSIS TAB (if available) ===
        if MARINE_AVAILABLE:
            with gr.TabItem("🧠 Consciousness Analysis"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        # 🧠🌊 Marine Consciousness Analysis

                        **Dive deep into audio consciousness!** This advanced analysis detects:

                        - **Salience Peaks**: Moments of heightened awareness
                        - **Emotional Content**: Emotional spectrum analysis
                        - **Consciousness Levels**: High/low consciousness states
                        - **Interactive Visualizations**: Explore patterns in your audio

                        *Powered by marine-inspired algorithms! 🐋*
                        """)

                        consciousness_audio_input = gr.Audio(
                            label="🎵 Audio for Consciousness Analysis",
                            type="filepath",
                            interactive=True
                        )

                        with gr.Group():
                            gr.Markdown("### Analysis Options")

                            detect_salience = gr.Checkbox(
                                value=True,
                                label="🧠 Detect Salience Peaks"
                            )

                            analyze_emotions = gr.Checkbox(
                                value=True,
                                label="💭 Analyze Emotional Content"
                            )

                            detect_ultrasonic = gr.Checkbox(
                                value=False,
                                label="🔊 Detect Ultrasonic Content"
                            )

                            salience_threshold = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                step=0.1,
                                label="Salience Threshold"
                            )

                        btn_consciousness = gr.Button("🌊 Analyze Consciousness", variant="primary", size="lg")

                        consciousness_status = gr.Markdown(
                            value="*Upload audio and click 'Analyze Consciousness' to begin.*",
                            elem_classes="status-bar"
                        )

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("🧠 Consciousness Transcript"):
                                consciousness_transcript = gr.HTML(
                                    value="<div style='text-align: center; padding: 40px; color: #666;'>Consciousness-enhanced transcript will appear here</div>"
                                )

                            with gr.TabItem("📈 Salience Visualization"):
                                salience_plot = gr.Plot(label="Salience Analysis")

                            with gr.TabItem("💭 Emotional Analysis"):
                                emotion_plot = gr.Plot(label="Emotional Profile")

                            with gr.TabItem("📋 Analysis Report"):
                                consciousness_report = gr.Markdown(
                                    value="Detailed consciousness analysis report will appear here."
                                )

    # === FOOTER ===
    with gr.Row():
        gr.HTML("""
        <div style="text-align: center; padding: 20px; opacity: 0.7; border-top: 1px solid #333; margin-top: 20px;">
            <p>🎧 <strong>Unified Vocalis Audio Workspace</strong> ⚡ | Powered by Whisper V3 Turbo, PyAnnote, and Marine Consciousness Algorithms</p>
            <p style="font-size: 0.8em;">Built with ❤️ for audio processing excellence</p>
        </div>
        """)

    # === EVENT HANDLERS ===

    # Update embedding model choices when type changes
    def update_embedding_models(model_type):
        local_embedding_models = get_local_embedding_models()
        if model_type in local_embedding_models:
            models = local_embedding_models[model_type]
        else:
            models = embedding2models.get(model_type, [])
        return gr.Dropdown(choices=models, value=models[0] if models else None)

    embedding_model_type.change(
        fn=update_embedding_models,
        inputs=embedding_model_type,
        outputs=embedding_model
    )

    # Toggle marine options visibility
    if MARINE_AVAILABLE:
        def toggle_marine_options(enabled):
            return [
                gr.Checkbox(visible=enabled),  # marine_salience
                gr.Checkbox(visible=enabled),  # analyze_emotions
                gr.Slider(visible=enabled)     # salience_threshold
            ]

        use_marine.change(
            fn=toggle_marine_options,
            inputs=use_marine,
            outputs=[marine_salience, analyze_emotions, salience_threshold]
        )

    # Main processing function
    def process_audio_main(audio, task, segmentation_model, embedding_model, num_speakers,
                          threshold, use_marine=False, marine_salience=False, analyze_emotions=False,
                          progress=gr.Progress()):

        if not audio:
            return (
                "<div class='chat-container'>Please upload an audio file.</div>",
                "Please upload an audio file.",
                "",
                {},
                "*Please upload an audio file.*",
                "<div class='processing-indicator' style='display: none;'></div>"
            )

        try:
            # Show processing indicator
            processing_html = "<div class='processing-indicator'><div class='spinner'></div><span>Initializing processing...</span></div>"

            progress(0.1, desc="Loading audio file...");

            # Load and validate audio
            audio_info = get_audio_info(audio)
            progress(0.2, desc="Analyzing audio properties...");

            # Process audio with progress updates
            result = process_audio_unified(
                audio, task, segmentation_model, embedding_model,
                num_speakers, threshold, use_marine, marine_salience, analyze_emotions
            )

            progress(0.8, desc="Generating visualizations...");

            if "error" in result:
                return (
                    f"<div class='chat-container'>Error: {result['error']}</div>",
                    f"Error: {result['error']}",
                    "",
                    {},
                    f"*Error: {result['error']}*",
                    "<div class='processing-indicator' style='display: none;'></div>"
                )

            progress(0.9, desc="Formatting results...");

            # Generate chat bubbles with audio controls
            segments = result.get("merged_segments", result.get("segments", []))
            speaker_names = result.get("speaker_names", {})

            chat_html = generate_chat_bubbles(segments, speaker_names, audio_path=audio)

            # Generate summary
            summary = ""
            if LLM_AVAILABLE:
                summary = result.get("summary", "")
                if summary:
                    summary = f"## 🤖 AI Summary\n\n{summary}\n\n"

                topics = result.get("topics", [])
                if topics:
                    summary += f"## 📌 Main Topics\n\n" + "\n".join([f"- {topic}" for topic in topics])

            if not summary:
                summary = "No summary available. Enable LLM features for AI-generated summaries."

            # Raw text
            raw_text = result.get("text", "")

            # Status message with performance metrics
            performance = result.get("performance", {})
            status_msg = f"""*✅ Processing completed successfully!*
- Duration: {performance.get('audio_duration', 'N/A')}
- Processing time: {performance.get('total_time', 'N/A')}
- Realtime factor: {performance.get('realtime_factor', 'N/A')}
- Speakers detected: {num_speakers}"""

            progress(1.0, desc="Complete!");

            # Hide processing indicator
            processing_html = "<div class='processing-indicator' style='display: none;'></div>"

            return (
                chat_html,
                summary,
                raw_text,
                result,
                status_msg,
                processing_html
            )

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            return (
                f"<div class='chat-container'>{error_msg}</div>",
                error_msg,
                "",
                {},
                f"*{error_msg}*",
                "<div class='processing-indicator' style='display: none;'></div>"
            )

    # Connect main processing with progress tracking
    if MARINE_AVAILABLE:
        btn_process.click(
            fn=process_audio_main,
            inputs=[
                audio_input, task, segmentation_model, embedding_model, num_speakers,
                threshold, use_marine, marine_salience, analyze_emotions
            ],
            outputs=[
                chat_container, output_summary, output_raw, output_json,
                status, processing_status
            ]
        )
    else:
        btn_process.click(
            fn=lambda a, t, s, e, n, th, progress: process_audio_main(a, t, s, e, n, th, False, False, False, progress),
            inputs=[audio_input, task, segmentation_model, embedding_model, num_speakers, threshold],
            outputs=[chat_container, output_summary, output_raw, output_json, status, processing_status]
        )

    # Audio enhancement processing
    def enhance_audio_lab(audio_path, method, highpass_cutoff, reduce_loud_segments):
        if not audio_path:
            return (
                None,
                "*Please upload an audio file first.*",
                None,
                None,
                "*Upload an audio file to start.*"
            )

        try:
            # Load original for comparison
            original_audio, original_sr = librosa.load(audio_path, sr=None, mono=True)

            # Process audio
            enhanced_audio, enhanced_sr, processing_details = denoise_audio_advanced(
                audio_path, method=method, highpass_cutoff=highpass_cutoff,
                reduce_loud_segments=reduce_loud_segments
            )

            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                enhanced_path = tmp_file.name
                sf.write(enhanced_path, enhanced_audio, enhanced_sr)

            # Processing report
            processing_report = f"""
## 🎉 Enhancement Complete!

**Original Audio:**
- Duration: {processing_details['original_duration']:.2f} seconds
- Sample Rate: {processing_details['sample_rate']} Hz

**Processing Applied:**
"""
            for step in processing_details['processing_steps']:
                processing_report += f"\n{step}"

            processing_report += f"""

**Settings Used:**
- Method: {processing_details['method'].title()}
- High-pass Filter: {processing_details['highpass_cutoff'] or 'None'} Hz
- Loud Segment Reduction: {'Yes' if processing_details['reduce_loud_segments'] else 'No'}

*Ready for transcription! 🎯*
"""

            # Generate waveforms
            original_plot = plot_waveform(original_audio, original_sr, title="📊 Original Audio")
            enhanced_plot = plot_waveform(enhanced_audio, enhanced_sr, title="✨ Enhanced Audio")

            # Comparison info
            original_rms = np.sqrt(np.mean(original_audio**2))
            enhanced_rms = np.sqrt(np.mean(enhanced_audio**2))
            noise_reduction = ((original_rms - enhanced_rms) / original_rms) * 100 if original_rms > 0 else 0

            comparison_text = f"""
## 🔍 Before vs After Analysis

- **Original RMS:** {original_rms:.4f}
- **Enhanced RMS:** {enhanced_rms:.4f}
- **Noise Reduction:** {noise_reduction:.1f}%
- **Processing Steps:** {len(processing_details['processing_steps'])}
"""

            return (
                enhanced_path,
                processing_report,
                original_plot,
                enhanced_plot,
                comparison_text
            )

        except Exception as e:
            error_msg = f"Enhancement error: {str(e)}"
            return (
                None,
                f"*{error_msg}*",
                None,
                None,
                f"*{error_msg}*"
            )

    btn_enhance.click(
        fn=enhance_audio_lab,
        inputs=[lab_audio_input, denoise_method, highpass_cutoff, reduce_loud],
        outputs=[enhanced_audio_output, processing_info, original_waveform, enhanced_waveform, comparison_info]
    )

    # Audio analysis processing
    def analyze_audio_file(audio_path):
        if not audio_path:
            return (
                "*Upload an audio file to begin analysis.*",
                None, None, None, None
            )

        try:
            # Get audio info
            audio_info = get_audio_info(audio_path)

            info_md = f"""
## Audio Information

- **Filename:** {os.path.basename(audio_path)}
- **Duration:** {audio_info['duration']:.2f} seconds
- **Sample Rate:** {audio_info['sample_rate']} Hz
- **Channels:** {audio_info['channels']}
- **Format:** {audio_info['format']}
- **Bit Depth:** {audio_info['bit_depth']} bits
"""

            # Generate visualizations
            waveform = plot_waveform(audio_path)
            spectrogram = plot_spectrogram(audio_path)
            pitch = plot_pitch_track(audio_path)
            chroma = plot_chromagram(audio_path)

            return (
                info_md,
                waveform, spectrogram, pitch, chroma
            )

        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            return (
                f"*{error_msg}*",
                None, None, None, None
            )

    btn_analyze.click(
        fn=analyze_audio_file,
        inputs=[analysis_audio_input],
        outputs=[audio_info_display, waveform_plot, spectrogram_plot, pitch_plot, chroma_plot]
    )

    # Consciousness analysis processing
    if MARINE_AVAILABLE:
        def process_consciousness(audio_file, detect_salience, analyze_emotions, detect_ultrasonic, threshold):
            if not audio_file:
                return (
                    "Please upload an audio file",
                    None,
                    None,
                    "Please upload an audio file"
                )

            try:
                pipeline = get_marine_pipeline()

                # Process with consciousness features
                result = pipeline.process_with_consciousness(
                    audio_file,
                    task="transcribe",
                    num_speakers=2,
                    use_marine_salience=detect_salience,
                    analyze_emotions=analyze_emotions,
                    detect_ultrasonic=detect_ultrasonic,
                    salience_threshold=threshold
                )

                # Format transcript
                transcript_html = "<div style='font-family: monospace; padding: 20px; background: #0a0a0a; color: #00ff88;'>"
                transcript_html += '<h3>🧠 Consciousness-Enhanced Transcript</h3>'

                for segment in result.get('merged_segments', []):
                    speaker = segment.get('speaker', 'Unknown')
                    text = segment.get('text', '')
                    start = segment.get('start', 0)
                    consciousness = segment.get('consciousness_level', 'normal')
                    salience = segment.get('max_salience', 0)

                    style = 'background: rgba(255, 0, 255, 0.2); border-left: 4px solid #ff00ff;' if consciousness == 'high' else 'background: rgba(0, 255, 136, 0.1);'
                    icon = '✨' if consciousness == 'high' else '';

                    transcript_html += f'''
                    <div style="{style} padding: 10px; margin: 5px 0;">
                        <span style="color: #00ccff;">[{start:.1f}s]</span>
                        <strong style="color: #ff00ff;">{speaker}:</strong> {icon}
                        <span style="color: #fff;">{text}</span>
                        {f'<small style="color: #888;"> (salience: {salience:.1f})</small>' if salience > 0 else ''}
                    </div>
                    '''

                transcript_html += '</div>'

                # Create visualizations
                salience_plot = create_salience_visualization(audio_file, result.get('salience', {})) if detect_salience else None
                emotion_plot = create_emotion_visualization(result.get('emotional_profile', {})) if analyze_emotions else None

                # Generate report
                report = f"""
## 🧠 Consciousness Analysis Report

**Audio Duration:** {result.get('duration', 0):.2f} seconds
**Speakers Detected:** {len(set([s.get('speaker') for s in result.get('merged_segments', [])]))}

### Key Findings:
- **Consciousness Peaks:** {len([s for s in result.get('merged_segments', []) if s.get('consciousness_level') == 'high'])}
- **High Salience Moments:** {len([s for s in result.get('merged_segments', []) if s.get('max_salience', 0) > threshold])}
- **Emotional Segments:** {len(result.get('emotional_profile', {}).get('emotions', {}))} emotion types detected

### Processing Complete! 🌊
"""

                return transcript_html, salience_plot, emotion_plot, report

            except Exception as e:
                error_msg = f"❌ Consciousness analysis error: {str(e)}"
                return error_msg, None, None, error_msg

        btn_consciousness.click(
            fn=process_consciousness,
            inputs=[consciousness_audio_input, detect_salience, analyze_emotions, detect_ultrasonic, salience_threshold],
            outputs=[consciousness_transcript, salience_plot, emotion_plot, consciousness_report]
        )

# === LAUNCH THE APP ===
if __name__ == "__main__":
    print("🎧⚡ Starting Unified Vocalis Audio Workspace...")
    print("🚀 Features loaded:")
    print("  ✅ Audio Processing & Transcription")
    print("  ✅ Speaker Diarization")
    print("  ✅ Audio Enhancement (Trisha's Lab)")
    print("  ✅ Audio Analysis & Visualization")
    if MARINE_AVAILABLE:
        print("  ✅ Marine Consciousness Analysis")
    else:
        print("  ⚠️  Marine Consciousness Analysis (not available)")

    unified_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False
    )