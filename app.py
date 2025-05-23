"""
# 🎧 CyberVox Audio Workspace ⚡

A powerful audio processing workspace featuring:
- Ultra-fast Whisper V3 Turbo Transcription
- Advanced Speaker Diarization
- Audio Analysis Tools
- Cyberpunk-themed UI
"""

CREDITS = """
## Credits

This project builds upon several amazing technologies:
- [8b-is Team](https://8b.is/?ref=Turbo-Whisper-Workspace)
- [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Claude](https://claude.ai/)
"""

import os
import sys
import json
import time
import re
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from pydub import AudioSegment
import tempfile
import uuid
import librosa
import soundfile as sf
import numpy as np


# Import LLM helper module
try:
    print("Attempting to import llm_helper module...")
    import llm_helper
    print("Successfully imported llm_helper module")
    LLM_AVAILABLE = True
except ImportError:
    print("LLM helper module not available")
    LLM_AVAILABLE = False

# Import common data structures
from common_data import COMMON_NAMES
# Import from local modules
from model import (
    read_wave,
    speaker_segmentation_models, embedding2models,
    get_local_segmentation_models, get_local_embedding_models
)
from utils.audio_processor import process_audio_file, extract_audio_features
from utils.audio_info import get_audio_info
from utils.visualizer import plot_waveform, plot_spectrogram, plot_pitch_track, plot_chromagram, plot_speaker_diarization
from diar import SpeakerDiarizer, format_as_conversation

# Import audio denoising functions - Trisha's Lab specials! 🧪⚡
import noisereduce as nr
from scipy.signal import butter, lfilter

# Load environment variables
load_dotenv()

# Import the AudioProcessingPipeline
from audio_pipeline import AudioProcessingPipeline

# Single global pipeline for the entire application
_GLOBAL_PIPELINE = None

# Function to get the global pipeline
def get_global_pipeline():
    """Get or create the global AudioProcessingPipeline"""
    global _GLOBAL_PIPELINE
    
    # Create the pipeline if it doesn't exist
    if _GLOBAL_PIPELINE is None:
        print("Creating new global AudioProcessingPipeline")
        _GLOBAL_PIPELINE = AudioProcessingPipeline()
        
    return _GLOBAL_PIPELINE

# === Trisha's Audio Lab Functions ===
# 🧪 These are the secret sauce for audio enhancement! 

def normalize_audio(y):
    """Normalize audio to -1..1 range - Trisha's favorite normalization trick!"""
    max_val = np.max(np.abs(y))
    return y / max_val if max_val > 0 else y

def highpass_filter(y, sr, cutoff):
    """High-pass filter to remove low-frequency noise - cuts through the mud!"""
    b, a = butter(N=4, Wn=cutoff/(sr/2), btype='high', analog=False)
    filtered = lfilter(b, a, y)
    return filtered

def denoise_audio_advanced(audio_path, method="standard", highpass_cutoff=None, reduce_loud_segments=False):
    """
    🎧 Trisha's Advanced Audio Denoising Laboratory! 
    
    Options:
    - method: "standard", "aggressive", or "gentle" 
    - highpass_cutoff: None or frequency in Hz (100, 150, 200 recommended)
    - reduce_loud_segments: True to apply dynamic volume reduction to loud parts
    
    Returns: (processed_audio, sample_rate, processing_info)
    """
    print(f"🧪 Trisha's Lab: Processing {audio_path} with method='{method}'")
    
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        processing_steps = []
        
        # Step 1: Normalize the original audio
        y_processed = normalize_audio(y)
        processing_steps.append("✅ Audio normalized")
        
        # Step 2: Apply high-pass filter if requested
        if highpass_cutoff:
            y_processed = normalize_audio(highpass_filter(y_processed, sr, highpass_cutoff))
            processing_steps.append(f"✅ High-pass filter applied ({highpass_cutoff}Hz)")
        
        # Step 3: Apply noise reduction based on method
        if method == "gentle":
            # Gentle denoising - less aggressive
            y_processed = normalize_audio(nr.reduce_noise(y=y_processed, sr=sr, stationary=False, prop_decrease=0.6))
            processing_steps.append("✅ Gentle noise reduction applied")
        elif method == "aggressive":
            # Aggressive denoising - more thorough
            y_processed = normalize_audio(nr.reduce_noise(y=y_processed, sr=sr, stationary=False, prop_decrease=0.9))
            processing_steps.append("✅ Aggressive noise reduction applied")
        else:  # standard
            # Standard denoising using first second as noise sample
            noise_sample = y_processed[:sr] if len(y_processed) > sr else y_processed
            y_processed = normalize_audio(nr.reduce_noise(y=y_processed, sr=sr, y_noise=noise_sample))
            processing_steps.append("✅ Standard noise reduction applied")
        
        # Step 4: Apply dynamic volume reduction to loud segments if requested
        if reduce_loud_segments:
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y_processed, frame_length=frame_length, hop_length=hop_length)[0]
            threshold = np.percentile(rms, 90)  # Find the loudest 10% of frames
            loud_frames = rms > threshold
            
            # Create a mask to reduce volume of loud segments
            mask = np.ones_like(y_processed)
            for i, is_loud in enumerate(loud_frames):
                if is_loud:
                    start = i * hop_length
                    end = min(start + frame_length, len(y_processed))
                    mask[start:end] *= 0.2  # Reduce volume by 80% - "Shhh!" says Trisha
            
            y_processed = normalize_audio(y_processed * mask)
            processing_steps.append("✅ Dynamic volume reduction applied to loud segments")
        
        processing_info = {
            "original_duration": len(y) / sr,
            "sample_rate": sr,
            "processing_steps": processing_steps,
            "method": method,
            "highpass_cutoff": highpass_cutoff,
            "reduce_loud_segments": reduce_loud_segments
        }
        
        print(f"🎉 Trisha's Lab: Processing complete! Steps: {len(processing_steps)}")
        return y_processed, sr, processing_info
        
    except Exception as e:
        print(f"💥 Trisha's Lab Error: {str(e)}")
        raise e

# Combined transcription and diarization
def process_audio_with_diarization(audio_path, task, segmentation_model, embedding_model,
                                   num_speakers=2, threshold=0.5, return_timestamps=True):
    """Process audio with both transcription and speaker diarization using the global pipeline"""
    try:
        # Get the global pipeline
        pipeline = get_global_pipeline()
        print("Using global pipeline for processing")
        
        # Process audio using the pipeline
        pipeline_result = pipeline.process_audio(
            audio_path=audio_path,
            task=task,
            segmentation_model=segmentation_model,
            embedding_model=embedding_model,
            num_speakers=num_speakers,
            threshold=threshold
        )
        
        # Check for errors in pipeline result
        if "error" in pipeline_result:
            return pipeline_result
            
        # Extract data from pipeline result
        segments = pipeline_result.get("segments", [])
        merged_segments = pipeline_result.get("merged_segments", [])
        duration = pipeline_result.get("duration", 0)
        processing_times = pipeline_result.get("processing_times", {})
        
        # Convert to speaker segments format for plotting
        speaker_segments = []
        for segment in segments:
            speaker_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "speaker": segment["speaker"]
            })
        
        # Format as conversation
        conversation = format_as_conversation(merged_segments)
        
        # Create speaker diarization plot
        diarization_plot = plot_speaker_diarization(speaker_segments, duration)
        
        # Prepare enhanced output with additional information
        output_json = {
            "text": pipeline_result.get("text", ""),
            "segments": merged_segments,
            "conversation": conversation,
            "diarization_plot": diarization_plot,
            "performance": {
                "transcription_time": f"{processing_times.get('transcription', 0):.2f}s",
                "diarization_time": f"{processing_times.get('diarization', 0):.2f}s",
                "total_time": f"{processing_times.get('total', 0):.2f}s",
                "audio_duration": f"{duration:.2f}s",
                "realtime_factor": f"{processing_times.get('total', 0)/duration:.2f}x" if duration > 0 else "N/A"
            }
        }
        
        # Add LLM-generated content if available
        if "speaker_names" in pipeline_result:
            output_json["speaker_names"] = pipeline_result["speaker_names"]
            
        if "summary" in pipeline_result:
            output_json["summary"] = pipeline_result["summary"]
            
        if "topics" in pipeline_result:
            output_json["topics"] = pipeline_result["topics"]
        
        # Return the enhanced result as JSON
        return output_json
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Processing error: {str(e)}"}

# UI Theme Configuration - Cyberpunk Green
cyberpunk_theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Quicksand"), "ui-sans-serif", "system-ui"],
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
    body_background_fill="#111111",
)

# Build Gradio UI
with gr.Blocks(theme=cyberpunk_theme, css="""
    #title {text-align: center; margin-bottom: 10px;}
    #title h1 {
        background: linear-gradient(90deg, #00ff9d 0%, #00ccff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .status-bar {
        border-left: 4px solid #00ff9d;
        padding-left: 10px;
        background-color: rgba(0, 255, 157, 0.1);
    }
    .footer {text-align: center; opacity: 0.7; margin-top: 20px;}
    .tabbed-content {min-height: 400px;}
    
    /* Chat Bubbles CSS */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 15px;
        max-height: 500px;
        height: 500px;
        overflow-y: auto;
        background-color: #1a1a1a;
        border-radius: 8px;
        border: 1px solid #333;
        scroll-behavior: smooth;
    }
    .chat-message {
        display: flex;
        flex-direction: column;
        max-width: 85%;
        transition: all 0.3s ease;
    }
    .chat-message.speaker-0 {
        align-self: flex-start;
    }
    .chat-message.speaker-1 {
        align-self: flex-end;
    }
    .speaker-name {
        font-size: 0.8em;
        margin-bottom: 2px;
        color: #00ff9d;
        font-weight: bold;
    }
    .message-bubble {
        padding: 10px 15px;
        border-radius: 18px;
        position: relative;
        word-break: break-word;
    }
    .speaker-0 .message-bubble {
        background-color: #333;
        border-bottom-left-radius: 4px;
        color: #fff;
    }
    .speaker-1 .message-bubble {
        background-color: #00cc7a;
        border-bottom-right-radius: 4px;
        color: #000;
    }
    .message-time {
        font-size: 0.7em;
        margin-top: 2px;
        color: #999;
        align-self: flex-end;
    }
    .active-message {
        transform: scale(1.02);
    }
    .speaker-0.active-message .message-bubble {
        background-color: #444;
        box-shadow: 0 0 10px rgba(0, 255, 157, 0.3);
    }
    .speaker-1.active-message .message-bubble {
        background-color: #00ff9d;
        box-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
    }
    .chat-controls {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 15px;
    }
    
    /* Conversation Summary and Topics */
    .conversation-summary, .conversation-topics {
        background-color: #222;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        border-left: 4px solid #00ff9d;
    }
    .conversation-summary h3, .conversation-topics h4 {
        color: #00ff9d;
        margin-top: 0;
        margin-bottom: 10px;
    }
    .conversation-summary p {
        color: #ddd;
        line-height: 1.5;
    }
    .conversation-topics ul {
        margin: 0;
        padding-left: 20px;
    }
    .conversation-topics li {
        color: #ddd;
        margin-bottom: 5px;
    }
""") as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div id="title">
                <h1>🎧 CyberVox Audio Workspace ⚡</h1>
                <p>Advanced audio processing with Whisper V3 Turbo and Speaker Diarization</p>
            </div>
            """)
            
    with gr.Tabs(elem_classes="tabbed-content") as tabs:
        # Chat Bubbles Tab (now primary and merged with transcription)
        with gr.TabItem("💬 CyberVox Chat"):
            with gr.Row():
                with gr.Column(scale=2, elem_classes="audio-input-column"):
                    audio_input = gr.Audio(
                        label="Audio Input",
                        type="filepath",
                        interactive=True,
                        elem_id="audio-input"
                    )
                    
                    # Add audio playback component
                    audio_playback = gr.Audio(
                        label="Audio Playback",
                        type="filepath",
                        interactive=False,
                        visible=False,
                        elem_id="audio-playback"
                    )
                    
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
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        # Get locally available models
                        local_segmentation_models = get_local_segmentation_models()
                        local_embedding_models = get_local_embedding_models()
                        
                        # Use locally available segmentation models
                        segmentation_model = gr.Dropdown(
                            choices=local_segmentation_models,
                            value=local_segmentation_models[0] if local_segmentation_models else speaker_segmentation_models[0],
                            label="Segmentation Model (Local)"
                        )
                        
                        # Use locally available embedding model types
                        local_embedding_types = list(local_embedding_models.keys())
                        embedding_model_type = gr.Dropdown(
                            choices=local_embedding_types,
                            value=local_embedding_types[0] if local_embedding_types else list(embedding2models.keys())[0],
                            label="Embedding Model Type (Local)"
                        )
                        
                        # Use locally available embedding models for the selected type
                        first_type = local_embedding_types[0] if local_embedding_types else list(embedding2models.keys())[0]
                        first_models = local_embedding_models.get(first_type, embedding2models[first_type])
                        embedding_model = gr.Dropdown(
                            choices=first_models,
                            value=first_models[0] if first_models else embedding2models[first_type][0],
                            label="Embedding Model (Local)"
                        )
                        
                        threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Clustering Threshold"
                        )
                        

                    btn_process = gr.Button("Generate Chat", variant="primary")
                
                with gr.Column(scale=3):
                    # Chat container for displaying chat bubbles
                    chat_container = gr.HTML(
                        label="Chat Bubbles",
                        elem_id="chat-bubbles-container",
                        value="<div class='chat-container'>Upload an audio file and click 'Generate Chat' to start.</div>"
                    )
                    
                    with gr.Tabs():
                        with gr.TabItem("Summary"):
                            output_conversation = gr.Markdown(
                                label="Conversation Summary",
                                elem_id="conversation-output",
                                value="Upload an audio file and click 'Generate Chat' to start. A summary will appear here."
                            )
                        
                        with gr.TabItem("Raw Text"):
                            output_raw = gr.Textbox(
                                label="Raw Transcription",
                                interactive=False,
                                elem_id="transcription-output",
                                lines=15
                            )
                        
                        with gr.TabItem("JSON Data"):
                            output_json = gr.JSON(
                                label="Detailed Results"
                            )
            
            with gr.Row():
                status = gr.Markdown(
                    value="*System ready. Upload an audio file to begin.*", 
                    elem_classes="status-bar"
                )
        
        # === Trisha's Audio Lab Tab === 🧪
        with gr.TabItem("🧪 Trisha's Audio Lab"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    # 🧪⚡ Trisha's Audio Enhancement Laboratory
                    
                    **Welcome to the lab, Hue!** Here's where the magic happens! 
                    
                    Upload your audio and let's clean it up with some serious science:
                    - **Noise Reduction**: Get rid of that pesky background noise
                    - **High-pass Filtering**: Cut out low-frequency rumble 
                    - **Dynamic Volume Control**: Tame those loud segments
                    
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
                            label="🎛️ Denoising Method",
                            info="Gentle = subtle, Standard = balanced, Aggressive = maximum cleanup"
                        )
                    
                    with gr.Row():
                        highpass_cutoff = gr.Dropdown(
                            choices=[None, 100, 150, 200],
                            value=None,
                            label="🔊 High-pass Filter (Hz)",
                            info="Remove frequencies below this threshold (good for speech: 100-150Hz)"
                        )
                        
                        reduce_loud = gr.Checkbox(
                            value=False,
                            label="🔇 Reduce Loud Segments",
                            info="Automatically lower volume of loud parts"
                        )
                    
                    btn_enhance = gr.Button("✨ Enhance Audio", variant="primary", size="lg")
                    
                    lab_status = gr.Markdown(
                        value="*Upload audio and click 'Enhance Audio' to start the magic! ⚡*", 
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
                                label="📊 Processing Report",
                                value="*Processing info will appear here*"
                            )
                        
                        with gr.TabItem("🔍 Before/After Comparison"):
                            with gr.Row():
                                original_waveform = gr.Plot(label="📊 Original Waveform")
                                enhanced_waveform = gr.Plot(label="✨ Enhanced Waveform")
                            
                            comparison_info = gr.Markdown(
                                value="*Upload and process audio to see before/after comparison*"
                            )

        # Audio Analysis Tab
        with gr.TabItem("📊 Audio Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_analysis_input = gr.Audio(
                        label="🎵 Audio Input",
                        type="filepath",
                        interactive=True
                    )
                    
                    btn_analyze = gr.Button("🔍 Analyze Audio", variant="primary")
                    
                    # Info panel
                    audio_info = gr.Markdown(
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

    
    # Connect embedding model type to embedding model choices
    def update_embedding_models(model_type):
        # Get locally available models
        local_embedding_models = get_local_embedding_models()
        
        # Use locally available models if available, otherwise fall back to all models
        if model_type in local_embedding_models:
            models = local_embedding_models[model_type]
        else:
            models = embedding2models[model_type]
            
        return gr.Dropdown(choices=models, value=models[0] if models else embedding2models[model_type][0])
    
    embedding_model_type.change(
        fn=update_embedding_models,
        inputs=embedding_model_type,
        outputs=embedding_model
    )
    


    # Helper function to identify speaker names in text
    def identify_speaker_names(segments):
        # Try to use LLM for name identification if available
        if LLM_AVAILABLE:
            try:
                # First try the LLM-based approach
                llm_names = llm_helper.identify_speaker_names_llm(segments)
                if llm_names and len(llm_names) > 0:
                    print(f"LLM identified names: {llm_names}")
                    return llm_names
            except Exception as e:
                print(f"Error using LLM for name identification: {e}")
        
        # Fallback to rule-based approach
        print("Using fallback method for name identification")
        if LLM_AVAILABLE:
            return llm_helper.identify_speaker_names_fallback(segments)
            
        # Built-in fallback if LLM helper is not available
        detected_names = {}
        name_mentions = {}
            
        # First pass: find potential speaker names in the text
        for segment in segments:
            speaker_id = segment['speaker']
            text = segment['text']
                
            # Check for specific names we want to prioritize (Alexandra, Veronica)
            for special_name in ["Alexandra", "Veronica"]:
                if re.search(f'\\b{special_name}\\b', text, re.IGNORECASE):
                    print(f"Found {special_name} mentioned in text: {text}")
                    # If this speaker is addressing the person, they're likely not that person
                    # So we'll mark this speaker as NOT being that person
                    if speaker_id in detected_names and detected_names[speaker_id].lower() == special_name.lower():
                        print(f"Speaker {speaker_id} is addressing {special_name}, so they can't be {special_name}")
                        detected_names.pop(speaker_id)
                    
                    # Add to name mentions for later assignment to OTHER speakers
                    if special_name not in name_mentions:
                        name_mentions[special_name] = 0
                    name_mentions[special_name] += 3
            
            # Extract names from text
            # Look for common name patterns directly
            for name in COMMON_NAMES:
                # Look for the name as a whole word with word boundaries
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    # Count name mentions
                    if name not in name_mentions:
                        name_mentions[name] = 0
                    name_mentions[name] += 1
                
            # Look for "I'm [Name]" or "My name is [Name]" patterns
            name_intro_patterns = [
                r"I'?m\s+(\w+)",
                r"[Mm]y name is\s+(\w+)",
                r"[Cc]all me\s+(\w+)",
                r"[Tt]his is\s+(\w+)"  
            ]
            
            for pattern in name_intro_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if match in COMMON_NAMES:
                        if speaker_id not in detected_names:
                            detected_names[speaker_id] = match
                        elif detected_names[speaker_id] != match:
                            # If multiple names are detected for the same speaker, keep the most frequent one
                            if match not in name_mentions:
                                name_mentions[match] = 0
                            name_mentions[match] += 3  # Give higher weight to explicit introduction
        
        # Second pass: assign names to speakers based on frequency and context
        for speaker_id in set([segment['speaker'] for segment in segments]):
            if speaker_id not in detected_names:
                # Special handling for Alexandra and Veronica
                for special_name in ["Alexandra", "Veronica"]:
                    if special_name in name_mentions and special_name not in detected_names.values():
                        # Find speakers who might be addressing this person
                        addressing_speakers = []
                        for seg in segments:
                            if seg['speaker'] != speaker_id and special_name.lower() in seg.get('text', '').lower():
                                addressing_speakers.append(seg['speaker'])
                        
                        # If this speaker is not addressing the special name, they might be that person
                        if speaker_id not in addressing_speakers:
                            detected_names[speaker_id] = special_name
                            print(f"Assigned {special_name} to {speaker_id} based on being addressed")
                            break
                
                # If no special name was assigned, use frequency-based approach
                if speaker_id not in detected_names:
                    # Find the most mentioned name that hasn't been assigned yet
                    available_names = [name for name, count in sorted(name_mentions.items(), key=lambda x: x[1], reverse=True) 
                                    if name not in detected_names.values()]                
                    if available_names:
                        detected_names[speaker_id] = available_names[0]
                        print(f"Assigned {available_names[0]} to {speaker_id} based on frequency")
        
        return detected_names

    # Function to generate chat bubbles from segments
    def process_chat(audio, task, segmentation_model, embedding_model, num_speakers, threshold):

        # Initialize default values
        chat_html = "<div class='chat-container'>Processing audio...</div>"
        audio_path = None
        summary_markdown = ""
        status_msg = "*Processing audio...*"
        
        try:
            # Check if audio is provided
            if audio is None:
                return "<div class='chat-container'>Please upload an audio file.</div>", None, "", "*Please upload an audio file.*"
                
            # Set audio path
            audio_path = audio
            # Process audio file using global pipeline
            # This handles transcription and diarization
            result = process_audio_with_diarization(
                audio_path,
                task,
                segmentation_model,
                embedding_model,
                num_speakers,
                threshold,
                return_timestamps=True
            )
            
            # Generate chat HTML with debug info
            print(f"Result keys: {result.keys()}")
            
            if isinstance(result, dict) and "error" in result:
                return f"<div class='chat-container'>Error: {result['error']}</div>", None, "", f"*Error: {result['error']}*"
                
            if "segments" in result:
                segments = result["segments"]
                print(f"Found {len(segments)} segments")
                
                # Use speaker_names from process_audio_with_diarization if available, otherwise use empty dict
                speaker_names = result.get("speaker_names", {})
                # print(f"Using speaker names from diarization: {speaker_names}")
                
                chat_html = "<div class='chat-container'>"
                
                # Generate chat bubbles for each segment
                for i, segment in enumerate(segments):
                    # Get the speaker ID
                    raw_speaker_id = segment['speaker'] if 'speaker' in segment else f"Speaker {i % 2}"
                    
                    try:
                        if raw_speaker_id.startswith("Speaker"):
                            speaker_id = int(raw_speaker_id.split()[-1])
                            # Use identified name if available, otherwise use Speaker X
                            if raw_speaker_id in speaker_names and speaker_names[raw_speaker_id]:
                                speaker_name = speaker_names[raw_speaker_id]
                            else:
                                speaker_name = raw_speaker_id
                        else:
                            speaker_id = i % 2
                            speaker_name = raw_speaker_id
                    except (ValueError, IndexError):
                        speaker_id = i % 2
                        speaker_name = f"Speaker {speaker_id}"
                        
                    speaker_class = f"speaker-{speaker_id % 2}"  # Alternate between two speaker styles
                    
                    # Format time for display
                    start_time = segment['start']
                    end_time = segment['end']
                    time_format = lambda t: f"{int(t // 60):02d}:{int(t % 60):02d}"
                    
                    # Create message bubble with time data attributes for highlighting
                    text_content = segment['text'].strip()
                    # Skip empty segments
                    if not text_content:
                        continue
                        
                    # Make sure HTML is properly escaped
                    import html
                    text_content = html.escape(text_content)
                    
                    chat_html += f"""
                    <div class='chat-message {speaker_class}' data-start='{start_time}' data-end='{end_time}'>
                        <div class='speaker-name'>{speaker_name}</div>
                        <div class='message-bubble'>{text_content}</div>
                        <div class='message-time'>{time_format(start_time)} - {time_format(end_time)}</div>
                    </div>
                    """
                # Add conversation summary if LLM is available
                summary_markdown = ""
                if LLM_AVAILABLE:
                    try:
                        # Use summary and topics from process_audio_with_diarization if available
                        summary = result.get("summary", "")
                        topics = result.get("topics", [])
                        
                        # Create summary for the Summary tab
                        if summary:
                            summary_markdown += f"## 🤖 AI Summary\n\n{summary}\n\n"
                            
                            # Also add to chat HTML
                            chat_html += f"""
                            <div class='conversation-summary'>
                                <h3>🤖 AI Summary</h3>
                                <p>{summary}</p>
                            </div>
                            """
                        
                        if topics and len(topics) > 0:
                            summary_markdown += f"## 📌 Main Topics\n\n" + "\n".join([f"- {topic}" for topic in topics])
                            
                            # Also add to chat HTML
                            topics_html = "<ul>" + "".join([f"<li>{topic}</li>" for topic in topics]) + "</ul>"
                            chat_html += f"""
                            <div class='conversation-topics'>
                                <h4>📌 Main Topics</h4>
                                {topics_html}
                            </div>
                            """
                    except Exception as e:
                        #print(f"Error processing summary and topics: {e}")
                        summary_markdown = f"*Error processing summary: {e}*"
                
                chat_html += "</div>"
                
                # Add a small script to initialize the audio sync on load
                chat_html += """
                <script>
                document.addEventListener('DOMContentLoaded', function() {
                    console.log('Chat container loaded - initializing audio sync');
                    if (window.setupAudioSync) {
                        window.setupAudioSync();
                    } else {
                        console.log('setupAudioSync not available yet');
                    }
                });
                
                // Also try immediately if DOM is already loaded
                if (document.readyState === 'complete' || document.readyState === 'interactive') {
                    setTimeout(function() {
                        console.log('Trying immediate setup');
                        if (window.setupAudioSync) {
                            window.setupAudioSync();
                        }
                    }, 500);
                }
                </script>
                </div>
                """
                
                # Return chat HTML, audio for playback, summary, and status
                return chat_html, audio_path, summary_markdown, f"*Processing completed successfully! Identified {num_speakers} speakers.*"
            else:
                return "<div class='chat-container'>No conversation segments found</div>", None, "", "*Processing completed, but no conversation segments were found.*"
                
        except Exception as e:
            print(f"Error in process_chat: {e}")
            import traceback
            traceback.print_exc()
            return "<div class='chat-container'>Error processing audio</div>", None, "", f"*Error: {str(e)}*"
    # Connect the chat process button directly - using global pipeline
    btn_process.click(
        fn=lambda a, t, s, e, n, th: process_chat(a, t, s, e, n, th),
        inputs=[
            audio_input,
            task,
            segmentation_model,
            embedding_model,
            num_speakers,
            threshold
        ],
        outputs=[
            chat_container,   # Chat bubbles container
            audio_playback,   # Audio playback
            output_conversation, # Summary tab
            status            # Status message
        ]
    )
    
    # === Trisha's Audio Lab Processing Function === 🧪
    def enhance_audio_lab(audio_path, method, highpass_cutoff, reduce_loud_segments):
        """
        🧪 Main function for Trisha's Audio Lab processing
        """
        if not audio_path:
            return (
                None,  # enhanced_audio_output
                "*Please upload an audio file first, Hue! 📁*",  # processing_info
                None,  # original_waveform
                None,  # enhanced_waveform
                "*Upload an audio file to start the enhancement! ⚡*",  # comparison_info
                "*Upload audio to begin! 🎵*"  # lab_status
            )
        
        try:
            gr.Info("🧪 Trisha's Lab is firing up! Analyzing your audio...")
            
            # Load original audio for comparison
            original_audio, original_sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Process the audio using Trisha's advanced denoising
            enhanced_audio, enhanced_sr, processing_details = denoise_audio_advanced(
                audio_path, 
                method=method, 
                highpass_cutoff=highpass_cutoff, 
                reduce_loud_segments=reduce_loud_segments
            )
            
            # Create temporary file for the enhanced audio
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                enhanced_path = tmp_file.name
                sf.write(enhanced_path, enhanced_audio, enhanced_sr)
            
            gr.Info("✨ Audio enhancement complete! Generating visualizations...")
            
            # Create processing report
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
            
            # Generate before/after waveform comparison
            original_plot = plot_waveform(original_audio, original_sr, title="📊 Original Audio")
            enhanced_plot = plot_waveform(enhanced_audio, enhanced_sr, title="✨ Enhanced Audio")
            
            # Create comparison info
            original_rms = np.sqrt(np.mean(original_audio**2))
            enhanced_rms = np.sqrt(np.mean(enhanced_audio**2))
            noise_reduction = ((original_rms - enhanced_rms) / original_rms) * 100 if original_rms > 0 else 0
            
            comparison_text = f"""
            ## 🔍 Before vs After Analysis
            
            **Original RMS Energy:** {original_rms:.4f}
            **Enhanced RMS Energy:** {enhanced_rms:.4f}
            **Estimated Noise Reduction:** {noise_reduction:.1f}%
            
            *Lower RMS typically indicates successful noise reduction! 📉*
            """
            
            success_message = f"*🎉 Enhancement complete! Applied {len(processing_details['processing_steps'])} processing steps.*"
            
            gr.Info("🎉 All done! Your enhanced audio is ready!")
            
            return (
                enhanced_path,  # enhanced_audio_output
                processing_report,  # processing_info  
                original_plot,  # original_waveform
                enhanced_plot,  # enhanced_waveform
                comparison_text,  # comparison_info
                success_message  # lab_status
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"💥 Lab explosion! Error: {str(e)}"
            return (
                None,  # enhanced_audio_output
                f"**Error:** {error_msg}",  # processing_info
                None,  # original_waveform
                None,  # enhanced_waveform  
                f"*{error_msg}*",  # comparison_info
                f"*{error_msg}*"  # lab_status
            )

    # Audio analysis functions
    def analyze_audio(audio_path):
        if not audio_path:
            return None, None, None, None, "*Error: No audio file provided*", None
        
        try:
            # Show info message
            gr.Info("Loading and analyzing audio file...")
            
            # Get the global pipeline
            pipeline = get_global_pipeline()
            
            # Use the pipeline to process the audio
            # This will handle GPU setup, model loading, etc.
            pipeline_result = pipeline.process_audio(
                audio_path=audio_path,
                task="transcribe",  # Default task
                segmentation_model="",  # Use default
                embedding_model="",  # Use default
                num_speakers=2,  # Default
                threshold=0.5  # Default
            )
            
            # Extract duration from pipeline result
            duration = pipeline_result.get("duration", 0)
            
            # Get audio information using utility function
            audio_features = get_audio_info(audio_path)
            
            # Load audio for visualizations
            audio, sr = process_audio_file(audio_path)
            
            # Format the information as markdown
            info_text = f"""
            ### 🎛️ Audio Information
            
            - **Duration**: {audio_features.get('duration', 0):.2f} seconds
            - **Sample Rate**: {audio_features.get('frame_rate', sr)} Hz
            - **Channels**: {audio_features.get('channels', 1)}
            - **Format**: {audio_features.get('format', 'Unknown')}
            - **Bit Depth**: {audio_features.get('sample_width', 2) * 8} bits
            - **Bitrate**: {audio_features.get('bitrate', 0) / 1000:.1f} kbps
            
            ### 📊 Audio Analysis
            
            - **RMS Energy**: {audio_features.get('rms', 0):.4f}
            - **Zero Crossing Rate**: {audio_features.get('zero_crossing_rate', 0):.4f}
            """
            
            # Add spectral features if available
            if 'spectral_centroid' in audio_features:
                info_text += f"\n- **Spectral Centroid**: {audio_features['spectral_centroid']:.2f} Hz"
            
            if 'spectral_bandwidth' in audio_features:
                info_text += f"\n- **Spectral Bandwidth**: {audio_features['spectral_bandwidth']:.2f} Hz"
                
            if 'spectral_rolloff' in audio_features:
                info_text += f"\n- **Spectral Rolloff**: {audio_features['spectral_rolloff']:.2f} Hz"
                
            if 'spectral_contrast' in audio_features:
                info_text += f"\n- **Spectral Contrast**: {audio_features['spectral_contrast']:.4f}"
            
            gr.Info("Generating visualizations...")
            
            # Generate visualizations using our enhanced functions
            waveform = plot_waveform(audio, sr, title="📊 Waveform")
            spectrogram = plot_spectrogram(audio, sr, title="🔊 Spectrogram")
            pitch_track = plot_pitch_track(audio, sr, title="🎵 Pitch Track")
            chromagram = plot_chromagram(audio, sr, title="🎹 Chromagram")
            
            # Get speaker segments from pipeline result
            segments = pipeline_result.get("segments", [])
            if segments:
                # Add speaker info to output
                info_text += "\n\n### 🎤 Speaker Segments\n"
                for segment in segments:
                    info_text += f"\n- **{segment['speaker']}**: {segment['start']:.2f}s - {segment['end']:.2f}s (Duration: {segment['end']-segment['start']:.2f}s)"
            else:
                info_text += f"\n\n### 🎤 Speaker Analysis\n\n*Speaker diarization unavailable or no speakers detected*"
            
            gr.Info("Analysis completed successfully!")
            return waveform, spectrogram, pitch_track, chromagram, "*Analysis completed successfully!*", info_text
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, None, None, f"*Error during analysis: {str(e)}*", None
    # Connect Trisha's Audio Lab enhance button 🧪
    btn_enhance.click(
        fn=enhance_audio_lab,
        inputs=[
            lab_audio_input,
            denoise_method,
            highpass_cutoff,
            reduce_loud
        ],
        outputs=[
            enhanced_audio_output,
            processing_info,
            original_waveform,
            enhanced_waveform,
            comparison_info,
            lab_status
        ],
        show_progress=True
    )
    
    # Connect the analyze button
    btn_analyze.click(
        fn=lambda a: analyze_audio(a),
        inputs=[audio_analysis_input],
        outputs=[waveform_plot, spectrogram_plot, pitch_plot, chroma_plot, analysis_status, audio_info],
        show_progress=True
    )
    
    # Footer
    gr.Markdown(CREDITS, elem_classes="footer")

# Only launch the app when running as main script
if __name__ == "__main__":
    # Launch with queue enabled
    demo.queue().launch()