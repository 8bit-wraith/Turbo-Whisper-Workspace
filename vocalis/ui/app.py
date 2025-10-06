"""
Gradio UI for Vocalis

This module provides a Gradio-based user interface for Vocalis.
"""

import os
import sys
import time
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv

# Import from vocalis modules
from vocalis.core.audio_pipeline import AudioProcessingPipeline
from vocalis.core.model import get_local_segmentation_models, get_local_embedding_models
from vocalis.core.model import speaker_segmentation_models, embedding2models
from vocalis.core.diar import format_as_conversation
from vocalis.utils.audio_info import get_audio_info
from vocalis.utils.visualizer import (
    plot_waveform, plot_spectrogram, plot_pitch_track, plot_chromagram, plot_speaker_diarization
)

# Load environment variables
load_dotenv()

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
        
        # Convert to speaker segments format for plotting (guard against missing keys)
        speaker_segments = []
        for segment in segments:
            speaker_segments.append({
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "speaker": segment.get("speaker", "Speaker 0"),
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

# Process chat function
def process_chat(audio, task, segmentation_model, embedding_model, num_speakers, threshold):
    """Process audio and generate chat bubbles"""
    if audio is None:
        return (
            "<div class='chat-container'>Upload an audio file and click 'Generate Chat' to start.</div>",
            "Upload an audio file and click 'Generate Chat' to start. A summary will appear here.",
            "",
            None,
            "*Upload an audio file to begin.*"
        )
    
    try:
        # Process the audio
        result = process_audio_with_diarization(
            audio, task, segmentation_model, embedding_model, num_speakers, threshold
        )
        
        # Check for errors
        if "error" in result:
            return (
                f"<div class='chat-container'>Error: {result['error']}</div>",
                f"Error: {result['error']}",
                "",
                None,
                f"*Error: {result['error']}*"
            )
        
        # Get segments and conversation
        segments = result.get("segments", [])
        conversation = result.get("conversation", "")
        
        # Create chat bubbles HTML
        chat_html = "<div class='chat-container'>"
        
        for i, segment in enumerate(segments):
            speaker = segment.get("speaker_name", segment.get("speaker", "Unknown"))
            text = segment.get("text", "")
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            
            # Determine speaker number (0 or 1) for styling
            speaker_num = 0 if "0" in speaker else 1
            
            chat_html += f"""
            <div class='chat-message speaker-{speaker_num}' data-start='{start_time}' data-end='{end_time}'>
                <div class='speaker-name'>{speaker}</div>
                <div class='message-bubble'>{text}</div>
                <div class='message-time'>{start_time:.1f}s - {end_time:.1f}s</div>
            </div>
            """
        
        chat_html += "</div>"
        
        # Get summary if available
        summary = result.get("summary", "No summary available.")
        
        # Format topics if available
        topics_html = ""
        if "topics" in result and result["topics"]:
            topics_html = "<div class='conversation-topics'><h4>Main Topics</h4><ul>"
            for topic in result["topics"]:
                topics_html += f"<li>{topic}</li>"
            topics_html += "</ul></div>"
        
        # Format performance metrics
        performance = result.get("performance", {})
        performance_text = f"""
        *Processing complete!*
        - Audio duration: {performance.get('audio_duration', 'N/A')}
        - Processing time: {performance.get('total_time', 'N/A')}
        - Realtime factor: {performance.get('realtime_factor', 'N/A')}
        """
        
        # Get raw text
        raw_text = result.get("text", "")
        
        return (
            chat_html,
            f"# Conversation Summary\n\n{summary}\n\n{topics_html}",
            raw_text,
            result,
            performance_text
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"Error processing audio: {str(e)}"
        return (
            f"<div class='chat-container'>{error_msg}</div>",
            error_msg,
            "",
            None,
            f"*{error_msg}*"
        )

# Analyze audio function
def analyze_audio(audio_path):
    """Analyze audio file and generate visualizations"""
    if not audio_path:
        return (
            "*Upload an audio file to begin analysis.*",
            None, None, None, None
        )
    
    try:
        # Get audio information
        audio_info = get_audio_info(audio_path)
        
        # Format audio info as markdown
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
        error_msg = f"Error analyzing audio: {str(e)}"
        return (
            f"*{error_msg}*",
            None, None, None, None
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
                <h1>🎧 Vocalis Audio Workspace ⚡</h1>
                <p>Advanced audio processing with Whisper V3 Turbo and Speaker Diarization</p>
            </div>
            """)
            
    with gr.Tabs(elem_classes="tabbed-content") as tabs:
        # Chat Bubbles Tab (now primary and merged with transcription)
        with gr.TabItem("💬 Vocalis Chat"):
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


    # Connect the process button to the process_chat function
    btn_process.click(
        fn=process_chat,
        inputs=[
            audio_input,
            task,
            segmentation_model,
            embedding_model,
            num_speakers,
            threshold
        ],
        outputs=[
            chat_container,
            output_conversation,
            output_raw,
            output_json,
            status
        ]
    )
    
    # Connect the analyze button to the analyze_audio function
    btn_analyze.click(
        fn=analyze_audio,
        inputs=[audio_analysis_input],
        outputs=[
            audio_info,
            waveform_plot,
            spectrogram_plot,
            pitch_plot,
            chroma_plot
        ]
    )

# Run the app if this file is executed directly
if __name__ == "__main__":
    demo.launch()