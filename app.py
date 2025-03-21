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

# Check for GPU availability at startup
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    print("\n" + "="*80)
    print(f"🚀 GPU DETECTED: {gpu_name} 🚀")
    print(f"Total GPU Memory: {total_memory_mb:.0f} MB")
    print(f"CUDA Version: {torch.version.cuda}")
    print("="*80 + "\n")
else:
    print("\n" + "="*80)
    print("⚠️ NO GPU DETECTED - RUNNING ON CPU ⚠️")
    print("Processing will be significantly slower without GPU acceleration.")
    print("="*80 + "\n")

# Import LLM helper module
try:
    import llm_helper
    LLM_AVAILABLE = True
    AVAILABLE_LLM_MODELS = list(llm_helper.AVAILABLE_MODELS.keys())
except ImportError:
    print("LLM helper module not available. Using fallback methods.")
    LLM_AVAILABLE = False
    AVAILABLE_LLM_MODELS = []

# Common name patterns for speaker recognition
COMMON_NAMES = [
    "Alex", "Alexandra", "Alexa", "Alexander", "Alice", "Alicia", "Amy", "Anna", "Anne", "Andrew", "Andy",
    "Bob", "Bobby", "Barbara", "Ben", "Benjamin", "Bill", "Billy", "Brian", "Bruce", 
    "Carl", "Carlos", "Carol", "Caroline", "Catherine", "Charlie", "Charlotte", "Chris", "Christine", "Christopher", "Cindy", "Claire",
    "Dan", "Daniel", "Dave", "David", "Debbie", "Deborah", "Diana", "Don", "Donald", "Donna", "Dorothy", "Doug", "Douglas",
    "Ed", "Edward", "Elizabeth", "Emily", "Emma", "Eric", "Eva", "Evelyn",
    "Frank", "Fred", "Frederick",
    "Gary", "George", "Grace", "Greg", "Gregory",
    "Harold", "Harry", "Heather", "Helen", "Henry", "Holly", "Howard",
    "Ian", "Irene", "Isaac", "Ivan",
    "Jack", "Jacob", "Jake", "James", "Jamie", "Jane", "Janet", "Jason", "Jean", "Jeff", "Jeffrey", "Jennifer", "Jenny", "Jeremy", "Jessica", "Jim", "Jimmy", "Joan", "Joe", "John", "Johnny", "Jonathan", "Joseph", "Josh", "Joshua", "Julia", "Julie", "Justin",
    "Karen", "Kate", "Katherine", "Kathleen", "Kathryn", "Kathy", "Katie", "Keith", "Kelly", "Ken", "Kenneth", "Kevin", "Kim", "Kimberly", "Kyle",
    "Larry", "Laura", "Lauren", "Lawrence", "Lee", "Leonard", "Leslie", "Linda", "Lisa", "Liz", "Louis", "Louise", "Lucy", "Lynn",
    "Maggie", "Marc", "Margaret", "Maria", "Mark", "Martha", "Martin", "Mary", "Matt", "Matthew", "Melissa", "Michael", "Michelle", "Mike", "Molly",
    "Nancy", "Natalie", "Nathan", "Neil", "Nicholas", "Nick", "Nicole", "Nina", "Noah", "Nora",
    "Olivia", "Oscar", "Owen",
    "Pam", "Pamela", "Patricia", "Patrick", "Paul", "Paula", "Peggy", "Peter", "Philip", "Phillip", "Phyllis",
    "Rachel", "Ralph", "Randy", "Raymond", "Rebecca", "Richard", "Rick", "Robert", "Robin", "Roger", "Ron", "Ronald", "Rose", "Roy", "Russell", "Ruth", "Ryan",
    "Sam", "Samantha", "Samuel", "Sandra", "Sandy", "Sara", "Sarah", "Scott", "Sean", "Sharon", "Sheila", "Shirley", "Sophia", "Spencer", "Stephanie", "Stephen", "Steve", "Steven", "Stuart", "Sue", "Susan", "Suzanne",
    "Tammy", "Ted", "Teresa", "Terry", "Thomas", "Tim", "Timothy", "Tina", "Todd", "Tom", "Tommy", "Tony", "Tracy",
    "Valerie", "Vanessa", "Vicky", "Victor", "Victoria", "Vincent", "Virginia",
    "Walter", "Wendy", "William", "Willy", "Winston",
    "Zachary", "Zoe"
]


# Import from local modules
from model import (
    get_speaker_diarization, read_wave,
    speaker_segmentation_models, embedding2models,
    get_local_segmentation_models, get_local_embedding_models
)
from utils.audio_processor import process_audio_file, extract_audio_features
from utils.audio_info import get_audio_info
from utils.visualizer import plot_waveform, plot_spectrogram, plot_pitch_track, plot_chromagram, plot_speaker_diarization
from diar import SpeakerDiarizer, format_as_conversation

# Load environment variables
load_dotenv()

# Cache for initialized components
_CACHE = {
    'gpu_setup_done': False,
    'whisper_model': None
}

# Set up GPU settings and memory management
def setup_gpu():
    """Configure GPU settings and optimize memory usage for RTX 4090"""
    # Check if GPU setup has already been done
    if _CACHE['gpu_setup_done']:
        return True
        
    if torch.cuda.is_available():
        # Enable cuDNN benchmark mode for optimal performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Enable TF32 precision on Ampere GPUs (like RTX 3090/4090)
        # This gives better performance with minimal precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory strategy
        torch.cuda.empty_cache()
        
        # Print detailed GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        
        # Get GPU memory information
        total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        
        print(f"🚀 GPU ACCELERATION ENABLED 🚀")
        print(f"GPU Device: {gpu_name}")
        print(f"Number of GPUs: {gpu_count}")
        print(f"Total GPU Memory: {total_memory_mb:.0f} MB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else 'Unknown'}")
        print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
        
        # Mark GPU setup as done
        _CACHE['gpu_setup_done'] = True
        
        return True
    else:
        print("⚠️ GPU NOT AVAILABLE - USING CPU ⚠️")
        print("Processing will be significantly slower without GPU acceleration.")
        print(f"PyTorch version: {torch.__version__}")
        return False

def clear_gpu_memory():
    """Free up GPU memory after processing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_gpu_memory_info():
    """Get current GPU memory usage information"""
    if not torch.cuda.is_available():
        return "GPU not available"
        
    try:
        # Get memory information for the current device
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - allocated_memory
        
        # Convert to MB for readability
        total_mb = total_memory / (1024 * 1024)
        reserved_mb = reserved_memory / (1024 * 1024)
        allocated_mb = allocated_memory / (1024 * 1024)
        free_mb = free_memory / (1024 * 1024)
        
        return {
            "total_mb": f"{total_mb:.2f} MB",
            "reserved_mb": f"{reserved_mb:.2f} MB",
            "allocated_mb": f"{allocated_mb:.2f} MB",
            "free_mb": f"{free_mb:.2f} MB",
            "utilization_percent": f"{(allocated_memory / total_memory) * 100:.2f}%"
        }
    except Exception as e:
        return f"Error getting GPU memory info: {str(e)}"

def check_gpu_efficiency():
    """Check if GPU is being used efficiently and provide recommendations"""
    if not torch.cuda.is_available():
        return "GPU not available - running on CPU"
    
    try:
        # Get GPU properties
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        # Check CUDA version
        cuda_version = torch.version.cuda
        major_version = int(cuda_version.split('.')[0]) if cuda_version else 0
        
        # Get memory information
        total_memory = props.total_memory / (1024 * 1024)  # MB
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
        utilization = (allocated_memory / total_memory) * 100
        
        recommendations = []
        
        # Check if using a modern GPU
        if props.major < 7:
            recommendations.append("⚠️ Using an older GPU architecture. Consider upgrading to an RTX series GPU for better performance.")
        
        # Check CUDA version
        if major_version < 11:
            recommendations.append("⚠️ Using an older CUDA version. Consider upgrading to CUDA 11.x or newer for better performance.")
        
        # Check memory utilization
        if utilization < 30:
            recommendations.append("⚠️ Low GPU memory utilization. Consider increasing batch size or model size to utilize GPU more efficiently.")
        elif utilization > 95:
            recommendations.append("⚠️ Very high GPU memory utilization. Consider reducing batch size to avoid out-of-memory errors.")
        
        # Check if using mixed precision
        if not torch.cuda.is_bf16_supported() and not torch.cuda.is_fp16_supported():
            recommendations.append("⚠️ GPU does not support efficient mixed precision. Performance may be limited.")
        
        # Return results
        result = {
            "gpu_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "cuda_version": cuda_version,
            "memory_utilization": f"{utilization:.2f}%",
            "recommendations": recommendations
        }
        
        return result
    except Exception as e:
        return f"Error checking GPU efficiency: {str(e)}"

# Load Whisper Model with optimizations for RTX 4090
def load_whisper_model(device="cuda:0" if torch.cuda.is_available() else "cpu"):
    """Load and optimize the Whisper model for maximum performance on RTX 4090"""
    # Check if model is already loaded
    if _CACHE['whisper_model'] is not None:
        return _CACHE['whisper_model']
        
    try:
        # Import the required modules directly
        import transformers
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        
        # Check if pipeline is available in transformers
        if not hasattr(transformers, 'pipeline'):
            print("Warning: transformers.pipeline not found, using alternative approach")
            
            # Alternative approach without pipeline - direct model usage
            try:
                print("Loading Whisper model directly without pipeline...")
                model_id = "openai/whisper-large-v3-turbo"
                
                # Load model with eager attention
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    attn_implementation="eager",
                    use_cache=True
                )
                model.to(device)
                
                # Load processor
                processor = AutoProcessor.from_pretrained(model_id)
                
                # Create a simple wrapper class that mimics the pipeline interface
                class WhisperWrapper:
                    def __init__(self, model, processor):
                        self.model = model
                        self.processor = processor
                        
                    def __call__(self, audio_path, **kwargs):
                        try:
                            # Load audio
                            with open(audio_path, "rb") as f:
                                audio_data = f.read()
                                
                            # Process audio
                            input_features = self.processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
                            input_features = input_features.to(device)
                            
                            # Generate tokens
                            predicted_ids = self.model.generate(input_features)
                            
                            # Decode tokens
                            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                            
                            return {"text": transcription}
                        except Exception as e:
                            print(f"Error in WhisperWrapper: {e}")
                            return {"text": f"Error: {str(e)}"}
                
                wrapper = WhisperWrapper(model, processor)
                print("Successfully created Whisper wrapper")
                
                # Cache the wrapper
                _CACHE['whisper_model'] = wrapper
                
                return wrapper
            except Exception as e:
                print(f"Error creating Whisper wrapper: {e}")
                return None
            
        # If pipeline is available, import it
        from transformers import pipeline
        
        # Use eager implementation directly since flash_attention causes issues
        print("Loading Whisper with eager implementation...")
        model_id = "openai/whisper-large-v3-turbo"
        
        # Load model with eager attention
        try:
            # Try to load with full optimization but using eager attention
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                attn_implementation="eager",  # Use eager instead of flash_attention_2
                use_cache=True
            )
            model.to(device)
            
            # Load processor
            processor = AutoProcessor.from_pretrained(model_id)
            
            # Create pipeline with optimized model
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                torch_dtype=torch.float16,
                device=device,
            )
            print("Successfully loaded Whisper with eager implementation and full optimization")
            
            # Cache the model
            _CACHE['whisper_model'] = pipe
            
            return pipe
            
        except Exception as model_err:
            print(f"Optimized model loading failed: {model_err}, falling back to simple pipeline")
            
            # Fallback to simpler pipeline implementation
            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3-turbo",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=device,
                model_kwargs={
                    "attn_implementation": "eager",  # Ensure eager implementation
                    "use_cache": True
                },
            )
            print("Successfully loaded Whisper with simple pipeline")
            
            # Cache the model
            _CACHE['whisper_model'] = pipe
            
            return pipe
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None

# Transcription function with GPU optimization
def transcribe_audio(audio_path, task="transcribe", return_timestamps=False):
    """Transcribe audio using Whisper model"""
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Ensure GPU is set up before loading model
            gpu_available = setup_gpu()
            if gpu_available:
                print("Using GPU for transcription")
            else:
                print("WARNING: Using CPU for transcription (much slower)")
                
            # Load model with appropriate device
            pipe = load_whisper_model(device="cuda:0" if gpu_available else "cpu")
            if not pipe:
                return {"error": "Failed to load transcription model"}
            
            # Prepare generate kwargs based on task
            generate_kwargs = {"task": task}
            
            # Log GPU memory before transcription if available
            if gpu_available:
                print("GPU memory before transcription:")
                print(get_gpu_memory_info())
                
            # Run transcription with parameters optimized for RTX 4090
            outputs = pipe(
                audio_path,
                chunk_length_s=60,  # Doubled chunk size for faster processing
                batch_size=512 if gpu_available else 32,  # Increased batch size for RTX 4090
                stride_length_s=5,  # Increased stride for faster processing
                generate_kwargs=generate_kwargs,
                return_timestamps=return_timestamps
            )
            
            # Log GPU memory after transcription if available
            if gpu_available:
                print("GPU memory after transcription (before cleanup):")
                print(get_gpu_memory_info())
            
            # Clear GPU memory
            clear_gpu_memory()
            
            # Log GPU memory after cleanup if available
            if gpu_available:
                print("GPU memory after cleanup:")
                print(get_gpu_memory_info())
            
            return outputs
    except Exception as e:
        return {"error": f"Transcription error: {str(e)}"}

# Speaker diarization function
def perform_diarization(audio_path, segmentation_model, embedding_model, num_speakers=2, threshold=0.5):
    """Perform speaker diarization on audio file"""
    try:
        # Create diarization model using our improved SpeakerDiarizer class
        diarizer = SpeakerDiarizer(
            segmentation_model=segmentation_model,
            embedding_model=embedding_model,
            num_speakers=num_speakers,
            threshold=threshold
        )
        
        # Estimate number of speakers if set to auto
        if num_speakers == 0:
            gr.Info("Estimating number of speakers...")
            num_speakers = diarizer.estimate_num_speakers(audio_path)
            gr.Info(f"Estimated number of speakers: {num_speakers}")
            diarizer.num_speakers = num_speakers
        
        # Process audio file
        segments = diarizer.process_file(audio_path)
        
        # Convert segments to dictionary format
        speakers = [segment.to_dict() for segment in segments]
        
        return speakers
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Diarization error: {str(e)}"}

# Combined transcription and diarization
def process_audio_with_diarization(audio_path, task, segmentation_model, embedding_model,
                                   num_speakers=2, threshold=0.5, return_timestamps=True):
    """Process audio with both transcription and speaker diarization optimized for RTX 4090"""
    try:
        # Configure GPU for optimal performance
        gpu_available = setup_gpu()
        if gpu_available:
            print("GPU setup successful, using CUDA for processing")
            # Log initial GPU memory state
            print("Initial GPU memory state:")
            print(get_gpu_memory_info())
        else:
            print("WARNING: GPU not available, falling back to CPU (this will be much slower)")
        
        # Track start time for performance metrics
        start_time = time.time()
        
        # Pre-load audio to avoid multiple file reads
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Create SpeakerDiarizer instance early to allow parallel initialization
        diarizer = SpeakerDiarizer(
            segmentation_model=segmentation_model,
            embedding_model=embedding_model,
            num_speakers=num_speakers,
            threshold=threshold
        )
        
        # Estimate number of speakers if set to auto
        if num_speakers == 0:
            gr.Info("Estimating number of speakers...")
            estimated_speakers = diarizer.estimate_num_speakers(audio_path)
            gr.Info(f"Estimated number of speakers: {estimated_speakers}")
            diarizer.num_speakers = estimated_speakers
        # Get transcription with timestamps
        transcription = transcribe_audio(audio_path, task, return_timestamps=True)
        transcription_time = time.time() - start_time
        
        # Log GPU memory state after transcription
        if gpu_available:
            print("GPU memory state after transcription:")
            print(get_gpu_memory_info())
        
        # Process audio file for diarization
        # Process audio file for diarization
        diarization_start = time.time()
        segments = diarizer.process_file(audio_path)
        diarization_time = time.time() - diarization_start
        
        # Log GPU memory state after diarization
        if gpu_available:
            print("GPU memory state after diarization:")
            print(get_gpu_memory_info())
        
        # Duration was already calculated above
        
        # If there was an error in transcription
        if isinstance(transcription, dict) and "error" in transcription:
            return transcription
        
        # Convert diarization segments to dictionary format
        speaker_segments = [segment.to_dict() for segment in segments]
        
        # Create transcript segments from Whisper output
        transcript_segments = []
        if "chunks" in transcription:
            for chunk in transcription["chunks"]:
                transcript_segments.append({
                    "text": chunk["text"],
                    "start": chunk["timestamp"][0],
                    "end": chunk["timestamp"][1]
                })
        else:
            # If no timestamps, create a single segment
            transcript_segments.append({
                "text": transcription["text"] if "text" in transcription else str(transcription),
                "start": 0,
                "end": duration
            })
            
        # Merge transcript with speaker information
        result = diarizer.create_transcript_with_speakers(transcript_segments, segments)
        
        # Format as conversation
        conversation = format_as_conversation(result)
        
        # Create speaker diarization plot
        diarization_plot = plot_speaker_diarization(speaker_segments, duration)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Prepare enhanced output with additional information
        output_json = {
            "text": transcription["text"] if "text" in transcription else "",
            "segments": result,
            "conversation": conversation,
            "diarization_plot": diarization_plot,
            "performance": {
                "transcription_time": f"{transcription_time:.2f}s",
                "diarization_time": f"{diarization_time:.2f}s",
                "total_time": f"{total_time:.2f}s",
                "audio_duration": f"{duration:.2f}s",
                "realtime_factor": f"{total_time/duration:.2f}x"
            }
        }
        
        # Add speaker names if LLM is available
        if LLM_AVAILABLE:
            try:
                # Identify speaker names
                speaker_names = llm_helper.identify_speaker_names_llm(result)
                if speaker_names and any(speaker_names.values()):
                    output_json["speaker_names"] = speaker_names
                    print(f"Identified speaker names: {speaker_names}")
                
                # Generate summary
                summary = llm_helper.summarize_conversation(result)
                if summary:
                    output_json["summary"] = summary
                    print(f"Generated summary: {summary}")
                
                # Extract topics
                topics = llm_helper.extract_topics(result)
                if topics:
                    output_json["topics"] = topics
                    print(f"Extracted topics: {topics}")
            except Exception as e:
                print(f"Error generating LLM content: {e}")
        
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
                with gr.Column(scale=2):
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
                        
                        # LLM model selection
                        if LLM_AVAILABLE and AVAILABLE_LLM_MODELS:
                            llm_model = gr.Dropdown(
                                choices=AVAILABLE_LLM_MODELS,
                                value=llm_helper.CURRENT_MODEL,
                                label="LLM Model for Summarization",
                                info="Select the LLM model to use for speaker identification and summarization"
                            )
                            
                            # Function to handle model change
                            def change_llm_model(model_name):
                                if LLM_AVAILABLE:
                                    llm_helper.set_current_model(model_name)
                                    print(f"Changed LLM model to: {model_name}")
                            
                            llm_model.change(fn=change_llm_model, inputs=llm_model, outputs=[])
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
        
        # Audio Tools Tab
        with gr.TabItem("🔧 Audio Tools"):
            gr.Markdown("Coming soon: Audio enhancement, noise reduction, and more tools!")
    
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
    
    # Define the processing function
    def process_audio(audio, task, segmentation_model, embedding_model, num_speakers, threshold):
        if not audio:
            return ("Upload an audio file to begin.", 
                   "*Error: No audio file provided*", 
                   None)
        
        try:
            # Process audio
            yield (
                "Processing audio...", 
                "*Starting transcription and diarization...*",
                None
            )
            
            # Process with transcription and diarization
            result = process_audio_with_diarization(
                audio, 
                task, 
                segmentation_model, 
                embedding_model, 
                num_speakers,
                threshold
            )
            
            # Check for errors
            if isinstance(result, dict) and "error" in result:
                yield (
                    "An error occurred during processing.",
                    f"*Error: {result['error']}*",
                    result
                )
                return
            
            # Handle successful processing
            if "conversation" in result:
                yield (
                    result["text"],
                    f"*Processing completed successfully! Identified {num_speakers} speakers.*",
                    result
                )
                return result["conversation"], f"*Processing completed successfully! Identified {num_speakers} speakers.*", result
            else:
                yield (
                    result["text"],
                    "*Processing completed, but speaker diarization might not be accurate.*",
                    result
                )
                return
            
        except Exception as e:
            yield (
                "An error occurred during processing.",
                f"*Error: {str(e)}*",
                {"error": str(e)}
            )
    
    # Connect the process button
    btn_process.click(
        fn=process_audio,
        inputs=[
            audio_input, 
            task, 
            segmentation_model, 
            embedding_model, 
            num_speakers, 
            threshold
        ],
        outputs=[
            output_raw, 
            status, 
            output_json
        ],
        show_progress=True
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
                # Find the most mentioned name that hasn't been assigned yet
                available_names = [name for name, count in sorted(name_mentions.items(), key=lambda x: x[1], reverse=True) 
                                if name not in detected_names.values()]                
                if available_names:
                    detected_names[speaker_id] = available_names[0]
        
        return detected_names

    # Function to generate chat bubbles from segments
    def process_chat(audio, task, segmentation_model, embedding_model, num_speakers, threshold, llm_model=None):
        # Set the LLM model if provided
        if LLM_AVAILABLE and llm_model is not None:
            try:
                if isinstance(llm_model, str) and llm_model in AVAILABLE_LLM_MODELS:
                    print(f"Setting LLM model to: {llm_model}")
                    llm_helper.set_current_model(llm_model)
            except Exception as e:
                print(f"Error setting LLM model: {e}")
                
        # Initialize default values
        chat_html = "<div class='chat-container'>Processing audio...</div>"
        audio_path = None
        summary_markdown = ""
        status_msg = "*Processing audio...*"
        
        try:
            # Check if audio is provided
            if audio is None:
                return "<div class='chat-container'>Please upload an audio file.</div>", None, "", "*Please upload an audio file.*"
                
            # Process audio file
            audio_path = audio
            result = process_audio_with_diarization(
                audio_path, 
                task, 
                segmentation_model, 
                embedding_model, 
                num_speakers, 
                threshold
            )
            
            # Generate chat HTML with debug info
            print(f"Result keys: {result.keys()}")
            
            if isinstance(result, dict) and "error" in result:
                return f"<div class='chat-container'>Error: {result['error']}</div>", None, "", f"*Error: {result['error']}*"
                
            if "segments" in result:
                segments = result["segments"]
                print(f"Found {len(segments)} segments")
                
                # Try to identify real speaker names
                speaker_names = identify_speaker_names(segments)
                print(f"Identified speaker names: {speaker_names}")
                
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
                        summary = llm_helper.summarize_conversation(segments)
                        topics = llm_helper.extract_topics(segments)
                        
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
                        print(f"Error generating summary: {e}")
                        summary_markdown = f"*Error generating summary: {e}*"
                        print(f"Error generating summary: {e}")
                
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
    
    # Connect the chat process button directly
    if LLM_AVAILABLE and AVAILABLE_LLM_MODELS:
        # With LLM model
        btn_process.click(
            fn=process_chat,
            inputs=[
                audio_input,
                task,
                segmentation_model,
                embedding_model,
                num_speakers,
                threshold,
                llm_model
            ],
            outputs=[
                chat_container,   # Chat bubbles container
                audio_playback,   # Audio playback
                output_conversation, # Summary tab
                status            # Status message
            ]
        )
    else:
        # Without LLM model
        btn_process.click(
            fn=lambda a, t, s, e, n, th: process_chat(a, t, s, e, n, th, None),
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
    
    # Audio analysis functions
    def analyze_audio(audio_path):
        if not audio_path:
            return None, None, None, None, "*Error: No audio file provided*", None
        
        try:
            # Update status
            yield None, None, None, None, "*Analyzing audio...*", None
            
            gr.Info("Loading and analyzing audio file...")
            
            # Process audio using our utility functions
            audio, sr = process_audio_file(audio_path)
            
            # Get comprehensive audio information
            audio_features = get_audio_info(audio_path)
            
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
            
            # Try to get speaker diarization
            try:
                gr.Info("Analyzing speakers...")
                diarizer = SpeakerDiarizer()
                segments = diarizer.process_file(audio_path)
                speaker_segments = [segment.to_dict() for segment in segments]
                
                # Add speaker info to output
                info_text += "\n\n### 🎤 Speaker Segments\n"
                for segment in speaker_segments:
                    info_text += f"\n- **{segment['speaker']}**: {segment['start']:.2f}s - {segment['end']:.2f}s (Duration: {segment['end']-segment['start']:.2f}s)"
            except Exception as diar_err:
                info_text += f"\n\n### 🎤 Speaker Analysis\n\n*Speaker diarization unavailable: {str(diar_err)}*"
            
            gr.Info("Analysis completed successfully!")
            yield waveform, spectrogram, pitch_track, chromagram, "*Analysis completed successfully!*", info_text
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield None, None, None, None, f"*Error during analysis: {str(e)}*", None
    
    # Connect the analyze button
    btn_analyze.click(
        fn=analyze_audio,
        inputs=[audio_analysis_input],
        outputs=[waveform_plot, spectrogram_plot, pitch_plot, chroma_plot, analysis_status, audio_info],
        show_progress=True
    )
    
    # Footer
    gr.Markdown(CREDITS, elem_classes="footer")

# Launch with queue enabled
demo.queue().launch()