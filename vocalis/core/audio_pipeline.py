"""
Audio Processing Pipeline for Vocalis

This module provides a flexible pipeline for audio processing tasks including
transcription, diarization, and LLM-based enhancements.
"""

import sys
import time
import os
import json
import hashlib
import re
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import traceback

# Import from diar.py
# Avoid importing heavy optional deps here; pull utilities lazily when needed

# Import LLM helper if available
try:
    from vocalis.llm import llm_helper
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("LLM helper not available. Some features rewill be disabled.")

# Cache for storing loaded models and GPU setup status
_PIPELINE_CACHE = {
    'transcription_model': None,
    'diarizer': None,
    'gpu_setup_done': False
}

class AudioProcessingPipeline:
    """
    A pipeline for processing audio with transcription, diarization, and LLM enhancements.
    
    This class handles the complete audio processing workflow and manages GPU resources
    efficiently by reusing models and controlling memory usage.
    """
    
    def __init__(self):
        """Initialize the audio processing pipeline."""
        self.gpu_available = self._setup_gpu()
        self.transcription_model = None
        self.diarizer = None
        self.llm_model = None
    
    def _setup_gpu(self) -> bool:
        """Configure GPU settings and optimize memory usage."""
        print("Setting up GPU...")
        # Check if GPU setup has already been done
        if _PIPELINE_CACHE['gpu_setup_done']:
            return True
            
        if not torch.cuda.is_available():
            print("No GPU available - running on CPU")
            return False
            
        try:
            # Get GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            cuda_version = torch.version.cuda
            print(f"Call Stack: {traceback.format_stack()}")

            # Print GPU information
            print("🚀 GPU ACCELERATION ENABLED 🚀")
            print(f"GPU Device: {gpu_name}")
            print(f"Number of GPUs: {gpu_count}")
            print(f"Total GPU Memory: {int(total_memory_mb)} MB")
            print(f"CUDA Version: {cuda_version}")
            print(f"PyTorch CUDA: {torch.version.cuda}")
            
            # Check for cuDNN
            if hasattr(torch.backends, 'cudnn'):
                print(f"cuDNN Version: {torch.backends.cudnn.version()}")
                print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
                
                # Enable TF32 for faster processing on Ampere GPUs (RTX 30xx, A100, etc)
                torch.backends.cuda.matmul.allow_tf32 = True
                
            # Check ONNX Runtime providers for diarization
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                print(f"🔍 ONNX Runtime Providers: {providers}")
                if 'CUDAExecutionProvider' not in providers:
                    print("⚠️ ONNX Runtime doesn't have CUDA support - diarization will use CPU")
                    print("💡 TIP: To enable GPU for diarization, install onnxruntime-gpu package")
            except ImportError:
                print("⚠️ Could not check ONNX Runtime providers - using default configuration")
                torch.backends.cudnn.allow_tf32 = True
                
                # Set cuDNN to benchmark mode for optimal performance
                torch.backends.cudnn.benchmark = True
            
            # Set memory allocation strategy
            if hasattr(torch.cuda, 'memory_stats'):
                # Use more aggressive memory allocation for RTX 4090
                torch.cuda.empty_cache()
                # Allow PyTorch to allocate memory as needed
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    # Use up to 90% of GPU memory
                    torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Mark GPU setup as done
            _PIPELINE_CACHE['gpu_setup_done'] = True
            return True
            
        except Exception as e:
            print(f"Error setting up GPU: {e}")
            return False
    
    def _clear_gpu_memory(self):
        """Free up GPU memory after processing."""
        print("Clearing GPU memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
    
    def _get_gpu_memory_info(self) -> Dict[str, str]:
        """Get current GPU memory usage information."""
        print("Getting GPU memory info...")
        if not torch.cuda.is_available():
            return {"error": "GPU not available"}
            
        try:
            # Get memory information
            if hasattr(torch.cuda, 'memory_stats'):
                stats = torch.cuda.memory_stats(0)
                allocated = stats.get('allocated_bytes.all.current', 0) / (1024 * 1024)
                reserved = stats.get('reserved_bytes.all.current', 0) / (1024 * 1024)
                free = stats.get('reserved_bytes.all.current', 0) / (1024 * 1024) - allocated
                
                # Get total memory
                total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                
                # Calculate utilization percentage
                utilization = (allocated / total) * 100 if total > 0 else 0
                
                return {
                    'total_mb': f"{total:.2f} MB",
                    'reserved_mb': f"{reserved:.2f} MB",
                    'allocated_mb': f"{allocated:.2f} MB",
                    'free_mb': f"{total - allocated:.2f} MB",
                    'utilization_percent': f"{utilization:.2f}%"
                }
            else:
                # Fallback for older PyTorch versions
                total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
                allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                free = reserved - allocated
                
                # Calculate utilization percentage
                utilization = (allocated / total) * 100 if total > 0 else 0
                
                return {
                    'total_mb': f"{total:.2f} MB",
                    'reserved_mb': f"{reserved:.2f} MB",
                    'allocated_mb': f"{allocated:.2f} MB",
                    'free_mb': f"{free:.2f} MB",
                    'utilization_percent': f"{utilization:.2f}%"
                }
        except Exception as e:
            return {"error": f"Error getting GPU memory info: {str(e)}"}
    
    def load_transcription_model(self, model_name: str = "openai/whisper-large-v3") -> bool:
        """
        Load the transcription model.
        
        Args:
            model_name: Name of the Whisper model to load
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # Check if model is already loaded in cache
        if _PIPELINE_CACHE['transcription_model'] is not None:
            self.transcription_model = _PIPELINE_CACHE['transcription_model']
            print(f"Using cached transcription model: {model_name}")
            return True
            
        try:
            from transformers import pipeline
            
            # Determine device based on GPU availability
            device = "cuda:0" if self.gpu_available else "cpu"
            print(f"Loading transcription model on {device}...")
            
            # Load model with appropriate device
            self.transcription_model = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=device,
                torch_dtype=torch.float16 if self.gpu_available else torch.float32
            )
            
            # Cache the model for future use
            _PIPELINE_CACHE['transcription_model'] = self.transcription_model
            return True
            
        except Exception as e:
            print(f"Error loading transcription model: {e}")
            return False
    
    def load_diarizer(self, segmentation_model: str, embedding_model: str, 
                     num_speakers: int = 2, threshold: float = 0.5) -> bool:
        """
        Load the speaker diarization model.
        
        Args:
            segmentation_model: Path to segmentation model
            embedding_model: Path to embedding model
            num_speakers: Number of speakers to detect (0 for auto)
            threshold: Clustering threshold
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # Check if model is already loaded in cache with same parameters
        if _PIPELINE_CACHE['diarizer'] is not None:
            diarizer = _PIPELINE_CACHE['diarizer']
            # Only reuse if parameters match
            if (diarizer.segmentation_model == segmentation_model and 
                diarizer.embedding_model == embedding_model and
                diarizer.threshold == threshold):
                
                # Update num_speakers if needed
                diarizer.num_speakers = num_speakers
                self.diarizer = diarizer
                print("Using cached diarizer")
                return True
        
        try:
            # Import locally to avoid heavy deps at module import time
            from vocalis.core.diar import SpeakerDiarizer
            # Create SpeakerDiarizer instance
            # Let's try a more direct approach with SpeakerDiarizer
            print("🎯 Setting up diarizer with optimized configuration")
            
            # First, let's check if sherpa-onnx is compiled with GPU support
            sherpa_gpu_support = False
            try:
                # Try to import sherpa_onnx to check if it has GPU support
                import sherpa_onnx
                # Check if sherpa_onnx has the CUDA provider available
                if hasattr(sherpa_onnx, 'is_cuda_available'):
                    sherpa_gpu_support = sherpa_onnx.is_cuda_available()
                    print(f"💡 Sherpa-ONNX CUDA support: {'Available' if sherpa_gpu_support else 'Not available'}")
                else:
                    # If we can't directly check, we'll try to infer from the build info
                    if hasattr(sherpa_onnx, 'build_info'):
                        build_info = sherpa_onnx.build_info()
                        print(f"💡 Sherpa-ONNX build info: {build_info}")
                        sherpa_gpu_support = 'CUDA' in build_info or 'GPU' in build_info
            except (ImportError, AttributeError) as e:
                print(f"⚠️ Could not check sherpa-onnx GPU support: {e}")
            
            # Try to use GPU if available
            if self.gpu_available and sherpa_gpu_support:
                print("🚀 Attempting to use GPU for diarization")
                try:
                    self.diarizer = SpeakerDiarizer(
                        segmentation_model=segmentation_model,
                        embedding_model=embedding_model,
                        num_speakers=num_speakers,
                        threshold=threshold,
                        use_gpu=True  # Try to use GPU
                    )
                    print("✅ Successfully initialized diarizer with GPU support")
                except Exception as gpu_error:
                    print(f"⚠️ GPU diarization failed: {str(gpu_error)}")
                    print("Falling back to CPU for diarization (this will be slower)")
                    self.diarizer = SpeakerDiarizer(
                        segmentation_model=segmentation_model,
                        embedding_model=embedding_model,
                        num_speakers=num_speakers,
                        threshold=threshold,
                        use_gpu=False  # Force CPU mode
                    )
            else:
                # No GPU support, use CPU directly
                if self.gpu_available and not sherpa_gpu_support:
                    print("⚠️ GPU available but sherpa-onnx was not compiled with GPU support")
                    print("💡 TIP: You might need to reinstall sherpa-onnx with GPU support")
                else:
                    print("Using CPU for diarization (GPU not available)")
                    
                self.diarizer = SpeakerDiarizer(
                    segmentation_model=segmentation_model,
                    embedding_model=embedding_model,
                    num_speakers=num_speakers,
                    threshold=threshold,
                    use_gpu=False  # Force CPU mode
                )
            
            # Cache the diarizer for future use
            _PIPELINE_CACHE['diarizer'] = self.diarizer
            return True
            
        except Exception as e:
            print(f"Error loading diarizer: {e}")
            return False
    
    def set_llm_model(self, model_name: str) -> bool:
        """
        Set the LLM model for name identification and summarization.
        
        Args:
            model_name: Name of the LLM model to use
            
        Returns:
            bool: True if model set successfully, False otherwise
        """
        if not LLM_AVAILABLE:
            print("LLM helper not available. Cannot set LLM model.")
            return False

    
    def transcribe(self, audio_path: str, task: str = "transcribe", 
                  return_timestamps: bool = True) -> Dict[str, Any]:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            task: Task type ('transcribe' or 'translate')
            return_timestamps: Whether to return timestamps
            
        Returns:
            Dict containing transcription results or error
        """
        # Make sure model is loaded
        if self.transcription_model is None:
            if not self.load_transcription_model():
                return {"error": "Failed to load transcription model"}
        
        try:
            # Log GPU memory before transcription if available
            if self.gpu_available:
                print("GPU memory before transcription:")
                print(self._get_gpu_memory_info())
                
            # Prepare generate kwargs based on task
            generate_kwargs = {"task": task}
                
            # Run transcription with parameters optimized for RTX 4090
            outputs = self.transcription_model(
                audio_path,
                chunk_length_s=60,  # Doubled chunk size for faster processing
                batch_size=512 if self.gpu_available else 32,  # Increased batch size for RTX 4090
                stride_length_s=5,  # Increased stride for faster processing
                generate_kwargs=generate_kwargs,
                return_timestamps=return_timestamps
            )
            
            # Log GPU memory after transcription if available
            if self.gpu_available:
                print("GPU memory after transcription:")
                print(self._get_gpu_memory_info())
                
            return outputs
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {"error": f"Transcription error: {str(e)}"}
    
    def diarize(self, audio_path: str, num_speakers: int = 2) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers to detect (0 for auto)
            
        Returns:
            List of diarization segments as dictionaries
        """
        # Make sure diarizer is loaded
        if self.diarizer is None:
            print("Diarizer not loaded. Cannot perform diarization.")
            return []
            
        try:
            # Update number of speakers if needed
            if self.diarizer.num_speakers != num_speakers:
                self.diarizer.num_speakers = num_speakers
                
            # Estimate number of speakers if set to auto
            if num_speakers == 0:
                print("Estimating number of speakers...")
                estimated_speakers = self.diarizer.estimate_num_speakers(audio_path)
                print(f"Estimated number of speakers: {estimated_speakers}")
                self.diarizer.num_speakers = estimated_speakers
            
            # Log GPU memory before diarization if available
            if self.gpu_available:
                print("GPU memory before diarization:")
                print(self._get_gpu_memory_info())
                
            # Process audio file for diarization
            segments = self.diarizer.process_file(audio_path)
            
            # Log GPU memory after diarization if available
            if self.gpu_available:
                print("GPU memory after diarization:")
                print(self._get_gpu_memory_info())
                
            # Convert DiarizationSegment objects to dictionaries for compatibility
            dict_segments = []
            for segment in segments:
                # Use to_dict() method if available, otherwise create a dict manually
                if hasattr(segment, 'to_dict'):
                    dict_segments.append(segment.to_dict())
                else:
                    dict_segments.append({
                        "speaker": f"Speaker {segment.speaker_id}" if hasattr(segment, 'speaker_id') else "Unknown",
                        "start": segment.start_time if hasattr(segment, 'start_time') else 0,
                        "end": segment.end_time if hasattr(segment, 'end_time') else 0,
                        "score": segment.score if hasattr(segment, 'score') else 1.0
                    })
            
            return dict_segments
            
        except Exception as e:
            print(f"Error during diarization: {e}")
            return []
    
    def identify_speaker_names(self, segments: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Identify speaker names from conversation segments.
        
        Args:
            segments: List of conversation segments
            
        Returns:
            Dictionary mapping speaker IDs to names
        """
        print("\n==== SPEAKER NAME IDENTIFICATION ====")
        print(f"LLM_AVAILABLE: {LLM_AVAILABLE}")
        
        if not LLM_AVAILABLE:
            print("LLM helper not available. Cannot identify speaker names.")
            return {}
        
        # Check if segments is empty or invalid
        if not segments or not isinstance(segments, list):
            print("No valid segments provided for speaker name identification")
            return {}
        
        print(f"Number of segments: {len(segments)}")
        if segments:
            print(f"First segment: {segments[0]}")
        
        # Validate segments structure
        valid_segments = []
        for i, segment in enumerate(segments[:10]):  # Check first 10 segments
            if not isinstance(segment, dict):
                print(f"Segment {i} is not a dictionary: {segment}")
                continue
                
            if 'speaker' not in segment:
                print(f"Segment {i} missing 'speaker' key: {segment}")
                continue
                
            if 'text' not in segment:
                print(f"Segment {i} missing 'text' key: {segment}")
                continue
                
            if not segment.get('text'):
                print(f"Segment {i} has empty text")
                continue
                
            valid_segments.append(segment)
            
        if not valid_segments:
            print("No valid segments found with required keys (speaker, text)")
            # Create some dummy segments for testing
            print("Creating dummy segments for testing...")
            valid_segments = [
                {"speaker": "Speaker 0", "text": "Hello, my name is Veronica.", "start": 0.0, "end": 2.0},
                {"speaker": "Speaker 1", "text": "Hi Veronica, I'm John.", "start": 2.0, "end": 4.0}
            ]
        
        print(f"Found {len(valid_segments)} valid segments for processing")
            
        try:
            # First try the LLM-based approach
            print("Attempting LLM-based name identification...")
            try:
                # Check if llm_helper is properly imported
                print(f"llm_helper module available: {hasattr(sys.modules, 'llm_helper')}")
                
                # Check if LLM is initialized
                llm = llm_helper.get_llm()
                print(f"LLM initialized: {llm is not None}")
                
                if llm is None:
                    print("LLM not initialized, skipping LLM-based approach")
                    raise Exception("LLM not initialized")
                
                print("Calling identify_speaker_names_llm...")
                llm_names = llm_helper.identify_speaker_names_llm(valid_segments)
                print(f"LLM name identification result: {llm_names}")
                
                if llm_names and len(llm_names) > 0:
                    print(f"LLM identified names: {llm_names}")
                    return llm_names
                else:
                    print("LLM did not identify any names")
            except Exception as e:
                print(f"LLM-based approach failed: {e}")
            
            # Fallback to rule-based approach
            print("Using fallback method for name identification")
            fallback_names = llm_helper.identify_speaker_names_fallback(valid_segments)
            print(f"Fallback name identification result: {fallback_names}")
            return fallback_names
            
        except Exception as e:
            print(f"Error identifying speaker names: {e}")
            return {}
    
    def generate_summary(self, segments: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the conversation.
        
        Args:
            segments: List of conversation segments
            
        Returns:
            Summary text
        """
        if not LLM_AVAILABLE:
            print("LLM helper not available. Cannot generate summary.")
            return ""
        
        try:
            return llm_helper.summarize_conversation(segments)
        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""
    
    def extract_topics(self, segments: List[Dict[str, Any]]) -> List[str]:
        """
        Extract topics from the conversation.
        
        Args:
            segments: List of conversation segments
            
        Returns:
            List of topics
        """
        if not LLM_AVAILABLE:
            print("LLM helper not available. Cannot extract topics.")
            return []
        
        try:
            return llm_helper.extract_topics(segments)
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []
    
    def process_audio(self, audio_path: str, task: str = "transcribe", 
                     segmentation_model: str = "pyannote/segmentation-3.0",
                     embedding_model: str = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx|25.3MB",
                     num_speakers: int = 2, threshold: float = 0.5,
                     use_cache: bool = True, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process audio with transcription and diarization.
        
        Args:
            audio_path: Path to audio file
            task: Task type ('transcribe' or 'translate')
            segmentation_model: Name of segmentation model
            embedding_model: Name of embedding model
            num_speakers: Number of speakers to detect (0 for auto)
            threshold: Clustering threshold
            
        Returns:
            Dict containing processing results or error
        """
        start_time = time.time()
        processing_times = {}

        # Result cache: keyed by audio file content hash and processing params
        cache_key = None
        if use_cache and not force_reprocess:
            try:
                cache_key = self._build_cache_key(
                    audio_path=audio_path,
                    task=task,
                    segmentation_model=segmentation_model,
                    embedding_model=embedding_model,
                    num_speakers=num_speakers,
                    threshold=threshold,
                )
                cached = self._cache_read(cache_key)
                if cached:
                    cached.setdefault("processing_times", {})
                    cached.setdefault("duration", 0)
                    cached["from_cache"] = True
                    return cached
            except Exception:
                # Fail closed: ignore cache on any error
                pass
        
        try:
            # Step 1: Transcribe audio
            print(f"Transcribing audio: {audio_path}")
            transcription_start = time.time()
            transcription = self.transcribe(audio_path, task)
            transcription_time = time.time() - transcription_start
            processing_times["transcription"] = transcription_time
            
            # Check for errors in transcription
            if isinstance(transcription, dict) and "error" in transcription:
                return transcription
                
            # Extract text and segments from transcription
            text = transcription.get("text", "")
            segments = transcription.get("chunks", [])
            
            # If no segments, try to get them from the 'segments' key
            if not segments and "segments" in transcription:
                segments = transcription["segments"]
                
            # If still no segments, create a single segment with the entire text
            if not segments:
                segments = [{"text": text, "start": 0, "end": 0}]
                
            # Step 2: Perform diarization
            print(f"Performing diarization with {num_speakers} speakers")
            diarization_start = time.time()
            
            # Load diarizer if needed
            if self.diarizer is None or self.diarizer.segmentation_model != segmentation_model or self.diarizer.embedding_model != embedding_model:
                print(f"Loading diarizer with models: {segmentation_model}, {embedding_model}")
                self.load_diarizer(segmentation_model, embedding_model, num_speakers, threshold)
                
            # Perform diarization
            diarization_segments = self.diarize(audio_path, num_speakers)
            diarization_time = time.time() - diarization_start
            processing_times["diarization"] = diarization_time
            
            # Step 3: Merge transcription with diarization
            print("Merging transcription with diarization")
            merged_segments = self._merge_transcription_with_diarization(transcription, diarization_segments)
            
            # Step 4: Get audio duration
            duration = self._get_duration_safe(audio_path, merged_segments)
            
            # Step 5: Identify speaker names if LLM is available
            speaker_names = {}
            if LLM_AVAILABLE and merged_segments:
                print("Identifying speaker names")
                speaker_names = self.identify_speaker_names(merged_segments)
                
                # Apply speaker names to segments
                if speaker_names:
                    for segment in merged_segments:
                        speaker_id = segment.get("speaker", "")
                        if speaker_id in speaker_names:
                            segment["speaker_name"] = speaker_names[speaker_id]
            
            # Step 6: Generate summary and extract topics if LLM is available
            summary = ""
            topics = []
            if LLM_AVAILABLE and merged_segments:
                print("Generating summary and extracting topics")
                summary = self.generate_summary(merged_segments)
                topics = self.extract_topics(merged_segments)
            
            # Calculate total processing time
            total_time = time.time() - start_time
            processing_times["total"] = total_time
            
            # Prepare result
            result = {
                "text": text,
                "segments": segments,
                "diarization_segments": diarization_segments,
                "merged_segments": merged_segments,
                "duration": duration,
                "processing_times": processing_times
            }
            
            # Add LLM-generated content if available
            if speaker_names:
                result["speaker_names"] = speaker_names
            if summary:
                result["summary"] = summary
            if topics:
                result["topics"] = topics
                
            # Persist to cache if enabled
            if use_cache:
                try:
                    if cache_key is None:
                        cache_key = self._build_cache_key(
                            audio_path=audio_path,
                            task=task,
                            segmentation_model=segmentation_model,
                            embedding_model=embedding_model,
                            num_speakers=num_speakers,
                            threshold=threshold,
                        )
                    to_store = dict(result)
                    to_store["from_cache"] = False
                    self._cache_write(cache_key, to_store)
                except Exception:
                    # Ignore cache write failures silently
                    pass

            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Processing error: {str(e)}"}

    # ----------------------------
    # Cache helpers
    # ----------------------------
    def _get_duration_safe(self, audio_path: str, merged_segments: List[Dict[str, Any]]) -> float:
        """Get duration using optional deps if present; fall back to segment end time."""
        try:
            # Lazy import to avoid hard dependency during tests
            from vocalis.core.audio_utils import get_audio_duration  # type: ignore
            return float(get_audio_duration(audio_path))
        except Exception:
            # Estimate duration from segments as a fallback
            if merged_segments:
                try:
                    return float(max(s.get("end", 0.0) for s in merged_segments))
                except Exception:
                    return 0.0
            return 0.0
    def _get_cache_dir(self) -> str:
        """Return cache directory, creating it if necessary."""
        cache_dir = os.environ.get("TW_CACHE_DIR", os.path.join(os.getcwd(), ".tw_cache"))
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            # Fallback to temp directory
            cache_dir = os.path.join("/tmp", "tw_cache")
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _compute_audio_hash(self, audio_path: str) -> str:
        """Compute SHA256 of the audio file contents in streaming fashion."""
        hasher = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _sanitize_for_key(self, value: str) -> str:
        """Sanitize a string so it can safely be used in filenames."""
        return re.sub(r"[^A-Za-z0-9._-]", "_", value)

    def _build_cache_key(self, audio_path: str, task: str, segmentation_model: str,
                         embedding_model: str, num_speakers: int, threshold: float) -> str:
        """Build a deterministic cache key from inputs and file hash."""
        file_hash = self._compute_audio_hash(audio_path)
        parts = [
            file_hash,
            task,
            segmentation_model,
            embedding_model,
            str(num_speakers),
            f"{threshold:.3f}",
        ]
        return "_".join(self._sanitize_for_key(p) for p in parts)

    def _cache_path(self, key: str) -> str:
        return os.path.join(self._get_cache_dir(), f"{key}.json")

    def _cache_read(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cache_write(self, key: str, data: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    
    def _merge_transcription_with_diarization(self, transcription, diarization_segments):
        """
        Merge transcription segments with diarization segments.
        
        Args:
            transcription: Transcription result from Whisper
            diarization_segments: List of diarization segments
            
        Returns:
            List of merged segments with speaker information
        """
        # Extract segments from transcription
        if isinstance(transcription, dict) and "segments" in transcription:
            transcript_segments = transcription["segments"]
        elif isinstance(transcription, dict) and "chunks" in transcription:
            transcript_segments = transcription["chunks"]
        else:
            transcript_segments = transcription
            
        # If no diarization segments, return transcript segments with alternating speakers
        if not diarization_segments:
            result = []
            for i, seg in enumerate(transcript_segments):
                result.append({
                    "speaker": f"Speaker {i % 2}",
                    "text": seg.get("text", ""),
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0)
                })
            return result
            
        # Lightweight merge without importing heavy diarization utilities
        def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
            start = max(a_start or 0.0, b_start or 0.0)
            end = min(a_end or start, b_end or start)
            return max(0.0, end - start)

        result = []
        for seg in transcript_segments:
            t_start = float(seg.get("start", 0.0))
            t_end = float(seg.get("end", t_start))
            best_speaker = None
            best_overlap = 0.0
            for d in diarization_segments:
                d_start = float(d.get("start", 0.0))
                d_end = float(d.get("end", d_start))
                ov = _overlap(t_start, t_end, d_start, d_end)
                if ov > best_overlap:
                    best_overlap = ov
                    best_speaker = d.get("speaker", None)
            result.append({
                "speaker": best_speaker if best_speaker is not None else "Speaker 0",
                "text": seg.get("text", ""),
                "start": t_start,
                "end": t_end,
            })
        return result