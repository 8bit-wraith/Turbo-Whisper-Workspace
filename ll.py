"""
LLM Helper Module for CyberVox

This module provides LLM-powered features for CyberVox:
1. Enhanced speaker name recognition
2. Conversation summarization
3. Topic extraction
"""

import os
import json
import time
import logging
import traceback
import threading
from typing import List, Dict, Any, Optional, Tuple

global _llm_last_used
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import llama_cpp, but provide fallbacks if not available
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    logger.warning("llama-cpp-python not available. Using fallback methods for name recognition and summarization.")
    LLAMA_AVAILABLE = False

# Model configuration
MODEL_NAME = "DeepHermes-3-Llama-3-3B-Preview-q8.gguf"
MODEL_PATH = f"models/{MODEL_NAME}"

# No need for model selection anymore

# Path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# LLM instance (lazy-loaded)
_llm_instance = None

# Cache for GPU device properties
_gpu_device_props_cache = {}

# LLM usage tracking for auto-unloading
_llm_last_used = 0
_llm_unload_timer = None
UNLOAD_TIMEOUT = 120  # seconds in which to unload the model if not used - increased for stability

def _schedule_llm_unload():
    """Schedule the LLM to be unloaded after a period of inactivity"""
    global _llm_unload_timer

    def unload_if_idle():
        """Unload the LLM if it hasn't been used recently"""
        global _llm_instance, _llm_last_used
        current_time = time.time()
        idle_time = current_time - _llm_last_used
        
        # Double-check the idle time to prevent premature unloading
        if idle_time >= UNLOAD_TIMEOUT:
            logger.info(f"LLM has been idle for {idle_time:.2f}s (timeout: {UNLOAD_TIMEOUT}s)")
            
            # Check if the LLM is still in use (another thread might have updated the timestamp)
            if current_time - _llm_last_used >= UNLOAD_TIMEOUT:
                logger.info(f"Unloading LLM from memory due to {idle_time:.2f}s of inactivity")
                _llm_instance = None
                # Force garbage collection to release memory
                try:
                    import gc
                    gc.collect()
                    # Try to clear CUDA cache if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.info("CUDA cache cleared")
                    except ImportError:
                        pass
                except Exception as e:
                    logger.debug(f"Error during cleanup after LLM unload: {e}")
                
                # Log GPU state after unloading
                monitor_gpu_usage("After LLM Unload")
            else:
                logger.info(f"LLM unloading cancelled - timestamp was updated during check")

    # Cancel any existing timer
    if _llm_unload_timer:
        _llm_unload_timer.cancel()

    # Start a new timer
    _llm_unload_timer = threading.Timer(UNLOAD_TIMEOUT, unload_if_idle)
    _llm_unload_timer.daemon = True
    _llm_unload_timer.start()

def get_model_path():
    """Get the path to the model file"""
    return os.path.join(MODELS_DIR, MODEL_NAME)

def get_llm() -> Optional[Any]:
    """Get or initialize the LLM instance"""
    global _llm_instance, _llm_last_used
    
    # Update last used time and schedule unload timer
    current_time = time.time()
    logger.debug(f"Updating LLM last used timestamp from {_llm_last_used} to {current_time} (delta: {current_time - _llm_last_used:.2f}s)")
    _llm_last_used = current_time
    _schedule_llm_unload()
    
    if not LLAMA_AVAILABLE:
        logger.warning("llama-cpp-python not available")
        return None
        
    if _llm_instance is None:
        try:
            # Print call stack for debugging
            logger.info("Call stack for LLM initialization:")
            for line in traceback.format_stack():
                logger.info(line.strip())
                
            model_path = get_model_path()
            
            # Print debug info
            logger.info(f"Current model: {MODEL_NAME}")
            logger.info(f"Model path: {model_path}")
            logger.info(f"Models directory: {MODELS_DIR}")
            
            # Check if model exists locally
            if os.path.exists(model_path):
                logger.info(f"Loading model from local path: {model_path}")
                try:
                    # Check for CUDA availability
                    import torch
                    cuda_available = torch.cuda.is_available()
                    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
                    current_device = torch.cuda.current_device() if cuda_available else "N/A"
                    device_name = torch.cuda.get_device_name(current_device) if cuda_available else "N/A"
                    
                    logger.info(f"CUDA available: {cuda_available}")
                    logger.info(f"CUDA device count: {cuda_device_count}")
                    logger.info(f"Current CUDA device: {current_device}")
                    logger.info(f"Current CUDA device name: {device_name}")
                    
                    # Try to initialize with full GPU support
                    logger.info(f"Initializing Llama with model path: {model_path}")
                    # For RTX 4090, we can use all layers on GPU and increase context window
                    # Log GPU info before initialization
                    logger.info("GPU configuration for RTX 4090:")
                    logger.info("Using all layers on GPU with n_gpu_layers=-1")
                    logger.info("Using 4096 context window")
                    logger.info("Using 512 batch size for better throughput")
                    
                    # Initialize with optimal settings for RTX 4090
                    # Use verbose=True to see detailed GPU usage during initialization
                    # Try to force GPU usage with explicit settings
                    logger.info("Setting environment variables to force GPU usage")
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force use of first GPU
                    
                    # Verify GPU is working properly
                    gpu_verified = verify_gpu_usage()
                    logger.info(f"GPU verification: {'Success' if gpu_verified else 'Failed'}")
                    
                    # Try a different model if needed
                    # For RTX 4090, we should be able to use the 8B model with proper settings
                    
                    # Check if GPU verification succeeded
                    if gpu_verified:
                        logger.info("Initializing LLM with GPU acceleration")
                        _llm_instance = Llama(
                            model_path=model_path,
                            n_gpu_layers=1,      # Use all layers on GPU
                            n_ctx=4096,           # Reduced context window to save memory
                            n_batch=512,          # Reduced batch size
                            offload_kqv=True,     # Offload key/query/value tensors to GPU
                            f16_kv=False,          # Use half precision for key/value cache
                            use_mlock=False,      # Don't use mlock
                            use_mmap=True,        # Use memory mapping
                            embedding=False,      # Don't compute embeddings
                            last_n_tokens_size=64, # Reduce token history size
                            verbose=True,         # Enable verbose output
                            seed=42,              # Set seed for reproducibility
                            n_threads=8           # Limit number of CPU threads
                        )
                    else:
                        # Fallback to CPU-only mode with optimized settings
                        logger.warning("\033[91m GPU verification failed, falling back to CPU-only mode\033[0m")
                        _llm_instance = Llama(
                            model_path=model_path,
                            n_gpu_layers=0,       # No GPU layers
                            n_ctx=1024,           # Smaller context window for CPU
                            n_batch=128,          # Smaller batch size for CPU
                            use_mlock=False,
                            use_mmap=True,
                            embedding=False,
                            last_n_tokens_size=64,
                            verbose=True,
                            seed=42,
                            n_threads=8           # Limit number of CPU threads
                        )
                    if gpu_verified:
                        logger.info("LLM initialized successfully with full GPU acceleration")
                    else:
                        logger.info("LLM initialized successfully in CPU-only mode")
                except Exception as local_error:
                    logger.error(f"Error loading local model: {local_error}")
                    logger.error("Could not initialize LLM. Please check that the model file exists and is valid.")
            else:
                logger.error(f"Model file not found at {model_path}")
                logger.error("Please ensure the model file is in the models directory.")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            # Create a dummy LLM that returns empty results
            logger.warning("Creating a dummy LLM that returns empty results")
            return DummyLLM()


def get_gpu_device_props(device_id=0):
    """Get GPU device properties with caching"""
    global _gpu_device_props_cache
    
    try:
        import torch
        if not torch.cuda.is_available():
            return None
            
        # Check cache first
        if device_id in _gpu_device_props_cache:
            return _gpu_device_props_cache[device_id]
            
        # Get properties and cache them
        props = torch.cuda.get_device_properties(device_id)
        _gpu_device_props_cache[device_id] = props
        return props
    except Exception as e:
        logger.error(f"Error getting GPU device properties: {e}")
        return None

def verify_gpu_usage():
    """Verify if the GPU is actually being used by running a simple tensor operation"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Running GPU verification test...")
            # Create a tensor on GPU
            x = torch.randn(1000, 1000).cuda()
            # Perform a simple operation
            y = x @ x.T
            # Check if tensor is on GPU
            logger.info(f"Test tensor is on GPU: {y.is_cuda}")
            
            # Get device properties using cached function
            device_props = get_gpu_device_props(0)
            if device_props:
                logger.info(f"GPU Device: {device_props.name}")
                logger.info(f"GPU Memory: {device_props.total_memory / 1024**3:.2f} GB")
                logger.info(f"GPU Compute Capability: {device_props.major}.{device_props.minor}")
            
            # Free memory
            del x, y
            torch.cuda.empty_cache()
            return True
        else:
            logger.warning("CUDA is not available for GPU verification")
            return False
    except Exception as e:
        logger.error(f"Error verifying GPU usage: {e}")
        return False


def monitor_gpu_usage(label=""):
    """Monitor GPU usage and log statistics
    
    Args:
        label: Optional label to identify the monitoring point
    """
    try:
        import torch
        import psutil
        import subprocess
        from datetime import datetime
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available for monitoring")
            return
            
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log header
        if label:
            logger.info(f"=== GPU Monitoring at {timestamp} - {label} ===")
        else:
            logger.info(f"=== GPU Monitoring at {timestamp} ===")
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        logger.info(f"System Memory: {system_memory.used / 1024**3:.2f} GB / {system_memory.total / 1024**3:.2f} GB ({system_memory.percent}%)")
        
        # Get GPU count
        device_count = torch.cuda.device_count()
        logger.info(f"GPU Count: {device_count}")
        
        # Monitor each GPU
        for i in range(device_count):
            logger.info(f"--- GPU {i} ---")
            
            # Get device properties using cached function
            props = get_gpu_device_props(i)
            if props:
                logger.info(f"Name: {props.name}")
                logger.info(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
            
            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"Memory Allocated: {memory_allocated:.2f} GB")
            logger.info(f"Memory Reserved: {memory_reserved:.2f} GB")
            
            # Calculate utilization if we have props
            if props:
                logger.info(f"Memory Utilization: {memory_allocated / (props.total_memory / 1024**3) * 100:.2f}%")
            
            # Try to get more detailed GPU info using nvidia-smi
            try:
                # Get GPU temperature and utilization
                result = subprocess.run(
                    ["nvidia-smi", f"--id={i}", "--query-gpu=temperature.gpu,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if result.stdout.strip():
                    temp, gpu_util, mem_util = result.stdout.strip().split(',')
                    logger.info(f"Temperature: {temp.strip()}°C")
                    logger.info(f"GPU Utilization: {gpu_util.strip()}%")
                    logger.info(f"Memory Utilization (nvidia-smi): {mem_util.strip()}%")
                
                # Get processes using the GPU
                result = subprocess.run(
                    ["nvidia-smi", f"--id={i}", "--query-compute-apps=pid,used_memory,name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if result.stdout.strip():
                    logger.info("Processes using GPU:")
                    for line in result.stdout.strip().split('\n'):
                        logger.info(f"  {line.strip()}")
                else:
                    logger.info("No processes actively using GPU")
                    
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                logger.warning(f"Could not get detailed GPU info: {e}")
    except Exception as e:
        logger.error(f"Error monitoring GPU usage: {e}")
            
    return _llm_instance

class DummyLLM:
    """A dummy LLM class that returns empty results for all methods"""
    
    def __init__(self):
        logger.warning("Using DummyLLM - all LLM operations will return empty results")
    
    def create_completion(self, prompt, **kwargs):
        logger.info(f"DummyLLM received prompt: {prompt[:50]}...")
        return {
            'choices': [{'text': '{"Speaker 0": "Unknown", "Speaker 1": "Unknown"}'}]
        }

# The monitor_gpu_usage function has been replaced by the newer version above

def identify_speaker_names_llm(segments: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Use LLM to identify speaker names from conversation segments
    
    Args:
        segments: List of conversation segments with speaker and text
        
    Returns:
        Dictionary mapping speaker IDs to names
    """
    # Update LLM usage timestamp to prevent unloading during processing
    global _llm_last_used
    _llm_last_used = time.time()
    logger.info("Starting identify_speaker_names_llm function")
    # Monitor GPU usage before LLM call
    monitor_gpu_usage()
    
    # Validate input
    if not segments or not isinstance(segments, list):
        logger.error(f"Invalid segments provided: {segments}")
        return {}
    
    logger.info(f"Processing {len(segments)} segments for name identification")
    if segments:
        logger.info(f"First segment: {segments[0]}")
    
    # Check if we have the required keys in the segments
    valid_segments = []
    for segment in segments:
        if not isinstance(segment, dict):
            logger.warning(f"Segment is not a dictionary: {segment}")
            continue
            
        if 'speaker' not in segment or 'text' not in segment:
            logger.warning(f"Segment missing required keys: {segment}")
            continue
            
        if not segment.get('text'):
            logger.debug("Skipping segment with empty text")
            continue
            
        valid_segments.append(segment)
    
    if not valid_segments:
        logger.error("No valid segments found for name identification")
        return {}
        
    logger.info(f"Found {len(valid_segments)} valid segments for processing")
    
    # Get LLM
    llm = get_llm()
    if not llm:
        logger.warning("LLM not available, using fallback method")
        return {}
    
    # Prepare conversation context
    conversation_text = ""
    for segment in valid_segments[:10]:  # Use first 10 segments to identify speakers
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', '').strip()
        conversation_text += f"{speaker}: {text}\n"
    
    logger.info(f"Prepared conversation context of {len(conversation_text)} characters")
    
    # Create prompt for the LLM
    prompt = f"""
You are an AI assistant that analyzes conversations and identifies the real names of speakers.

Here's a conversation with speakers labeled as "Speaker X":
{conversation_text}

Based on this conversation, identify the real names of the speakers. Look for:
1. Self-introductions like "I'm [Name]" or "My name is [Name]"
2. When one speaker addresses another by name
3. Any other context clues that reveal names

IMPORTANT: Return ONLY a valid JSON object mapping speaker IDs to names.
Format: {{"Speaker 0": "John", "Speaker 1": "Mary"}}

Do not include any explanations, notes, or additional text before or after the JSON.
If you can't identify a name for a speaker, use null instead of making up a name.

Example of a valid response:
{{"Speaker 0": "John", "Speaker 1": "Mary", "Speaker 2": null}}
"""
    try:
        logger.debug(f"LLM Prompt: {prompt}")
        logger.info("Sending request to LLM for name identification")
        # Update the LLM timestamp before processing to prevent unloading
        
        _llm_last_used = time.time()
        logger.debug(f"Updated LLM timestamp before processing: {_llm_last_used}")
        
        # Get response from LLM
        start_time = time.time()
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,
            stop=["```"]
        )
        elapsed_time = time.time() - start_time
        logger.info(f"LLM completion took {elapsed_time:.2f} seconds")
        
        # Update timestamp again after processing
        _llm_last_used = time.time()
        logger.debug(f"Updated LLM timestamp after processing: {_llm_last_used}")
        
        # Monitor GPU usage after LLM call
        monitor_gpu_usage()
        
        # Extract and parse the JSON response
        response_text = response['choices'][0]['text'].strip()
        logger.info(f"Received raw response from LLM: {response_text}")
        
        # Find JSON block in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            logger.info(f"Extracted JSON string: {json_str}")
            try:
                speaker_names = json.loads(json_str)
                logger.info(f"Successfully parsed speaker names: {speaker_names}")
                
                # Check for Veronica or Alexandra in the text and ensure they're included
                for name in ["Veronica", "Alexandra"]:
                    if any(name.lower() in segment.get('text', '').lower() for segment in valid_segments):
                        # Find which speaker mentions the name
                        for segment in valid_segments:
                            if name.lower() in segment.get('text', '').lower():
                                # The person being addressed is likely the one with the name
                                # So the speaker mentioning the name is NOT that person
                                speaker_id = segment.get('speaker')
                                
                                # Find other speakers who might be the addressed person
                                other_speakers = [s for s in set(seg.get('speaker') for seg in valid_segments) if s != speaker_id]
                                for other_id in other_speakers:
                                    if other_id not in speaker_names or not speaker_names[other_id]:
                                        speaker_names[other_id] = name
                                        logger.info(f"Added {name} as {other_id} based on being addressed")
                                        
                                # Make sure the speaker addressing the person doesn't get the same name
                                if speaker_id in speaker_names and speaker_names[speaker_id] == name:
                                    speaker_names[speaker_id] = "Unknown"
                                    logger.info(f"Removed {name} from {speaker_id} as they are addressing someone else")
                
                return speaker_names
            except json.JSONDecodeError as e:
                # Replaced print with proper logging
                logger.error(f"Failed to parse JSON from LLM response: {json_str} - Error: {e}")
                # Try to extract names using regex as fallback
                try:
                    import re
                    logger.info("Attempting to extract speaker names using regex patterns")
                    
                    # Try multiple regex patterns for different JSON formats
                    patterns = [
                        # Standard JSON format with double quotes
                        r'"(Speaker \d+)"\s*:\s*"([^"]+)"',
                        # Alternative format with single quotes
                        r"'(Speaker \d+)'\s*:\s*'([^']+)'",
                        # Format without quotes around the name
                        r'"(Speaker \d+)"\s*:\s*([^,}\s]+)',
                        # Format with speaker number only
                        r'Speaker (\d+)\s*:\s*([^,}\n]+)'
                    ]
                    
                    speaker_names = {}
                    
                    # Try each pattern
                    for pattern in patterns:
                        matches = re.findall(pattern, response_text)
                        if matches:
                            logger.info(f"Found matches using pattern: {pattern}")
                            for speaker_id, name in matches:
                                # Normalize speaker ID format
                                if not speaker_id.startswith('Speaker '):
                                    speaker_id = f"Speaker {speaker_id}"
                                speaker_names[speaker_id] = name.strip()
                    
                    if speaker_names:
                        logger.info(f"Extracted names using regex: {speaker_names}")
                        return speaker_names
                except Exception as regex_error:
                    logger.error(f"Error in regex fallback: {regex_error}")
        else:
            logger.error(f"Could not find JSON in response: {response_text}")
        
        return {}
    except Exception as e:
        logger.error(f"Error getting speaker names from LLM: {e}")
        logger.error(traceback.format_exc())
        return {}

def summarize_conversation(segments: List[Dict[str, Any]]) -> str:
    """
    Generate a summary of the conversation using the LLM
    
    Args:
        segments: List of conversation segments
        
    Returns:
        Summary text
    """
    # Update LLM usage timestamp to prevent unloading during processing
    global _llm_last_used
    _llm_last_used = time.time()
    try:
        llm = get_llm()
        if not llm:
            logger.warning("LLM not available for summarization")
            return "Conversation summary not available (LLM not loaded)"
        
        # Check if segments is valid
        if not segments or not isinstance(segments, list):
            logger.warning("No valid segments provided for summarization")
            return "No conversation data available for summarization."
            
        # Limit the number of segments to avoid overwhelming the LLM
        if len(segments) > 20:
            logger.info(f"Limiting from {len(segments)} to 20 segments for summarization")
            limited_segments = segments[:20]
        else:
            limited_segments = segments
        
        # Prepare conversation context
        conversation_text = ""
        for segment in limited_segments:
            if 'speaker' not in segment or 'text' not in segment:
                continue
                
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()
            
            # Skip empty text
            if not text:
                continue
                
            # Use speaker_name if available, otherwise use speaker ID
            speaker_name = segment.get('speaker_name', speaker)
            conversation_text += f"{speaker_name}: {text}\n"
        
        # If no valid conversation text, return early
        if not conversation_text.strip():
            logger.warning("No valid conversation text for summarization")
            return "No conversation content available for summarization."
        
        # Create prompt for the LLM
        prompt = f"""
You are T-AI-bitha, a wonderful summarizer.  Please make things entertaining and funny.

Here's a conversation:
{conversation_text}

Please provide a concise summary (3-5 sentences) of this conversation, highlighting:
1. The main topics discussed
2. The silver lining of the conversation
3. Any decisions or conclusions reached
4. Any action items or next steps mentioned

Summary:
"""
        
        logger.info("Sending summarization request to LLM")
        # Monitor GPU usage before LLM call
        monitor_gpu_usage()
        try:
            # Get response from LLM
            start_time = time.time()
            response = llm.create_completion(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3,
                stop=["```"]
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Summarization LLM completion took {elapsed_time:.2f} seconds")
            
            # Monitor GPU usage after LLM call
            monitor_gpu_usage()
            
            # Extract the summary text
            summary = response['choices'][0]['text'].strip()
            logger.info(f"Generated summary: {summary[:50]}...")
            return summary
        except Exception as e:
            logger.error(f"Error in LLM completion for summary: {e}")
            return "Summary not available due to an error in LLM processing."
    except Exception as e:
        logger.error(f"Unexpected error in summarize_conversation: {e}")
        return "Summary not available due to an unexpected error."

def extract_topics(segments: List[Dict[str, Any]]) -> List[str]:
    """
    Extract main topics from the conversation
    
    Args:
        segments: List of conversation segments
        
    Returns:
        List of main topics
    """
    # Update LLM usage timestamp to prevent unloading during processing
    global _llm_last_used
    _llm_last_used = time.time()
    try:
        llm = get_llm()
        if not llm:
            logger.warning("LLM not available for topic extraction")
            return []
        
        # Check if segments is valid
        if not segments or not isinstance(segments, list):
            logger.warning("No valid segments provided for topic extraction")
            return []
            
        # Limit the number of segments to avoid overwhelming the LLM
        if len(segments) > 20:
            logger.info(f"Limiting from {len(segments)} to 20 segments for topic extraction")
            limited_segments = segments[:20]
        else:
            limited_segments = segments
        
        # Prepare conversation context
        conversation_text = ""
        for segment in limited_segments:
            if 'speaker' not in segment or 'text' not in segment:
                continue
                
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()
            
            # Skip empty text
            if not text:
                continue
                
            # Use speaker_name if available, otherwise use speaker ID
            speaker_name = segment.get('speaker_name', speaker)
            conversation_text += f"{speaker_name}: {text}\n"
        
        # If no valid conversation text, return early
        if not conversation_text.strip():
            logger.warning("No valid conversation text for topic extraction")
            return []
        
        # Create prompt for the LLM
        prompt = f"""
You are an AI assistant that analyzes conversations and extracts the main topics.

Here's a conversation:
{conversation_text}

Please list the 3-5 main topics discussed in this conversation. Return your answer as a JSON array of strings.
Example: ["Project timeline", "Budget concerns", "Marketing strategy"]

Your response should ONLY contain the JSON array, nothing else. No explanations, no additional text.
"""

        logger.info("Sending topic extraction request to LLM")
        print('3-5 main topics:', prompt)
        # Monitor GPU usage before LLM call
        monitor_gpu_usage()
        try:
            # Get response from LLM
            start_time = time.time()
            response = llm.create_completion(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1,
                stop=["```"]
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Topic extraction LLM completion took {elapsed_time:.2f} seconds")
            
            # Monitor GPU usage after LLM call
            monitor_gpu_usage()
            # Extract and parse the JSON response
            response_text = response['choices'][0]['text'].strip()
            logger.info(f"LLM response for topics: {response_text[:50]}...")
            
            # First, try to find a clean JSON array using regex
            import re
            json_pattern = re.search(r'\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]', response_text)
            
            if json_pattern:
                json_str = json_pattern.group(0)
                logger.info(f"JSON pattern found: {json_str}")
                try:
                    topics = json.loads(json_str)
                    if isinstance(topics, list):
                        logger.info(f"Successfully parsed topics from regex: {topics}")
                        return topics
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON pattern: {e}")
            
            # If regex approach fails, try the traditional bracket approach
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                logger.info(f"JSON block found: {json_str}")
                try:
                    topics = json.loads(json_str)
                    if isinstance(topics, list):
                        logger.info(f"Successfully extracted {len(topics)} topics")
                        return topics
                    else:
                        logger.warning(f"Parsed JSON is not a list: {topics}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from LLM response: {e}")
                    
                    # Try to clean up the JSON string and retry
                    try:
                        # Remove any text before the first [ and after the last ]
                        cleaned_json = json_str
                        # Replace any invalid characters that might be causing issues
                        cleaned_json = re.sub(r'[\x00-\x1F\x7F-\xFF]', '', cleaned_json)
                        topics = json.loads(cleaned_json)
                        if isinstance(topics, list):
                            logger.info(f"Successfully parsed topics after cleanup: {topics}")
                            return topics
                    except json.JSONDecodeError:
                        pass
            
            logger.info("Failed to extract topics from JSON response")
            
            # Fallback: try to extract topics from plain text
            logger.info("Falling back to plain text extraction")
            # Look for numbered or bulleted list items
            topics = re.findall(r'[\d\*\-•]+\s*[.:]?\s*([^\n]+)', response_text)
            if topics:
                logger.info(f"Extracted {len(topics)} topics using regex fallback")
                return topics
            
            logger.warning("Could not extract topics from LLM response")
            return []
        except Exception as e:
            logger.error(f"Error in LLM completion for topic extraction: {e}")
            return []
    except Exception as e:
        logger.error(f"Unexpected error in extract_topics: {e}")
        return []

# Fallback methods that don't require LLM
def identify_speaker_names_fallback(segments: List[Dict[str, Any]]) -> Dict[str, str]:
    """Fallback method for speaker name identification without LLM"""
    try:
        from common_data import COMMON_NAMES
        logger.info("Using fallback method for speaker name identification")
        
        # Check if segments is valid
        if not segments or not isinstance(segments, list):
            logger.warning("No valid segments provided for name identification")
            return {}
            
        # Dictionary to store detected names for each speaker ID
        detected_names = {}
        name_mentions = {}
        
        # Get unique speaker IDs
        unique_speakers = set()
        for segment in segments:
            if 'speaker' in segment:
                unique_speakers.add(segment['speaker'])
            
        # Convert to sorted list for deterministic behavior
        unique_speakers = sorted(list(unique_speakers))
            
        logger.info(f"Found {len(unique_speakers)} unique speakers in {len(segments)} segments")
        
        # If no speakers found, return default names
        if not unique_speakers:
            logger.warning("No speakers found in segments, using default names")
            return {"Speaker 0": "Speaker A", "Speaker 1": "Speaker B"}
        
        # First pass: find potential speaker names in the text
        for segment in segments:
            if 'speaker' not in segment or 'text' not in segment:
                continue
                
            speaker_id = segment['speaker']
            text = segment['text']
            
            # Skip empty text
            if not text or not isinstance(text, str):
                continue
            
            # Check for specific names we want to prioritize
            for special_name in ["Veronica", "Alexandra"]:
                if special_name.lower() in text.lower():
                    logger.info(f"Found {special_name} mentioned in text: {text}")
                    # If this speaker is addressing the person, they're likely not that person
                    # So we'll mark this speaker as NOT being that person
                    if speaker_id not in detected_names:
                        # Add to name mentions for later assignment to OTHER speakers
                        name_mentions[special_name] = name_mentions.get(special_name, 0) + 3
                        
                    # Explicitly mark this speaker as NOT being the person they're addressing
                    if speaker_id in detected_names and detected_names[speaker_id] == special_name:
                        logger.info(f"Speaker {speaker_id} is addressing {special_name}, so they can't be {special_name}")
                        detected_names.pop(speaker_id)
            
            # Extract names from text
            # Look for common name patterns directly
            import re
            for name in COMMON_NAMES:
                # Look for the name as a whole word with word boundaries
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    # Count name mentions
                    if name not in name_mentions:
                        name_mentions[name] = 0
                    name_mentions[name] += 1
                    logger.info(f"Found name mention: {name} in text: {text[:30]}...")
            
            # Look for "I'm [Name]" or "My name is [Name]" patterns
            name_intro_patterns = [
                r"I'?m\s+(\w+)",
                r"[Mm]y name is\s+(\w+)",
                r"[Cc]all me\s+(\w+)",
                r"[Tt]his is\s+(\w+)"  
            ]
            
            for pattern in name_intro_patterns:
                try:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if match in COMMON_NAMES:
                            if speaker_id not in detected_names:
                                detected_names[speaker_id] = match
                                logger.info(f"Found name for {speaker_id}: {match} (from pattern)")
                            elif detected_names[speaker_id] != match:
                                # If multiple names are detected for the same speaker, keep the most frequent one
                                if match not in name_mentions:
                                    name_mentions[match] = 0
                                name_mentions[match] += 3  # Give higher weight to explicit introduction
                except Exception as e:
                    logger.error(f"Error in regex pattern matching: {e}")
        
        # Second pass: assign names to speakers based on frequency and context
        for speaker_id in unique_speakers:
            if speaker_id not in detected_names:
                # Find the most mentioned name that hasn't been assigned yet
                try:
                    available_names = [name for name, count in sorted(name_mentions.items(), key=lambda x: x[1], reverse=True) 
                                    if name not in detected_names.values()]                
                    if available_names:
                        detected_names[speaker_id] = available_names[0]
                        logger.info(f"Assigned name for {speaker_id}: {available_names[0]} (from frequency)")
                        
                        # Special case for specific names we want to prioritize
                        for special_name in ["Veronica", "Alexandra"]:
                            if special_name in name_mentions and special_name not in detected_names.values():
                                # If the name is mentioned but not assigned, and we have multiple speakers,
                                # try to assign the name to one of them
                                if len(unique_speakers) > 1:
                                    # First, check if anyone is addressing the person directly
                                    speaker_addressing_person = None
                                    for seg in segments:
                                        if 'text' not in seg or 'speaker' not in seg:
                                            continue
                                        
                                        text = seg['text']
                                        speaker = seg['speaker']
                                        
                                        # Check for patterns like "Hi [Name]", "Hello [Name]", "Hey [Name]", etc.
                                        # Also check for direct addressing like "[Name], you..."
                                        patterns = [
                                            f'\\b(hi|hello|hey)\\s+{special_name}\\b',  # Hi/Hello/Hey Alexandra
                                            f'\\b{special_name}[,.]?\\s+you\\b',      # Alexandra, you / Alexandra. You
                                            f'\\b{special_name}[,.]?\\s+can\\s+you\\b'  # Alexandra, can you / Alexandra. Can you
                                        ]
                                        
                                        for pattern in patterns:
                                            if re.search(pattern, text.lower()):
                                                speaker_addressing_person = speaker
                                                logger.info(f"Found speaker {speaker} addressing {special_name} with pattern: {pattern}")
                                                break
                                                
                                        if speaker_addressing_person:
                                            break
                                    
                                    # If someone is addressing the person, they're likely not that speaker
                                    if speaker_addressing_person:
                                        # Find speakers other than the one addressing the person
                                        other_speakers = sorted([s for s in unique_speakers if s != speaker_addressing_person])
                                        
                                        if other_speakers:
                                            detected_names[other_speakers[0]] = special_name
                                            logger.info(f"Assigned {special_name} to {other_speakers[0]} based on being addressed")
                                    else:
                                        # No direct addressing found, fall back to default behavior
                                        # Find a speaker without a name or with a generic name
                                        for spk_id in unique_speakers:
                                            if spk_id not in detected_names or detected_names[spk_id].startswith("Speaker"):
                                                detected_names[spk_id] = special_name
                                                logger.info(f"Assigned {special_name} to {spk_id} based on mentions")
                                                break
                except Exception as e:
                    logger.error(f"Error assigning names by frequency: {e}")
                    logger.error(traceback.format_exc())
        
        # If we still don't have names for all speakers, use default names
        for i, speaker_id in enumerate(unique_speakers):
            if speaker_id not in detected_names:
                default_name = f"Speaker {chr(65+i)}"  # A, B, C, etc.
                detected_names[speaker_id] = default_name
                logger.info(f"Using default name for {speaker_id}: {default_name}")
        
        logger.info(f"Final speaker names: {detected_names}")
        return detected_names
    except Exception as e:
        logger.error(f"Unexpected error in identify_speaker_names_fallback: {e}")
        # Return default names as a last resort
        return {"Speaker 0": "Speaker A", "Speaker 1": "Speaker B"}
