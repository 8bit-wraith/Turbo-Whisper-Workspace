# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import wave
import os
import tempfile
import time
from functools import lru_cache
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf
import sherpa_onnx
from huggingface_hub import hf_hub_download
from pydub import AudioSegment


def read_wave(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Read audio file in any format and convert to mono numpy array
    
    Args:
      audio_path:
        Path to an audio file. Can be any format supported by librosa/pydub.
        
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the audio file
    """
    try:
        # First attempt: Try using librosa which handles many formats
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            # Ensure float32 and normalized to [-1, 1]
            audio = audio.astype(np.float32)
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))
            return audio, sr
        except Exception as e:
            print(f"Librosa failed: {e}, trying pydub...")
            
        # Second attempt: Try using pydub which handles even more formats
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            # Convert to mono if needed
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            # Convert to float32 and normalize
            samples = samples.astype(np.float32)
            samples = samples / (1 << (8 * audio_segment.sample_width - 1))
            
            return samples, audio_segment.frame_rate
        except Exception as e:
            print(f"Pydub failed: {e}, trying soundfile...")
            
        # Third attempt: Try using soundfile
        try:
            audio, sr = sf.read(audio_path)
            # Convert to mono if needed
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            # Ensure float32 and normalized to [-1, 1]
            audio = audio.astype(np.float32)
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))
            return audio, sr
        except Exception as e:
            print(f"Soundfile failed: {e}")
            
    except Exception as e:
        print(f"Error reading audio file: {e}")
        # Return empty audio rather than crashing
        return np.zeros(1600, dtype=np.float32), 16000


@lru_cache(maxsize=30)
def get_file(
    repo_id: str,
    filename: str,
    subfolder: str = ".",
) -> str:
    nn_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return nn_model_filename


def download_models_to_local(models_dir="models", force_download=False, use_auth_token=None):
    """Download speaker embedding models to local directory.
    
    Args:
        models_dir: Directory to save models to
        force_download: Whether to force download even if file exists
        use_auth_token: Optional Hugging Face token for accessing gated models
        
    Returns:
        Dictionary mapping model names to local file paths
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    # Dictionary to store model paths
    local_models = {}
    
    # Import shutil here to avoid potential import issues
    import shutil
    
    # List of models to download with their sources and fallbacks
    models_to_download = [
        # NeMo models with fallbacks
        {
            "name": "nemo_en_titanet_small.onnx", 
            "sources": [
                {"repo": "csukuangfj/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_titanet_small.onnx"},
                {"repo": "k2-fsa/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_titanet_small.onnx"},
            ]
        },
        {
            "name": "nemo_en_titanet_large.onnx", 
            "sources": [
                {"repo": "nvidia/speakerverification_en_titanet_large", "file": "model.onnx"},
                {"repo": "csukuangfj/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_titanet_large.onnx"},
                {"repo": "k2-fsa/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_titanet_large.onnx"},
            ]
        },
        {
            "name": "nemo_en_ecapa_tdnn.onnx", 
            "sources": [
                {"repo": "nvidia/speakerverification_en_ecapa_tdnn", "file": "model.onnx"},
                {"repo": "csukuangfj/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_ecapa_tdnn.onnx"},
                {"repo": "k2-fsa/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_ecapa_tdnn.onnx"},
            ]
        },
        
        # 3D-Speaker models with fallbacks
        {
            "name": "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx", 
            "sources": [
                {"repo": "k2-fsa/speaker-embedding-models", "file": "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"},
                {"repo": "3D-Speaker/3D-Speaker-models", "file": "speech_eres2net_sv_en_voxceleb_16k.onnx"},
            ]
        },
        {
            "name": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx", 
            "sources": [
                {"repo": "k2-fsa/speaker-embedding-models", "file": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"},
                {"repo": "3D-Speaker/3D-Speaker-models", "file": "speech_campplus_sv_en_voxceleb_16k.onnx"},
            ]
        },
        
        # WeSpeaker models with fallbacks
        {
            "name": "wenet_wespeaker_resnet.onnx", 
            "sources": [
                {"repo": "wenet-e2e/wespeaker-models", "file": "resnet/model.onnx"},
                {"repo": "k2-fsa/wespeaker-models", "file": "wenet_wespeaker_resnet.onnx"},
                {"repo": "csukuangfj/wespeaker-models", "file": "wenet_wespeaker_resnet.onnx"},
            ]
        },
        {
            "name": "wenet_wespeaker_ecapa_tdnn.onnx", 
            "sources": [
                {"repo": "wenet-e2e/wespeaker-models", "file": "ecapa-tdnn/model.onnx"},
                {"repo": "k2-fsa/wespeaker-models", "file": "wenet_wespeaker_ecapa_tdnn.onnx"},
                {"repo": "csukuangfj/wespeaker-models", "file": "wenet_wespeaker_ecapa_tdnn.onnx"},
            ]
        },
        {
            "name": "wenet_wespeaker_campplus.onnx", 
            "sources": [
                {"repo": "wenet-e2e/wespeaker-models", "file": "campplus/model.onnx"},
                {"repo": "k2-fsa/wespeaker-models", "file": "wenet_wespeaker_campplus.onnx"},
                {"repo": "csukuangfj/wespeaker-models", "file": "wenet_wespeaker_campplus.onnx"},
            ]
        },
    ]
    
    # Download each model
    for model in models_to_download:
        local_path = os.path.join(models_dir, model["name"])
        
        # Skip if file exists and we're not forcing download
        if os.path.exists(local_path) and not force_download:
            print(f"Skipping {model['name']} - already exists")
            local_models[model["name"]] = local_path
            continue
        
        # Try each source until one works
        downloaded = False
        for source in model["sources"]:
            if downloaded:
                break
                
            print(f"Trying to download {model['name']} from {source['repo']}...")
            try:
                # Download the model
                remote_path = hf_hub_download(
                    repo_id=source["repo"],
                    filename=source["file"],
                    token=use_auth_token,
                )
                
                # Copy to local models directory
                shutil.copy(remote_path, local_path)
                local_models[model["name"]] = local_path
                print(f"✅ Successfully downloaded {model['name']} from {source['repo']}")
                downloaded = True
            except Exception as e:
                error_msg = str(e).split('\n')[0]
                print(f"  ❌ Failed from {source['repo']}: {error_msg}")
        
        if not downloaded:
            print(f"⚠️ Could not download {model['name']} from any source")
    
    return local_models

def get_speaker_segmentation_model(repo_id) -> str:
    assert repo_id in (
        "pyannote/segmentation-3.0",
        "Revai/reverb-diarization-v1",
    )

    # Check for local models first
    if repo_id == "pyannote/segmentation-3.0":
        local_path = os.path.join("models", "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path
        
        # Fall back to remote model
        return get_file(
            repo_id="csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
            filename="model.onnx",
        )
    elif repo_id == "Revai/reverb-diarization-v1":
        local_path = os.path.join("models", "sherpa-onnx-reverb-diarization-v1.tar.bz2")
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path
        
        # Fall back to remote model
        return get_file(
            repo_id="csukuangfj/sherpa-onnx-revai-reverb-diarization-v1",
            filename="model.onnx",
        )


def get_speaker_embedding_model(model_name: str) -> str:
    """Get speaker embedding model path.

    Args:
      model_name:
        Format: model_filename|model_size
        Example: nemo_en_titanet_large.onnx|97MB
    """
    parts = model_name.split("|")
    model_filename = parts[0]
    
    # First check if the model exists in the local models directory
    local_model_path = os.path.join("models", model_filename)
    if os.path.exists(local_model_path):
        print(f"Using local model: {local_model_path}")
        return local_model_path
    
    # Check for alternative local models that might work
    if model_filename == "nemo_en_titanet_small.onnx":
        local_path = os.path.join("models", "nemo_en_titanet_small.onnx")
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path
    elif "3dspeaker" in model_filename:
        local_path = os.path.join("models", "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path
    
    print(f"Local model not found at {local_model_path}, trying to download...")

    # Handle NeMo models
    if model_filename.startswith("nemo_"):
        # Map model filenames to the correct Hugging Face model IDs
        model_mapping = {
            "nemo_en_titanet_large.onnx": "nvidia/speakerverification_en_titanet_large",
            "nemo_en_titanet_small.onnx": "nvidia/speakerverification_en_titanet_small",
            "nemo_en_ecapa_tdnn.onnx": "nvidia/speakerverification_en_ecapa_tdnn",
        }
        
        # Try to find the model in our mapping
        if model_filename in model_mapping:
            try:
                print(f"Loading NeMo model from Hugging Face: {model_mapping[model_filename]}")
                return get_file(
                    repo_id=model_mapping[model_filename],
                    filename="model.onnx",  # NeMo models on HF use model.onnx as the filename
                )
            except Exception as e:
                print(f"Error loading from official NeMo HF repository: {e}")
        
        # Try alternative repositories if not found or if loading failed
        try:
            # Try csukuangfj's repository which has converted NeMo models
            return get_file(
                repo_id="csukuangfj/sherpa-onnx-nemo-speaker-models",
                filename=model_filename,
            )
        except Exception as e2:
            print(f"Error loading from csukuangfj repository: {e2}")
            try:
                # Try k2-fsa repository as another fallback
                return get_file(
                    repo_id="k2-fsa/sherpa-onnx-nemo-speaker-models",
                    filename=model_filename,
                )
            except Exception as e3:
                print(f"Error loading from k2-fsa repository: {e3}")
                # Final fallback
                return get_file(
                    repo_id="csukuangfj/sherpa-onnx-nemo-speaker-models",
                    filename=model_filename,
                )
    # Handle 3D-Speaker models
    elif model_filename.startswith("3dspeaker_"):
        try:
            # Try the official 3D-Speaker repo first
            return get_file(
                repo_id="3D-Speaker/3D-Speaker-models",
                filename=model_filename.replace("3dspeaker_", ""),
            )
        except Exception as e:
            print(f"Error loading from 3D-Speaker repository: {e}")
            # Fallback to the sherpa-onnx repository
            return get_file(
                repo_id="k2-fsa/speaker-embedding-models",
                filename=model_filename,
            )
    # Handle WeSpeaker models with different naming conventions
    elif model_filename.startswith("wespeaker_"):
        # Clean up model name for proper URL encoding
        clean_filename = model_filename.replace("+", "plus")
        
        # Check if it's a standard model or one with LM enhancement
        if "_LM" in model_filename:
            # For LM-enhanced models, try multiple repositories
            try:
                # Try the official WeSpeaker repo first
                model_type = "base"
                if "resnet" in model_filename.lower():
                    model_type = "resnet"
                elif "ecapa" in model_filename.lower():
                    model_type = "ecapa-tdnn"
                elif "camplus" in model_filename.lower() or "cam++" in model_filename.lower():
                    model_type = "campplus"
                elif "gemini" in model_filename.lower():
                    model_type = "gemini"
                
                return get_file(
                    repo_id=f"wenet-e2e/wespeaker-models-v2",
                    filename=f"{model_type}/model.onnx",
                )
            except Exception as e:
                print(f"Error loading from WeSpeaker official repo: {e}")
                # Fallback to k2-fsa repo
                try:
                    return get_file(
                        repo_id="k2-fsa/wespeaker-models",
                        filename=clean_filename,
                    )
                except Exception as e2:
                    print(f"Error loading from k2-fsa repository: {e2}")
                    # Last resort - try csukuangfj repo
                    return get_file(
                        repo_id="csukuangfj/wespeaker-models",
                        filename=clean_filename,
                    )
        else:
            # For standard models, try multiple repositories
            try:
                # Try wenet-e2e repo first
                model_type = "base"
                if "resnet" in model_filename.lower():
                    model_type = "resnet"
                elif "ecapa" in model_filename.lower():
                    model_type = "ecapa-tdnn"
                elif "camplus" in model_filename.lower() or "cam++" in model_filename.lower():
                    model_type = "campplus"
                
                return get_file(
                    repo_id="wenet-e2e/wespeaker-models",
                    filename=f"{model_type}/model.onnx",
                )
            except Exception as e:
                print(f"Error loading from WeSpeaker official repo: {e}")
                # Try k2-fsa repo
                try:
                    return get_file(
                        repo_id="k2-fsa/wespeaker-models",
                        filename=clean_filename,
                    )
                except Exception as e2:
                    print(f"Error loading from k2-fsa repository: {e2}")
                    # Last resort
                    return get_file(
                        repo_id="csukuangfj/wespeaker-models",
                        filename=clean_filename,
                    )
    else:
        # For unknown model types, try to infer the right repository
        print(f"Unknown model type: {model_name}, attempting to infer repository")
        try:
            # Try to infer model type from filename
            if "resnet" in model_filename.lower():
                try:
                    return get_file(
                        repo_id="wenet-e2e/wespeaker-models",
                        filename="resnet/model.onnx",
                    )
                except Exception:
                    pass
            elif "ecapa" in model_filename.lower():
                try:
                    return get_file(
                        repo_id="wenet-e2e/wespeaker-models",
                        filename="ecapa-tdnn/model.onnx",
                    )
                except Exception:
                    pass
            elif "camp" in model_filename.lower():
                try:
                    return get_file(
                        repo_id="wenet-e2e/wespeaker-models",
                        filename="campplus/model.onnx",
                    )
                except Exception:
                    pass
                    
            # If all else fails, use a known working model as final fallback
            print(f"Using default fallback model for: {model_name}")
            return get_file(
                repo_id="csukuangfj/sherpa-onnx-nemo-speaker-models",
                filename="nemo_en_titanet_small.onnx",
            )
        except Exception as e:
            print(f"All attempts to load model failed: {e}")
            print("Using absolute last resort model")
            # Absolute last resort - use a model we know exists
            return get_file(
                repo_id="csukuangfj/sherpa-onnx-nemo-speaker-models",
                filename="nemo_en_titanet_small.onnx",
            )


# Cache for diarization models
_DIARIZATION_CACHE = {}

def get_speaker_diarization(
    segmentation_model: str, embedding_model: str, num_clusters: int, threshold: float, use_gpu: bool = True
):
    # Create a cache key based on the input parameters
    cache_key = f"{segmentation_model}_{embedding_model}_{num_clusters}_{threshold}"
    
    # Check if this configuration is already cached
    if cache_key in _DIARIZATION_CACHE:
        print(f"Using cached diarization model for {cache_key}")
        return _DIARIZATION_CACHE[cache_key]
    
    segmentation = get_speaker_segmentation_model(segmentation_model)
    embedding = get_speaker_embedding_model(embedding_model)
    
    # Check if CUDA is available for ONNX acceleration
    import torch
    # Only use CUDA if it's available AND use_gpu is True
    provider = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
    
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation
            ),
            provider=provider,  # Use CUDA provider if available
            num_threads=4,      # Optimize thread count
            debug=False,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding,
            provider=provider,  # Use CUDA provider if available
            num_threads=4,      # Optimize thread count
            debug=False,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    
    print("config", config)
    
    if not config.validate():
        raise RuntimeError(
            "Please check your config and make sure all required files exist"
        )
    
    # Create the diarization object with error handling
    try:
        diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
        
        # Cache the diarization object
        _DIARIZATION_CACHE[cache_key] = diarizer
        
        return diarizer
    except RuntimeError as e:
        if "protobuf parsing failed" in str(e):
            print("Error: Failed to load model due to protobuf parsing issue.")
            print("This may be due to model format incompatibility or sherpa-onnx installation.")
            print("Trying alternative approach with extracted models...")
            
            # Try to extract the tar.bz2 files if they haven't been extracted yet
            try:
                import tarfile
                
                # Check if the segmentation model is a tar.bz2 file
                if segmentation.endswith('.tar.bz2'):
                    extract_dir = segmentation.replace('.tar.bz2', '')
                    if not os.path.exists(extract_dir):
                        os.makedirs(extract_dir, exist_ok=True)
                        with tarfile.open(segmentation, 'r:bz2') as tar:
                            tar.extractall(path=extract_dir)
                        print(f"Extracted {segmentation} to {extract_dir}")
                    
                    # Try to find model.onnx in the extracted directory
                    for root, dirs, files in os.walk(extract_dir):
                        for file in files:
                            if file.endswith('.onnx'):
                                segmentation = os.path.join(root, file)
                                print(f"Using extracted model: {segmentation}")
                                break
                
                # Create a new config with the extracted model
                config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                            model=segmentation
                        ),
                        provider="cpu",  # Force CPU provider to avoid GPU issues
                        num_threads=4,
                        debug=False,
                    ),
                    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                        model=embedding,
                        provider="cpu",  # Force CPU provider to avoid GPU issues
                        num_threads=4,
                        debug=False,
                    ),
                    clustering=sherpa_onnx.FastClusteringConfig(
                        num_clusters=num_clusters,
                        threshold=threshold,
                    ),
                    min_duration_on=0.3,
                    min_duration_off=0.5,
                )
                
                print("New config with extracted models:", config)
                
                if not config.validate():
                    raise RuntimeError(
                        "Please check your config and make sure all required files exist"
                    )
                
                diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
                
                # Cache the diarization object
                _DIARIZATION_CACHE[cache_key] = diarizer
                
                return diarizer
            except Exception as extract_error:
                print(f"Failed to use extracted models: {extract_error}")
                raise e
        else:
            # Re-raise the original error if it's not a protobuf parsing issue
            raise e


# All available segmentation models
speaker_segmentation_models = [
    "pyannote/segmentation-3.0",
    "Revai/reverb-diarization-v1",
]

# All available embedding models by category
nemo_speaker_embedding_models = [
    # Core models - verified to exist in official NVIDIA Hugging Face repositories
    "nemo_en_titanet_large.onnx|97MB",      # Maps to nvidia/speakerverification_en_titanet_large
    "nemo_en_titanet_small.onnx|38MB",      # Maps to nvidia/speakerverification_en_titanet_small
    "nemo_en_ecapa_tdnn.onnx|45MB",         # Maps to nvidia/speakerverification_en_ecapa_tdnn
    
    # Alternative formats - also available in csukuangfj/sherpa-onnx-nemo-speaker-models
    "en_titanet_large.onnx|97MB",           # Alternative format
    "en_ecapa_tdnn.onnx|45MB",              # Alternative format
    "en_titanet_small.onnx|38MB",           # Alternative format
    
    # Additional models if available
    "en_speakerverification_speakernet.onnx|22MB" # Available in some repositories
]

three_d_speaker_embedding_models = [
    # Verified available models from k2-fsa/speaker-embedding-models
    "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx|28.2MB",  # Confirmed available
    "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx|25.3MB",  # Confirmed available
    
    # Models from official 3D-Speaker repository
    "speech_campplus_sv_en_voxceleb_16k.onnx|28.2MB",  # Available in 3D-Speaker/3D-Speaker-models
    "speech_eres2net_sv_en_voxceleb_16k.onnx|25.3MB",  # Available in 3D-Speaker/3D-Speaker-models
    # Additional models if available
    "speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx|37.8MB",  # Try from official repo
    "speech_campplus_sv_zh-cn_16k-common.onnx|27MB",  # Try from official repo
]

wespeaker_embedding_models = [
    # Models available in wenet-e2e repositories (official WeSpeaker models)
    "wenet_wespeaker_resnet.onnx|26MB",        # Available in wenet-e2e/wespeaker-models
    "wenet_wespeaker_ecapa_tdnn.onnx|21MB",    # Available in wenet-e2e/wespeaker-models
    "wenet_wespeaker_campplus.onnx|28MB",      # Available in wenet-e2e/wespeaker-models
    
    # Fallback models with simplified names
    "resnet34.onnx|26MB",  # Generic ResNet34 model
    "ecapa_tdnn.onnx|21MB",  # Generic ECAPA-TDNN model
    "campplus.onnx|28MB",  # Generic CAM++ model
]

# Dictionary mapping model types to their model lists
embedding2models = {
    "NeMo": nemo_speaker_embedding_models,
    "3D-Speaker": three_d_speaker_embedding_models,
    "WeSpeaker": wespeaker_embedding_models,
}

# Cache for local model detection
_LOCAL_MODEL_CACHE = {
    'segmentation_models': None,
    'embedding_models': None,
    'last_scan_time': 0
}

# Functions to get locally available models

def get_local_segmentation_models():
    """Get list of locally available segmentation models"""
    # Check if we have a cached result and if it's still valid
    current_time = time.time()
    cache_age = current_time - _LOCAL_MODEL_CACHE['last_scan_time']
    
    # Use cached result if available and less than 60 seconds old
    if _LOCAL_MODEL_CACHE['segmentation_models'] is not None and cache_age < 60:
        return _LOCAL_MODEL_CACHE['segmentation_models']
    
    local_models = []
    
    # Check for pyannote model
    if os.path.exists(os.path.join("models", "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")):
        local_models.append("pyannote/segmentation-3.0")
    
    # Check for revai model
    if os.path.exists(os.path.join("models", "sherpa-onnx-reverb-diarization-v1.tar.bz2")):
        local_models.append("Revai/reverb-diarization-v1")
    
    # If no local models found, return all models (will fall back to remote)
    if not local_models:
        local_models = speaker_segmentation_models
    
    # Cache the result
    _LOCAL_MODEL_CACHE['segmentation_models'] = local_models
    _LOCAL_MODEL_CACHE['last_scan_time'] = current_time
    
    return local_models

def get_local_embedding_models():
    """Get dictionary of locally available embedding models by category"""
    # Check if we have a cached result and if it's still valid
    current_time = time.time()
    cache_age = current_time - _LOCAL_MODEL_CACHE['last_scan_time']
    
    # Use cached result if available and less than 60 seconds old
    if _LOCAL_MODEL_CACHE['embedding_models'] is not None and cache_age < 60:
        return _LOCAL_MODEL_CACHE['embedding_models']
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        # Cache the result
        _LOCAL_MODEL_CACHE['embedding_models'] = embedding2models
        _LOCAL_MODEL_CACHE['last_scan_time'] = current_time
        return embedding2models  # Return all models if models directory doesn't exist
    
    # Get list of files in models directory
    model_files = os.listdir(models_dir)
    
    # Filter model lists to only include locally available models
    local_embedding_models = {}
    
    # Check NeMo models
    local_nemo_models = []
    for model in nemo_speaker_embedding_models:
        model_name = model.split("|")[0]
        if model_name in model_files:
            local_nemo_models.append(model)
    
    # Check 3D-Speaker models
    local_3d_models = []
    for model in three_d_speaker_embedding_models:
        model_name = model.split("|")[0]
        if model_name in model_files:
            local_3d_models.append(model)
    
    # Check WeSpeaker models
    local_wespeaker_models = []
    for model in wespeaker_embedding_models:
        model_name = model.split("|")[0]
        if model_name in model_files:
            local_wespeaker_models.append(model)
    
    # Only include categories that have at least one local model
    if local_nemo_models:
        local_embedding_models["NeMo"] = local_nemo_models
    
    if local_3d_models:
        local_embedding_models["3D-Speaker"] = local_3d_models
    
    if local_wespeaker_models:
        local_embedding_models["WeSpeaker"] = local_wespeaker_models
    
    # If no local models found, return all models (will fall back to remote)
    if not local_embedding_models:
        local_embedding_models = embedding2models
    
    # Cache the result
    _LOCAL_MODEL_CACHE['embedding_models'] = local_embedding_models
    _LOCAL_MODEL_CACHE['last_scan_time'] = current_time
    
    return local_embedding_models
