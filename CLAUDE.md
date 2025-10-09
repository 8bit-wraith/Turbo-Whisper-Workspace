# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vocalis is an advanced audio processing system built by 8b.is that provides ultra-fast Whisper V3 Turbo transcription, speaker diarization, audio analysis, and security monitoring capabilities. The project serves audio via FastAPI and Gradio interfaces.

## Key Architecture Components

### Core Audio Pipeline (`vocalis/core/audio_pipeline.py`)
The AudioProcessingPipeline class orchestrates the entire audio workflow:
- Manages GPU resources efficiently with model caching
- Handles transcription via Whisper V3 Turbo
- Performs speaker diarization using pyannote.audio and sherpa-onnx
- Implements caching system to avoid redundant processing
- Integrates with LLM for conversation summarization and enhancements

### Marine-Enhanced Pipeline (`vocalis/core/marine_enhanced_pipeline.py`) ⭐ NEW
Marine-Sense integration adds advanced audio intelligence:
- **Marine VAD**: O(1) voice activity detection (removes silence before transcription)
- **Sound Localization**: Multi-channel TDOA triangulation for 3D source tracking
- **Salience Scoring**: Identifies "consciousness moments" (most important audio segments)
- **Emotional Analysis**: Ultrasonic frequency analysis (requires 192kHz audio)
- See `MARINE_VAD_LOCALIZATION.md` for full documentation

### Security Monitoring System
- `vocalis/security/security_monitor.py` - General security incident detection
- `vocalis/security/bar_security_monitor.py` - Specialized monitoring for bar/venue environments
- Threat level assessment (1-5 scale) with incident reporting

### Application Entry Points
- `app.py` - Main Gradio-based UI application
- `app_api.py` - FastAPI server for programmatic access
- `app_vocalis.py` - Alternative Vocalis-branded interface

## Development Commands

### Virtual Environment & Dependencies
```bash
# Initial setup - creates venv and installs all dependencies
./scripts/manage.sh setup

# Update all dependencies to latest versions
./scripts/manage.sh update

# Download speaker embedding models locally
./scripts/manage.sh models
```

### Running the Application
```bash
# Start standard Vocalis application
./scripts/manage.sh start

# Start Trisha's Audio Lab (enhanced features demo)
./scripts/manage.sh lab

# Demo Marine-Sense features (VAD, localization, salience)
python demo_marine_features.py

# Stop running application
./scripts/manage.sh stop

# Restart application
./scripts/manage.sh restart
```

### API Server
```bash
# Run FastAPI server on port 8000
python -m vocalis api --port 8000

# Run with default port (8420 per company standard)
python app_api.py
```

### Testing
```bash
# Run all tests with unittest
./scripts/manage.sh test
python -m unittest discover -s tests

# Test Marine features (VAD, localization, salience)
python tests/test_marine_features.py

# Test GPU functionality
./scripts/manage.sh gpu

# Test audio enhancement features
./scripts/manage.sh test-lab

# Run specific test file
python tests/test_cache.py
```

### Maintenance
```bash
# Clean Python cache files and temp files
./scripts/manage.sh clean

# Run security monitoring on audio files
python -m vocalis security --input audio.flac --threat-level 2

# Monitor bar directory for security incidents
./scripts/bar_monitor.sh /path/to/audio/dir
```

## Important Technical Details

### GPU Optimization
- The pipeline automatically detects and configures GPU (CUDA) when available
- Uses TF32 precision on Ampere GPUs for faster processing
- Implements model caching to avoid redundant GPU memory allocation
- Falls back to CPU gracefully when GPU unavailable

### Caching System
- Results cached in `.tw_cache` directory (configurable via TW_CACHE_DIR env var)
- Cache keys based on audio file hash and processing parameters
- Use `force_reprocess=True` to bypass cache
- Cache improves performance significantly for repeated processing

### Model Management
- Models are loaded lazily on first use
- Shared model instances across pipeline to minimize memory usage
- Speaker embedding models stored in `speaker_models/` directory

### Audio Format Support
- Primary format: .m4a (optimized in latest update)
- Also supports: .wav, .flac, .mp3, .ogg
- Audio normalization and enhancement features available

## Testing Approach

The project uses Python's unittest framework. Key testing patterns:
- Mock heavy dependencies (torch, models) for unit tests
- Create minimal dummy audio files for testing
- Test cache roundtrip functionality
- Verify GPU detection and configuration

Example test execution:
```python
# Run cache test specifically
python -m pytest tests/test_cache.py::test_pipeline_cache_roundtrip
```

## Company Standards

- API endpoints use port 8420 (Mem|8 API standard)
- Port 8422 for Cheet API
- Port 8424 for internal websites/dev
- Port 8428 for LLM endpoints
- Prefer Rust for new development when possible
- Use pnpm for Node.js projects, uv for Python