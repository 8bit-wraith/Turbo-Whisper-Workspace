#!/bin/bash

# Colors for cyberpunk terminal aesthetics
GREEN='\033[38;5;46m'
CYAN='\033[38;5;51m'
MAGENTA='\033[38;5;201m'
YELLOW='\033[38;5;226m'
RED='\033[38;5;196m'
RESET='\033[0m'
BOLD='\033[1m'

# ASCII Art Banner
show_banner() {
    echo -e "${GREEN}${BOLD}"
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  ${CYAN}▀█▀ █ █ █▀█ █▄▄ █▀█   █▀█ █ █ █▀▄ █ █▀█${GREEN}  ║"
    echo "║  ${CYAN} █  █▄█ █▀▄ █▄█ █▄█   █▀█ █▄█ █▄▀ █ █▄█${GREEN}  ║"
    echo "║                                              ║"
    echo "║  ${MAGENTA}W O R K S P A C E   M A N A G E R  v2.0${GREEN}    ║"
    echo "║  ${YELLOW}🧪 Now featuring Trisha's Audio Lab! 🧪${GREEN}    ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo -e "${RESET}"
}

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}${BOLD}[!] Virtual environment not found. Creating...${RESET}"
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}${BOLD}[✓] Virtual environment created at ${VENV_DIR}${RESET}"
    else
        echo -e "${GREEN}${BOLD}[✓] Virtual environment found at ${VENV_DIR}${RESET}"
    fi
}

# Function to activate virtual environment
activate_venv() {
    source "${VENV_DIR}/bin/activate"
    echo -e "${GREEN}${BOLD}[✓] Virtual environment activated${RESET}"
}

# Install dependencies
install_deps() {
    echo -e "${CYAN}${BOLD}[*] Installing dependencies...${RESET}"
    
    # Check if pip is available
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}${BOLD}[!] pip not found. Installing pip...${RESET}"
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python get-pip.py
        rm get-pip.py
    fi
    
    # Install dependencies with progress
    echo -e "${CYAN}${BOLD}[*] Installing requirements from ${PROJECT_ROOT}/requirements.txt${RESET}"
    pip install -r "${PROJECT_ROOT}/requirements.txt"
    
    # Special handling for potential problematic packages
    echo -e "${CYAN}${BOLD}[*] Checking special dependencies...${RESET}"
    
    # Ensure python-dotenv is installed
    pip install python-dotenv
    
    # Ensure soundfile is installed
    pip install soundfile
    
    # Ensure scikit-learn is installed
    pip install scikit-learn
    
    echo -e "${GREEN}${BOLD}[✓] All dependencies installed${RESET}"
}

# Start the application
start_app() {
    echo -e "${CYAN}${BOLD}[*] Starting CyberVox Audio Workspace...${RESET}"
    
    # Check if .env file exists, create if not
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        echo -e "${YELLOW}${BOLD}[!] .env file not found. Creating...${RESET}"
        echo "# CyberVox Audio Workspace Environment Variables" > "${PROJECT_ROOT}/.env"
        echo "HF_TOKEN=" >> "${PROJECT_ROOT}/.env"
        echo -e "${GREEN}${BOLD}[✓] .env file created${RESET}"
    fi
    
    cd "$PROJECT_ROOT"
    python app.py
}

# Start the application with Trisha's Lab demo
start_lab() {
    echo -e "${MAGENTA}${BOLD}[*] 🧪⚡ Starting Trisha's Audio Lab Demo! ⚡🧪${RESET}"
    echo -e "${YELLOW}${BOLD}[*] This will launch CyberVox with enhanced audio processing features!${RESET}"
    echo -e "${CYAN}${BOLD}[*] Look for the '🧪 Trisha's Audio Lab' tab in the interface!${RESET}"
    
    # Check if .env file exists, create if not
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        echo -e "${YELLOW}${BOLD}[!] .env file not found. Creating...${RESET}"
        echo "# CyberVox Audio Workspace Environment Variables" > "${PROJECT_ROOT}/.env"
        echo "HF_TOKEN=" >> "${PROJECT_ROOT}/.env"
        echo -e "${GREEN}${BOLD}[✓] .env file created${RESET}"
    fi
    
    cd "$PROJECT_ROOT"
    
    # Check if the demo script exists
    if [ -f "${PROJECT_ROOT}/demo_trishas_lab.py" ]; then
        echo -e "${GREEN}${BOLD}[✓] Found Trisha's Lab demo script${RESET}"
        python demo_trishas_lab.py
    else
        echo -e "${YELLOW}${BOLD}[!] Demo script not found, starting regular app...${RESET}"
        python app.py
    fi
}

# Test audio enhancement functionality
test_lab() {
    echo -e "${MAGENTA}${BOLD}[*] 🧪 Testing Trisha's Audio Lab functionality...${RESET}"
    
    cd "$PROJECT_ROOT"
    
    # Create a simple test script for audio enhancement
    if [ ! -f "${PROJECT_ROOT}/test_audio_lab.py" ]; then
        echo -e "${YELLOW}${BOLD}[!] Creating audio lab test script...${RESET}"
        cat > "${PROJECT_ROOT}/test_audio_lab.py" << 'EOL'
#!/usr/bin/env python3
"""
🧪 Trisha's Audio Lab Test Script
Tests the audio enhancement functionality
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('lab_test')

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("🧪 Testing imports for Trisha's Audio Lab...")
    
    try:
        import librosa
        logger.info("✅ librosa imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import librosa: {e}")
        return False
    
    try:
        import noisereduce as nr
        logger.info("✅ noisereduce imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import noisereduce: {e}")
        return False
    
    try:
        import soundfile as sf
        logger.info("✅ soundfile imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import soundfile: {e}")
        return False
    
    try:
        from scipy.signal import butter, lfilter
        logger.info("✅ scipy.signal imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import scipy.signal: {e}")
        return False
    
    try:
        import gradio as gr
        logger.info("✅ gradio imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import gradio: {e}")
        return False
    
    return True

def test_audio_functions():
    """Test that audio enhancement functions work"""
    logger.info("🎧 Testing audio enhancement functions...")
    
    try:
        # Import the enhanced app functions
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app import normalize_audio, highpass_filter
        
        import numpy as np
        
        # Test normalize_audio function
        test_audio = np.array([0.1, 0.5, -0.3, 0.8, -0.9])
        normalized = normalize_audio(test_audio)
        logger.info(f"✅ normalize_audio test passed: max={np.max(np.abs(normalized)):.3f}")
        
        # Test highpass_filter function  
        test_audio_long = np.random.randn(1000)
        filtered = highpass_filter(test_audio_long, 44100, 100)
        logger.info(f"✅ highpass_filter test passed: length={len(filtered)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Audio function test failed: {e}")
        return False

def main():
    logger.info("=== Starting Trisha's Audio Lab Test ===")
    
    # Test 1: Import test
    if not test_imports():
        logger.error("❌ Import tests failed!")
        return False
    
    # Test 2: Function test
    if not test_audio_functions():
        logger.error("❌ Function tests failed!")
        return False
    
    logger.info("🎉 All tests passed! Trisha's Audio Lab is ready to rock!")
    logger.info("=== Test Complete ===")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOL
        echo -e "${GREEN}${BOLD}[✓] Audio lab test script created${RESET}"
    fi
    
    # Run the test
    python test_audio_lab.py
}

# Stop the application (find and kill the process)
stop_app() {
    echo -e "${YELLOW}${BOLD}[*] Stopping CyberVox Audio Workspace...${RESET}"
    pids=$(pgrep -f "python app.py")
    if [ -z "$pids" ]; then
        echo -e "${YELLOW}[!] No running instances found${RESET}"
    else
        echo -e "${CYAN}[*] Killing processes: $pids${RESET}"
        kill $pids
        echo -e "${GREEN}[✓] Application stopped${RESET}"
    fi
}

# Run tests
run_tests() {
    echo -e "${CYAN}${BOLD}[*] Running tests...${RESET}"
    cd "$PROJECT_ROOT"
    python -m unittest discover -s tests
}

# Test GPU functionality
test_gpu() {
    echo -e "${MAGENTA}${BOLD}[*] Testing GPU functionality...${RESET}"
    cd "$PROJECT_ROOT"
    
    # Create a simple test script if it doesn't exist
    if [ ! -f "${PROJECT_ROOT}/test_gpu.py" ]; then
        echo -e "${YELLOW}${BOLD}[!] Creating GPU test script...${RESET}"
        cat > "${PROJECT_ROOT}/test_gpu.py" << 'EOL'
#!/usr/bin/env python3
import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_test')

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the necessary functions
try:
    from llm_helper import verify_gpu_usage, monitor_gpu_usage, get_llm
    logger.info("Successfully imported GPU functions from llm_helper")
except ImportError as e:
    logger.error(f"Failed to import from llm_helper: {e}")
    sys.exit(1)

def main():
    logger.info("=== Starting GPU Test ===")
    
    # Test 1: Verify GPU availability
    logger.info("Test 1: Verifying GPU availability...")
    gpu_available = verify_gpu_usage()
    if gpu_available:
        logger.info("✅ GPU verification passed!")
    else:
        logger.error("❌ GPU verification failed!")
        return
    
    # Test 2: Monitor GPU usage
    logger.info("Test 2: Monitoring initial GPU state...")
    monitor_gpu_usage("Initial State")
    
    # Test 3: Load LLM and check GPU usage
    logger.info("Test 3: Loading LLM and checking GPU usage...")
    try:
        start_time = time.time()
        llm = get_llm()
        load_time = time.time() - start_time
        
        if llm:
            logger.info(f"✅ LLM loaded successfully in {load_time:.2f} seconds")
            monitor_gpu_usage("After LLM Load")
            
            # Test 4: Run a simple inference
            logger.info("Test 4: Running simple inference...")
            prompt = "Summarize the following in one sentence: AI models are becoming increasingly powerful."
            
            start_time = time.time()
            response = llm.create_completion(prompt, max_tokens=100, temperature=0.7, top_p=0.95)
            inference_time = time.time() - start_time
            
            logger.info(f"Inference completed in {inference_time:.2f} seconds")
            logger.info(f"Response: {response['choices'][0]['text']}")
            
            # Final GPU monitoring
            monitor_gpu_usage("After Inference")
        else:
            logger.error("❌ Failed to load LLM")
    except Exception as e:
        logger.error(f"Error during LLM testing: {e}")
    
    logger.info("=== GPU Test Complete ===")

if __name__ == "__main__":
    main()
EOL
        echo -e "${GREEN}${BOLD}[✓] GPU test script created${RESET}"
    fi
    
    # Run the GPU test
    python test_gpu.py
}

# Clean cache files
clean_cache() {
    echo -e "${CYAN}${BOLD}[*] Cleaning cache files...${RESET}"
    
    # Remove Python cache files
    find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} +
    find "${PROJECT_ROOT}" -type f -name "*.pyc" -delete
    
    # Remove temporary files
    find "${PROJECT_ROOT}" -type f -name "*.tmp" -delete
    
    echo -e "${GREEN}${BOLD}[✓] Cache files cleaned${RESET}"
}

# Update dependencies
update_deps() {
    echo -e "${CYAN}${BOLD}[*] Updating dependencies...${RESET}"
    
    # Activate virtual environment if not already active
    if [ -z "$VIRTUAL_ENV" ]; then
        activate_venv
    fi
    
    # Update pip
    pip install --upgrade pip
    
    # Update all packages
    pip install --upgrade -r "${PROJECT_ROOT}/requirements.txt"
    
    echo -e "${GREEN}${BOLD}[✓] Dependencies updated${RESET}"
}

# Download speaker embedding models
download_models() {
    echo -e "${CYAN}${BOLD}[*] Downloading speaker embedding models...${RESET}"
    
    # Check if the download script exists
    if [ ! -f "${PROJECT_ROOT}/scripts/download_models.sh" ]; then
        echo -e "${RED}${BOLD}[!] Model download script not found.${RESET}"
        exit 1
    fi
    
    # Run the download script with any passed arguments
    "${PROJECT_ROOT}/scripts/download_models.sh" "$@"
    
    echo -e "${GREEN}${BOLD}[✓] Model download process completed${RESET}"
}

# Show help
show_help() {
    echo -e "${CYAN}${BOLD}Usage:${RESET}"
    echo -e "  ${GREEN}./scripts/manage.sh${RESET} ${YELLOW}<command>${RESET}"
    echo
    echo -e "${CYAN}${BOLD}Basic Commands:${RESET}"
    echo -e "  ${YELLOW}setup${RESET}      Create virtual environment and install dependencies"
    echo -e "  ${YELLOW}models${RESET}     Download speaker embedding models to local directory"
    echo -e "  ${YELLOW}start${RESET}      Start the standard CyberVox application"
    echo -e "  ${YELLOW}stop${RESET}       Stop the application"
    echo -e "  ${YELLOW}restart${RESET}    Restart the application"
    echo
    echo -e "${MAGENTA}${BOLD}🧪 Trisha's Audio Lab Commands:${RESET}"
    echo -e "  ${YELLOW}lab${RESET}        🧪⚡ Start Trisha's Audio Lab Demo (enhanced features)"
    echo -e "  ${YELLOW}test-lab${RESET}   🔬 Test audio enhancement functionality"
    echo
    echo -e "${CYAN}${BOLD}Maintenance Commands:${RESET}"
    echo -e "  ${YELLOW}update${RESET}     Update dependencies"
    echo -e "  ${YELLOW}clean${RESET}      Clean cache files"
    echo -e "  ${YELLOW}test${RESET}       Run standard tests"
    echo -e "  ${YELLOW}gpu${RESET}        Test GPU functionality"
    echo -e "  ${YELLOW}help${RESET}       Show this help message"
    echo
    echo -e "${GREEN}${BOLD}🎯 Quick Start:${RESET}"
    echo -e "  ${CYAN}1.${RESET} ${GREEN}./scripts/manage.sh setup${RESET}    # Install everything"
    echo -e "  ${CYAN}2.${RESET} ${GREEN}./scripts/manage.sh lab${RESET}      # Start Trisha's Lab! 🧪"
}

# Main script logic
main() {
    show_banner
    
    if [ "$#" -eq 0 ]; then
        show_help
        exit 0
    fi

    case "$1" in
        setup)
            check_venv
            activate_venv
            install_deps
            ;;
        models)
            shift  # Remove the first argument (models)
            check_venv
            activate_venv
            download_models "$@"
            ;;
        start)
            check_venv
            activate_venv
            start_app
            ;;
        lab)
            check_venv
            activate_venv
            start_lab
            ;;
        test-lab)
            check_venv
            activate_venv
            test_lab
            ;;
        stop)
            stop_app
            ;;
        restart)
            stop_app
            sleep 2
            check_venv
            activate_venv
            start_app
            ;;
        restart-lab)
            stop_app
            sleep 2
            check_venv
            activate_venv
            start_lab
            ;;
        update)
            check_venv
            update_deps
            ;;
        clean)
            clean_cache
            ;;
        test)
            check_venv
            activate_venv
            run_tests
            ;;
        gpu)
            check_venv
            activate_venv
            test_gpu
            ;;
        help)
            show_help
            ;;
        *)
            echo -e "${RED}${BOLD}[!] Unknown command: $1${RESET}"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
