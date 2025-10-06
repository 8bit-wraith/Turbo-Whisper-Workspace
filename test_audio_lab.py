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
