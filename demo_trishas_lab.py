#!/usr/bin/env python3
"""
🧪⚡ Trisha's Audio Lab Demo Script
Welcome to the Audio Enhancement Laboratory!

This script demonstrates the new audio denoising features
that have been integrated into the Gradio app.

Run this to launch the enhanced CyberVox workspace with Trisha's Lab!
"""

import os
import sys

def main():
    print("""
    🧪⚡ Welcome to Trisha's Audio Enhancement Laboratory! ⚡🧪
    
    ═══════════════════════════════════════════════════════════════════
    
    🎉 NEW FEATURES ADDED:
    
    📍 **Trisha's Audio Lab Tab**: 
       - 🎛️ Advanced denoising methods (Gentle, Standard, Aggressive)
       - 🔊 High-pass filtering (100Hz, 150Hz, 200Hz)
       - 🔇 Dynamic volume reduction for loud segments
       - 📊 Before/after waveform comparison
       - ✨ Real-time processing reports
    
    🎯 **Perfect for**:
       - Cleaning up noisy recordings
       - Improving transcription quality  
       - Removing background hum and rumble
       - Balancing audio levels
    
    ═══════════════════════════════════════════════════════════════════
    
    🚀 Starting CyberVox with Trisha's Audio Lab...
    
    Look for the new "🧪 Trisha's Audio Lab" tab in the interface!
    """)
    
    # Import and launch the app
    try:
        print("📦 Loading Gradio app...")
        import app
        print("🎧 Launching CyberVox Audio Workspace...")
        
        # Launch with enhanced settings for local development
        app.demo.queue().launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_api=False,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Trisha's Lab... Thanks for visiting! 👋")
    except Exception as e:
        print(f"💥 Lab explosion! Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()