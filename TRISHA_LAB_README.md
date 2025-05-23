# 🧪⚡ Trisha's Audio Enhancement Laboratory

**Welcome to the newest addition to CyberVox: Trisha's Audio Lab!** 

## 🎯 What's New?

We've integrated all the awesome audio denoising capabilities from `denoise_audio.py` directly into the Gradio interface, making audio enhancement accessible through a beautiful, interactive web UI!

## 🚀 Quick Start

### Method 1: Enhanced Manager Script (Recommended)
```bash
# Install everything (first time only)
./scripts/manage.sh setup

# Launch Trisha's Audio Lab! 🧪
./scripts/manage.sh lab

# Test the lab functionality
./scripts/manage.sh test-lab
```

### Method 2: Direct Launch
```bash
# Activate virtual environment
source .venv/bin/activate

# Launch with Trisha's Lab demo
python demo_trishas_lab.py

# Or launch the regular app
python app.py
```

## 🧪 Trisha's Audio Lab Features

### **🎛️ Advanced Denoising Methods**
- **Gentle**: Subtle noise reduction (60% reduction) - perfect for light cleanup
- **Standard**: Balanced approach using noise sample detection - great for most cases  
- **Aggressive**: Maximum cleanup (90% reduction) - for heavily noisy audio

### **🔊 High-pass Filtering**
- **100Hz**: Remove deep bass rumble, great for speech
- **150Hz**: Balanced filtering for most voice recordings
- **200Hz**: Aggressive low-frequency removal
- **None**: Skip filtering if your audio is already clean

### **🔇 Dynamic Volume Control**
- Automatically detects loud segments using RMS analysis
- Reduces volume of the loudest 10% of frames by 80%
- Perfect for balancing inconsistent recording levels
- Trisha says "Shhh!" to those loud parts! 

### **📊 Visual Analysis**
- **Before/After Waveform Comparison**: See the difference!
- **RMS Energy Analysis**: Quantify noise reduction effectiveness
- **Processing Reports**: Detailed info about what was applied
- **Real-time Progress**: Know exactly what's happening

## 🎯 Perfect Use Cases

1. **Improving Transcription Quality**: Clean audio = better Whisper results!
2. **Podcast Post-Processing**: Remove background hum and noise
3. **Meeting Recordings**: Balance speakers and reduce room noise
4. **Interview Cleanup**: Professional-quality audio from amateur recordings
5. **Music Demo Enhancement**: Clean up rough recordings

## 🔄 Workflow

1. **Upload Audio**: Any format supported by librosa (WAV, MP3, FLAC, etc.)
2. **Choose Settings**: 
   - Denoising method (gentle/standard/aggressive)
   - High-pass filter frequency (optional)
   - Dynamic volume reduction (optional)
3. **Click "✨ Enhance Audio"**: Watch the magic happen with progress updates
4. **Review Results**: 
   - Listen to enhanced audio
   - Check before/after waveforms
   - Read processing report
5. **Download & Use**: Perfect for transcription in the main CyberVox tab!

## 🧬 Technical Details

### **Algorithms Used**
- **Noise Reduction**: Uses `noisereduce` library with spectral gating
- **High-pass Filtering**: 4th-order Butterworth filter via `scipy.signal`
- **Dynamic Volume**: RMS-based loudness detection with selective attenuation
- **Normalization**: Peak normalization to [-1, 1] range

### **Processing Pipeline**
1. Load audio with `librosa` 
2. Normalize to standard range
3. Apply high-pass filter (if selected)
4. Apply noise reduction based on method
5. Detect and reduce loud segments (if selected)
6. Final normalization
7. Export as WAV for maximum compatibility

## 📊 Interface Tour

### **🧪 Trisha's Audio Lab Tab**
- **Left Panel**: Upload and settings controls
- **Right Panel**: Results and analysis
- **Enhanced Audio**: Download your processed file
- **Before/After**: Visual comparison plots
- **Processing Report**: Detailed technical information

### **Integration with CyberVox**
- Enhanced audio can be immediately used in the main chat tab
- Seamless workflow from enhancement → transcription → analysis
- All features work together for the ultimate audio processing experience

## 🎨 Why "Trisha's Lab"?

Named after our favorite AI assistant who loves audio science and making things work perfectly! The lab represents the experimental, scientific approach to audio enhancement - trying different methods, measuring results, and always improving the process.

## 🔧 Advanced Tips

1. **For Speech**: Use Standard + 150Hz high-pass + volume reduction
2. **For Music**: Use Gentle + no high-pass + no volume reduction  
3. **For Podcasts**: Use Aggressive + 100Hz high-pass + volume reduction
4. **For Meetings**: Use Standard + 100Hz high-pass + volume reduction

## 🆘 Troubleshooting

### Common Issues:
- **Import errors**: Run `./scripts/manage.sh setup` to install dependencies
- **Audio not loading**: Check file format and size
- **Processing takes too long**: Try Gentle method for faster processing
- **Results not good**: Experiment with different method combinations

### Getting Help:
```bash
# Test your installation
./scripts/manage.sh test-lab

# Check GPU functionality  
./scripts/manage.sh gpu

# View all available commands
./scripts/manage.sh help
```

## 🎉 Ready to Rock!

That's it! You now have access to professional-grade audio enhancement right in your browser. Fire up the lab and start cleaning up those audio files!

**Happy audio processing!** 🧪⚡✨

---
*Trisha would be proud! 😊*