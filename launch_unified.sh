#!/bin/bash
# 🎧⚡ Unified Vocalis Audio Workspace Launcher ⚡🎧

echo "🎧⚡ Starting Unified Vocalis Audio Workspace..."

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "🔧 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system Python."
fi

# Check if unified app exists
if [ ! -f "unified_vocalis_app.py" ]; then
    echo "❌ Error: unified_vocalis_app.py not found!"
    exit 1
fi

# Launch the unified app
echo "🚀 Launching unified interface..."
echo "📱 Access at: http://localhost:7860"
echo "🎯 Features:"
echo "  ✅ Audio Processing & Transcription"
echo "  ✅ Speaker Diarization"
echo "  ✅ Audio Enhancement (Trisha's Lab)"
echo "  ✅ Audio Analysis & Visualization"
echo "  ✅ Consciousness Analysis (if available)"
echo "  ✅ Interactive Chat Bubbles"
echo "  ✅ Real-time Processing"
echo "  ✅ Multiple Export Formats"
echo ""

python unified_vocalis_app.py