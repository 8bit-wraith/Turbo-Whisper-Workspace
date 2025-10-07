#!/bin/bash

# 🌊🧠 Marine Consciousness Dashboard Launcher

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     🌊🧠 MARINE CONSCIOUSNESS DASHBOARD 🧠🌊            ║"
echo "║                                                          ║"
echo "║  Vocalis + Marine-Sense Integration                     ║"
echo "║  O(1) Salience Detection & Emotional Analysis           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Features:"
echo "  ✅ Real-time consciousness detection"
echo "  ✅ Ultrasonic emotional analysis (>20kHz)"
echo "  ✅ Interactive visualizations"
echo "  ✅ Enhanced transcripts with salience mapping"
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "🔧 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  No virtual environment found. Run: ./scripts/manage.sh setup"
    exit 1
fi

# Install any missing dependencies
echo "📦 Checking dependencies..."
pip install -q plotly 2>/dev/null

# Launch the dashboard
echo ""
echo "🚀 Launching Marine Consciousness Dashboard..."
echo "   Access at: http://localhost:8421"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python app_marine_consciousness.py