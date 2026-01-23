#!/bin/bash
# Install Qwen Omni + Claude Hybrid Voice System for Aria
# This script sets up the dependencies for the cost-effective voice system

set -e

echo "=================================================="
echo "Installing Qwen + Claude Hybrid Voice System"
echo "=================================================="

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

echo ""
echo "[1/5] Installing core dependencies..."
pip install --upgrade pip

# Core ML dependencies
pip install torch torchvision torchaudio

echo ""
echo "[2/5] Installing Qwen2.5-Omni dependencies..."
# Transformers with Qwen support
pip install transformers>=4.45.0 accelerate

# For 4-bit quantization (saves memory)
pip install bitsandbytes

echo ""
echo "[3/5] Installing audio dependencies..."
pip install sounddevice numpy soundfile

# Optional: mlx-whisper for Apple Silicon optimization
if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon - installing MLX optimizations..."
    pip install mlx mlx-whisper
fi

echo ""
echo "[4/5] Installing Ollama (fallback LLM)..."
# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Pull recommended models
echo "Pulling Ollama models (this may take a while)..."
ollama pull qwen2.5:7b || echo "Warning: Could not pull qwen2.5:7b"
ollama pull deepseek-r1:7b || echo "Warning: Could not pull deepseek-r1:7b"

echo ""
echo "[5/5] Verifying installation..."

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError:
    print('Transformers: NOT INSTALLED')

try:
    import sounddevice
    print(f'Sounddevice: OK')
except ImportError:
    print('Sounddevice: NOT INSTALLED')

try:
    import mlx
    print(f'MLX: {mlx.__version__}')
except ImportError:
    print('MLX: Not installed (optional)')
"

echo ""
echo "=================================================="
echo "Installation complete!"
echo "=================================================="
echo ""
echo "To test the hybrid voice system:"
echo "  python -m aria.qwen_voice"
echo ""
echo "Or run the quick test:"
echo "  python test_qwen_hybrid.py"
echo ""
echo "Cost comparison:"
echo "  OpenAI Realtime: ~\$0.30/min"
echo "  Qwen+Claude Hybrid: ~\$0.01/complex request (simple = FREE)"
echo ""
