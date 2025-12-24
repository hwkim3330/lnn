# LNN - Liquid Neural Network

Real-time adaptive neural networks with ODE-based dynamics.

![LNN Tau Dynamics](lnn_tau_dynamics.png)

## Core Concept

```
ODE: dh/dt = (-h + f(W_in·x + W_rec·h)) / τ

Where τ (time constant) adapts based on input:
τ = τ_min + (τ_max - τ_min) · σ(W_τ·[x, h])
```

τ가 입력에 따라 실시간으로 변합니다:
- 낮은 τ → 빠른 반응
- 높은 τ → 느린 기억

## Models

### 1. LNN Lite (Browser-ready)
```
VisionLNNLite: 0.77M params (~1.5MB FP16)
- Vision encoder: Tiny CNN
- LNN core: 4 Liquid layers
- Output: Action + Coordinates
```

```python
from web.lnn_lite import VisionLNNLite

model = VisionLNNLite(num_actions=6)
result = model(image)
# result["action_logits"], result["coords"], result["tau_values"]
```

### 2. LFM2-Liquid (Full VLM)
```
LFM2-VL + LNN: 455M params (~900MB FP16)
- LFM2-VL: Vision-Language Model
- Liquid layers: Merged with ShortConv
```

```python
from core.lfm2_liquid_merged import LFM2LiquidMerged

model = LFM2LiquidMerged(liquid_layers=[0, 8, 15])
response = model.generate(image, "What do you see?")
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Image     │ ──► │   Vision    │ ──► │    LNN      │
│  (224x224)  │     │   Encoder   │     │  (Liquid    │
└─────────────┘     └─────────────┘     │   Layers)   │
                                        └──────┬──────┘
                                               │
                    ┌──────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   τ₀   τ₁   τ₂   τ₃    │  ← Adaptive time constants
        │  ─────────────────────  │
        │    Liquid Dynamics      │
        │    dh/dt = (-h+f)/τ    │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Action Head           │
        │   ├── action_logits     │
        │   └── coordinates       │
        └─────────────────────────┘
```

## Files

```
lnn-repo/
├── core/
│   ├── liquid_neural_net.py    # Pure LNN implementation
│   ├── lfm2_liquid_merged.py   # LFM2-VL + LNN merged
│   └── lfm2_lnn_hybrid.py      # LFM2-VL + LNN hooks
├── browser/
│   ├── browser_lnn_real.py     # Browser agent with LNN
│   └── browser_lfm2.py         # Browser agent with LFM2
├── web/
│   └── lnn_lite.py             # Lightweight for WebGPU
└── lnn_tau_dynamics.png        # Visualization
```

## Quick Start

```bash
# Test LNN Lite (browser-ready)
python3 web/lnn_lite.py

# Test merged model (requires LFM2-VL weights)
python3 core/lfm2_liquid_merged.py

# Run browser agent
DISPLAY=:1 python3 browser/browser_lnn_real.py --task "Go to google.com"
```

## Browser Deployment

The LNN Lite model (1.5MB) can run in browsers via WebGPU/WebNN:

```bash
# Export to ONNX
python3 -c "from web.lnn_lite import VisionLNNLite, export_to_onnx; export_to_onnx(VisionLNNLite())"

# Then convert to WASM using onnxruntime-web
```

## Key Features

- **Adaptive τ**: Time constant changes based on input
- **Temporal Memory**: State persists across timesteps
- **ODE Dynamics**: Continuous-time neural computation
- **Lightweight**: 0.77M params for browser version

## References

- [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439) (Hasani et al., 2021)
- [LFM2-VL](https://huggingface.co/LiquidAI/LFM2-VL-450M)
