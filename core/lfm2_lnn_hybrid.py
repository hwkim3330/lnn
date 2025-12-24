#!/usr/bin/env python3
"""
LFM2-VL + LNN 하이브리드 모델

LFM2-VL의 ShortConv를 Liquid Neural Network로 대체/증강
- 원본 모델 구조 유지
- ShortConv 옆에 LiquidLayer 추가
- 시간 의존적 동적 상태 유지

This makes LNN a REALITY by integrating it with a working VLM!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class LiquidCell(nn.Module):
    """
    Liquid Neural Network Cell

    ODE: dx/dt = (-x + f(W_in*input + W_rec*x)) / tau

    tau (time constant) adapts based on input
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dt: float = 0.1,
        tau_min: float = 1.0,
        tau_max: float = 10.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Input projection
        self.W_in = nn.Linear(input_dim, hidden_dim)

        # Recurrent connection
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Adaptive time constant network
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def get_tau(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute adaptive time constant"""
        combined = torch.cat([x, h], dim=-1)
        tau_raw = self.tau_net(combined)
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_raw
        return tau

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [batch, seq, input_dim]
            h: Hidden state [batch, hidden_dim]

        Returns:
            output: [batch, seq, hidden_dim]
            final_h: [batch, hidden_dim]
            tau_history: [batch, seq, 1]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=x.dtype)

        outputs = []
        tau_history = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_dim]

            # Adaptive time constant
            tau = self.get_tau(x_t, h)  # [batch, 1]
            tau_history.append(tau)

            # ODE dynamics: dh/dt = (-h + f(x, h)) / tau
            f_xh = torch.tanh(self.W_in(x_t) + self.W_rec(h))
            dh = (-h + f_xh) / tau

            # Euler step
            h = h + self.dt * dh
            h = self.norm(h)

            outputs.append(h)

        output = torch.stack(outputs, dim=1)  # [batch, seq, hidden]
        tau_history = torch.stack(tau_history, dim=1)  # [batch, seq, 1]

        return output, h, tau_history


class LiquidLayer(nn.Module):
    """
    Full Liquid Layer that can replace/augment ShortConv

    Matches LFM2's ShortConv interface
    """

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()

        # Match ShortConv's projection dimensions
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Liquid cell
        self.liquid = LiquidCell(
            input_dim=hidden_dim * 3,
            hidden_dim=hidden_dim,
            dt=0.1,
            tau_min=1.0,
            tau_max=5.0
        )

        # Gate to blend with original
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Cache for streaming
        self.register_buffer('hidden_state', None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
    ) -> torch.Tensor:
        """
        Process hidden states through liquid dynamics

        Args:
            hidden_states: [batch, seq, hidden_dim]

        Returns:
            output: [batch, seq, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Project input (like ShortConv's in_proj)
        projected = self.in_proj(hidden_states)  # [batch, seq, hidden*3]

        # Get cached state
        h = self.hidden_state
        if h is None or h.shape[0] != batch_size:
            h = torch.zeros(batch_size, hidden_dim,
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)

        # Liquid dynamics
        liquid_out, final_h, tau = self.liquid(projected, h)

        # Update cache
        self.hidden_state = final_h.detach()

        # Output projection
        output = self.out_proj(liquid_out)

        # Return in same format as ShortConv
        return output

    def reset_state(self):
        """Reset hidden state for new sequence"""
        self.hidden_state = None


class LFM2LNNHybrid(nn.Module):
    """
    LFM2-VL with LNN augmentation

    Loads original LFM2-VL and adds Liquid layers
    """

    def __init__(
        self,
        model_path: str = "/home/kim/models/lfm2-vl-450m",
        lnn_layers: List[int] = [0, 4, 8, 12],  # Which layers get LNN
        device: str = "cuda",
        load_to_ram_first: bool = True
    ):
        super().__init__()

        self.device = device
        self.lnn_layer_ids = lnn_layers

        print("=" * 60)
        print("LFM2-VL + LNN HYBRID MODEL")
        print("=" * 60)

        # Step 1: Load original model to CPU (RAM) first
        print("\n[1/4] Loading LFM2-VL to RAM...")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.original_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu" if load_to_ram_first else device,
            trust_remote_code=True,
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        # Step 2: Create LNN layers
        print(f"\n[2/4] Creating LNN layers for positions {lnn_layers}...")
        self.liquid_layers = nn.ModuleDict()
        hidden_dim = 1024  # LFM2's hidden dim

        for layer_idx in lnn_layers:
            self.liquid_layers[str(layer_idx)] = LiquidLayer(hidden_dim)

        # Step 3: Move to device in parts
        print(f"\n[3/4] Moving to {device}...")

        # Move vision tower
        print("    Vision Tower → GPU")
        self.original_model.model.vision_tower = \
            self.original_model.model.vision_tower.to(device)

        # Move projector
        print("    Projector → GPU")
        self.original_model.model.multi_modal_projector = \
            self.original_model.model.multi_modal_projector.to(device)

        # Move embeddings
        print("    Embeddings → GPU")
        self.original_model.model.language_model.embed_tokens = \
            self.original_model.model.language_model.embed_tokens.to(device)

        # Move decoder layers one by one
        print("    Decoder Layers → GPU")
        for i, layer in enumerate(self.original_model.model.language_model.layers):
            layer.to(device)

        # Move rotary embeddings
        self.original_model.model.language_model.rotary_emb = \
            self.original_model.model.language_model.rotary_emb.to(device)
        self.original_model.model.language_model.pos_emb = \
            self.original_model.model.language_model.pos_emb.to(device)
        self.original_model.model.language_model.embedding_norm = \
            self.original_model.model.language_model.embedding_norm.to(device)

        # Move LM head
        self.original_model.lm_head = self.original_model.lm_head.to(device)

        # Move liquid layers
        print("    LNN Layers → GPU")
        self.liquid_layers = self.liquid_layers.to(device)

        # Step 4: Hook into decoder layers
        print("\n[4/4] Hooking LNN into decoder layers...")
        self._install_hooks()

        print("\n" + "=" * 60)
        print("READY!")
        print(f"  Original params: {sum(p.numel() for p in self.original_model.parameters())/1e6:.1f}M")
        print(f"  LNN params: {sum(p.numel() for p in self.liquid_layers.parameters())/1e6:.2f}M")
        print("=" * 60)

    def _install_hooks(self):
        """Install forward hooks to inject LNN processing"""
        self.hooks = []

        for layer_idx in self.lnn_layer_ids:
            layer = self.original_model.model.language_model.layers[layer_idx]
            liquid = self.liquid_layers[str(layer_idx)]

            def make_hook(lnn):
                def hook(module, input, output):
                    # output is (hidden_states, ...)
                    if isinstance(output, tuple):
                        hidden = output[0]
                        lnn_out = lnn(hidden)
                        # Add LNN output as residual
                        new_hidden = hidden + 0.1 * lnn_out  # Small contribution
                        return (new_hidden,) + output[1:]
                    return output
                return hook

            h = layer.register_forward_hook(make_hook(liquid))
            self.hooks.append(h)
            print(f"    Layer {layer_idx}: LNN hook installed")

    def reset_lnn_states(self):
        """Reset all LNN hidden states"""
        for layer in self.liquid_layers.values():
            layer.reset_state()

    @torch.no_grad()
    def generate(
        self,
        image,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """Generate response for image + prompt"""

        # Reset LNN states for new sequence
        self.reset_lnn_states()

        # Build messages
        from PIL import Image
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]

        # Process with chat template
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            images=[image] if not isinstance(image, list) else image,
            text=text,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        output_ids = self.original_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id
        )

        # Decode only new tokens
        response = self.processor.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def get_tau_values(self) -> Dict[int, torch.Tensor]:
        """Get current tau values from each LNN layer"""
        taus = {}
        for idx, layer in self.liquid_layers.items():
            if layer.liquid.tau_net is not None:
                # Get from last forward pass
                taus[int(idx)] = getattr(layer, '_last_tau', None)
        return taus


def demo():
    """Demo the hybrid model"""
    print("\n" + "=" * 60)
    print("LFM2-VL + LNN DEMO")
    print("=" * 60)

    from PIL import Image
    import io

    # Create model
    model = LFM2LNNHybrid(
        lnn_layers=[0, 4, 8, 12],  # 4 LNN layers spread across decoder
        device="cuda"
    )

    # Create test image
    img = Image.new('RGB', (224, 224), color='blue')

    # Test generation
    print("\n[TEST] Generating with LNN-augmented model...")
    response = model.generate(
        img,
        "Describe this image briefly.",
        max_new_tokens=50
    )
    print(f"Response: {response}")

    # Show model info
    print("\n[INFO] LNN Layers:")
    for idx, layer in model.liquid_layers.items():
        print(f"  Layer {idx}: LiquidLayer (hidden_dim=1024)")


if __name__ == "__main__":
    demo()
