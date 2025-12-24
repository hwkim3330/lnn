#!/usr/bin/env python3
"""
Liquid Neural Network (LNN) - From Scratch

Based on: "Liquid Time-constant Networks" (Hasani et al., 2021)

Key features:
- Continuous-time dynamics (ODE-based)
- Adaptive time constants
- State that evolves over time
- Real-time adaptation

Architecture:
  x(t+1) = x(t) + dt * f(x(t), I(t), θ)

Where f is the neural dynamics function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class LiquidCell(nn.Module):
    """
    Single Liquid Neural Network Cell

    Implements continuous-time dynamics:
    dx/dt = (-x + f(x, I)) / τ

    Where:
    - x: hidden state
    - I: input
    - τ: time constant (learned, adaptive)
    - f: nonlinear transformation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dt: float = 0.1,
        tau_min: float = 0.5,
        tau_max: float = 5.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Input projection
        self.W_in = nn.Linear(input_dim, hidden_dim)

        # Recurrent connections (state -> state)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Time constant network (adaptive τ)
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()  # Output in [0, 1], scaled to [tau_min, tau_max]
        )

        # Activation
        self.activation = nn.Tanh()

        # Layer norm for stability
        self.ln = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialize recurrent weights to be stable
        nn.init.orthogonal_(self.W_rec.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.zeros_(self.W_in.bias)

    def get_tau(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute adaptive time constant"""
        combined = torch.cat([x, h], dim=-1)
        tau_raw = self.tau_net(combined)
        # Scale to [tau_min, tau_max]
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_raw
        return tau

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with ODE dynamics

        Args:
            x: Input tensor (batch, input_dim)
            h: Previous hidden state (batch, hidden_dim)

        Returns:
            new_h: Updated hidden state
            tau: Time constants used
        """
        batch_size = x.size(0)

        # Initialize hidden state if needed
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Compute adaptive time constant
        tau = self.get_tau(x, h)

        # Compute dynamics: dx/dt = (-x + f(x, I)) / τ
        # f(x, I) = tanh(W_in * I + W_rec * x)
        input_contrib = self.W_in(x)
        recurrent_contrib = self.W_rec(h)

        f_xI = self.activation(input_contrib + recurrent_contrib)

        # ODE step: x(t+dt) = x(t) + dt * dx/dt
        # dx/dt = (-h + f_xI) / tau
        dh = (-h + f_xI) / tau
        new_h = h + self.dt * dh

        # Normalize for stability
        new_h = self.ln(new_h)

        return new_h, tau


class LiquidNeuralNetwork(nn.Module):
    """
    Multi-layer Liquid Neural Network

    Stacks multiple LiquidCells with residual connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        dt: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Liquid cells
        self.cells = nn.ModuleList([
            LiquidCell(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dt=dt
            )
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Track hidden states for continuous operation
        self.hidden_states = None

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden states"""
        self.hidden_states = [
            torch.zeros(batch_size, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

    def forward(
        self,
        x: torch.Tensor,
        reset_state: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass

        Args:
            x: Input (batch, input_dim)
            reset_state: Whether to reset hidden states

        Returns:
            output: (batch, output_dim)
            info: Dictionary with tau values and states
        """
        batch_size = x.size(0)

        # Initialize or reset hidden states
        if self.hidden_states is None or reset_state:
            self.init_hidden(batch_size, x.device)

        # Project input
        h = self.input_proj(x)

        # Pass through liquid cells
        all_taus = []
        for i, cell in enumerate(self.cells):
            # Residual connection
            residual = h

            # Liquid dynamics
            h_new, tau = cell(h, self.hidden_states[i])

            # Update stored hidden state (for continuous operation)
            self.hidden_states[i] = h_new.detach()  # Detach to prevent BPTT explosion

            # Residual + dropout
            h = residual + self.dropout(h_new)

            all_taus.append(tau.mean().item())

        # Output projection
        output = self.output_proj(h)

        info = {
            'tau_values': all_taus,
            'hidden_norm': [hs.norm().item() for hs in self.hidden_states]
        }

        return output, info


class VisionLNN(nn.Module):
    """
    Vision-based Liquid Neural Network for Browser Agent

    Architecture:
    1. Vision Encoder (ViT/SigLIP - frozen or fine-tuned)
    2. Task Encoder (text embedding)
    3. Fusion Layer
    4. LNN Core (continuous dynamics)
    5. Action Head
    """

    def __init__(
        self,
        vision_dim: int = 768,    # SigLIP output dim
        task_dim: int = 128,      # Task embedding dim
        hidden_dim: int = 256,
        num_actions: int = 6,
        num_layers: int = 4,
        dt: float = 0.1,
    ):
        super().__init__()

        # Task encoder (simple for now)
        self.task_vocab_size = 1000
        self.task_embed = nn.Embedding(self.task_vocab_size, task_dim)
        self.task_encoder = nn.LSTM(task_dim, task_dim, batch_first=True)

        # Fusion: combine vision + task
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + task_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # LNN Core
        self.lnn = LiquidNeuralNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dt=dt,
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        # Value head (for actor-critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def encode_task(self, task_tokens: torch.Tensor) -> torch.Tensor:
        """Encode task description"""
        # task_tokens: (batch, seq_len)
        embedded = self.task_embed(task_tokens)
        _, (hidden, _) = self.task_encoder(embedded)
        return hidden.squeeze(0)

    def forward(
        self,
        vision_features: torch.Tensor,  # From ViT/SigLIP
        task_features: torch.Tensor,    # Task embedding
        reset_state: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass

        Returns:
            action_logits: (batch, num_actions)
            value: (batch, 1)
            info: LNN dynamics info
        """
        # Fuse vision and task
        combined = torch.cat([vision_features, task_features], dim=-1)
        fused = self.fusion(combined)

        # LNN dynamics
        lnn_output, info = self.lnn(fused, reset_state=reset_state)

        # Action and value
        action_logits = self.action_head(lnn_output)
        value = self.value_head(lnn_output)

        return action_logits, value, info


def test_lnn():
    """Test LNN components"""
    print("=" * 60)
    print("LIQUID NEURAL NETWORK TEST")
    print("=" * 60)

    # Test LiquidCell
    print("\n[1] LiquidCell Test")
    cell = LiquidCell(input_dim=64, hidden_dim=128)
    x = torch.randn(4, 64)
    h = None

    for step in range(5):
        h, tau = cell(x, h)
        print(f"  Step {step}: h.norm={h.norm():.3f}, tau.mean={tau.mean():.3f}")

    # Test LNN
    print("\n[2] LiquidNeuralNetwork Test")
    lnn = LiquidNeuralNetwork(
        input_dim=768,
        hidden_dim=256,
        output_dim=128,
        num_layers=4
    )

    x = torch.randn(4, 768)  # Simulated ViT output

    for step in range(5):
        out, info = lnn(x, reset_state=(step == 0))
        print(f"  Step {step}: out.norm={out.norm():.3f}, "
              f"tau={[f'{t:.2f}' for t in info['tau_values']]}")

    # Count parameters
    total_params = sum(p.numel() for p in lnn.parameters())
    print(f"\n  LNN parameters: {total_params:,}")

    # Test VisionLNN
    print("\n[3] VisionLNN Test")
    vlnn = VisionLNN(
        vision_dim=768,
        task_dim=128,
        hidden_dim=256,
        num_actions=6,
        num_layers=4
    )

    vision_feat = torch.randn(4, 768)
    task_feat = torch.randn(4, 128)

    for step in range(5):
        action_logits, value, info = vlnn(
            vision_feat, task_feat,
            reset_state=(step == 0)
        )
        probs = F.softmax(action_logits, dim=-1)
        print(f"  Step {step}: action_probs={probs[0].tolist()[:3]}..., "
              f"value={value[0].item():.3f}")

    total_params = sum(p.numel() for p in vlnn.parameters())
    print(f"\n  VisionLNN parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("LNN TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_lnn()
