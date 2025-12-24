#!/usr/bin/env python3
"""
LNN Online Learner

실시간 브라우저 조작하면서 학습
- 행동 → 결과 → 보상 → 가중치 업데이트
- 매 스텝마다 학습 (Online Learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from PIL import Image
import io
import time
from collections import deque
from typing import Optional, Dict, List, Tuple


class LiquidCellTrainable(nn.Module):
    """학습 가능한 Liquid Cell"""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.W_in = nn.Linear(hidden_dim, hidden_dim)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tau_net = nn.Linear(hidden_dim * 2, 1)
        self.norm = nn.LayerNorm(hidden_dim)

        # Initialize for stability
        nn.init.orthogonal_(self.W_rec.weight, gain=0.5)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=-1)
        tau = 1.0 + 4.0 * torch.sigmoid(self.tau_net(combined))

        f_xh = torch.tanh(self.W_in(x) + self.W_rec(h))
        dh = (-h + f_xh) / tau
        new_h = h + 0.1 * dh
        new_h = self.norm(new_h)

        return new_h, tau


class OnlineLNN(nn.Module):
    """
    온라인 학습용 LNN

    Actor-Critic 구조:
    - Actor: 행동 확률 출력
    - Critic: 상태 가치 추정
    """

    def __init__(
        self,
        num_actions: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Vision encoder (simple CNN)
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, hidden_dim)
        )

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Liquid layers
        self.cells = nn.ModuleList([
            LiquidCellTrainable(hidden_dim)
            for _ in range(num_layers)
        ])

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Coordinate head
        self.coord = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # x_mean, y_mean, x_std, y_std
        )

        # Hidden states
        self.hidden_states: Optional[List[torch.Tensor]] = None

    def reset_hidden(self, batch_size: int = 1, device: str = "cuda"):
        self.hidden_states = [
            torch.zeros(batch_size, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Returns:
            action_logits, value, coord_params, tau_values
        """
        batch_size = image.shape[0]
        device = image.device

        if self.hidden_states is None:
            self.reset_hidden(batch_size, device)

        # Vision encoding
        vis = self.vision(image)
        h = self.input_proj(vis)

        # Liquid dynamics
        tau_values = []
        for i, cell in enumerate(self.cells):
            h, tau = cell(h, self.hidden_states[i])
            self.hidden_states[i] = h
            tau_values.append(tau)

        # Outputs
        action_logits = self.actor(h)
        value = self.critic(h)
        coord_params = self.coord(h)

        return {
            "action_logits": action_logits,
            "value": value,
            "coord_params": coord_params,
            "tau_values": torch.cat(tau_values, dim=-1)
        }

    def get_action(self, image: torch.Tensor) -> Tuple[int, float, float, float, Dict]:
        """
        Get action with exploration

        Returns:
            action_idx, x, y, log_prob, info
        """
        with torch.no_grad():
            out = self.forward(image)

        # Action sampling
        probs = F.softmax(out["action_logits"], dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Coordinate sampling
        cp = out["coord_params"][0]
        x_mean, y_mean = torch.sigmoid(cp[0]).item(), torch.sigmoid(cp[1]).item()
        x_std, y_std = 0.1 * torch.sigmoid(cp[2]).item(), 0.1 * torch.sigmoid(cp[3]).item()

        x = np.clip(np.random.normal(x_mean, x_std), 0, 1)
        y = np.clip(np.random.normal(y_mean, y_std), 0, 1)

        info = {
            "probs": probs[0].cpu().numpy(),
            "value": out["value"].item(),
            "tau": out["tau_values"][0].cpu().numpy()
        }

        return action.item(), x, y, log_prob.item(), info


class OnlineLearner:
    """
    온라인 학습 시스템

    PPO-style online learning with:
    - Experience buffer (small, recent)
    - Frequent updates
    - Adaptive learning rate
    """

    def __init__(
        self,
        model: OnlineLNN,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        update_freq: int = 5,  # Update every N steps
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_freq = update_freq

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Experience buffer
        self.buffer = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": []
        }

        # Stats
        self.total_steps = 0
        self.total_updates = 0
        self.episode_rewards = []

    def store(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ):
        """Store experience"""
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["rewards"].append(reward)
        self.buffer["values"].append(value)
        self.buffer["dones"].append(done)

        self.total_steps += 1

        # Update if buffer is full
        if len(self.buffer["rewards"]) >= self.update_freq:
            self.update()

    def update(self) -> Dict[str, float]:
        """Perform online update"""
        if len(self.buffer["rewards"]) == 0:
            return {}

        # Convert to tensors
        states = torch.cat(self.buffer["states"]).to(self.device)
        actions = torch.tensor(self.buffer["actions"], device=self.device)
        old_log_probs = torch.tensor(self.buffer["log_probs"], device=self.device)
        rewards = self.buffer["rewards"]
        values = self.buffer["values"]
        dones = self.buffer["dones"]

        # Compute returns and advantages (GAE)
        returns = []
        advantages = []
        gae = 0
        next_value = values[-1] if not dones[-1] else 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)

        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        self.model.train()
        self.model.reset_hidden(len(rewards), self.device)

        # Forward pass
        outputs = self.model(states)
        action_logits = outputs["action_logits"]
        values_pred = outputs["value"].squeeze()

        # Policy loss
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values_pred, returns)

        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        self.total_updates += 1

        # Clear buffer
        for key in self.buffer:
            self.buffer[key] = []

        # Reset hidden states
        self.model.reset_hidden(1, self.device)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item()
        }

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "total_updates": self.total_updates
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.total_updates = checkpoint["total_updates"]


class BrowserEnvironment:
    """브라우저 환경"""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.screen_width = 1280
        self.screen_height = 720
        self.page = None
        self.action_names = ["click", "type", "navigate", "scroll", "key", "done"]

    def setup(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            viewport={"width": self.screen_width, "height": self.screen_height}
        )
        self.page = self._context.new_page()

    def cleanup(self):
        if hasattr(self, "_browser"):
            self._browser.close()
        if hasattr(self, "_pw"):
            self._pw.stop()

    def get_screenshot(self) -> torch.Tensor:
        """Get screenshot as tensor [1, 3, 224, 224]"""
        screenshot_bytes = self.page.screenshot()
        img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        img = img.resize((224, 224))

        # To tensor
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(arr).unsqueeze(0)

    def execute_action(self, action_idx: int, x: float, y: float) -> Tuple[float, bool, str]:
        """
        Execute action and return reward

        Returns:
            reward, done, info_str
        """
        action_name = self.action_names[action_idx]
        px = int(x * self.screen_width)
        py = int(y * self.screen_height)

        reward = 0.0
        done = False
        info = ""

        try:
            if action_name == "click":
                self.page.mouse.click(px, py)
                reward = 0.1  # Small reward for action
                info = f"click({px}, {py})"
                time.sleep(0.3)

            elif action_name == "type":
                self.page.keyboard.type("a")  # Simple for now
                reward = 0.1
                info = "type('a')"

            elif action_name == "navigate":
                # Navigate to random site
                sites = ["google.com", "github.com", "wikipedia.org"]
                site = np.random.choice(sites)
                self.page.goto(f"https://{site}", wait_until="domcontentloaded", timeout=10000)
                reward = 1.0  # Bigger reward for navigation
                info = f"navigate({site})"
                time.sleep(1)

            elif action_name == "scroll":
                self.page.mouse.wheel(0, 300)
                reward = 0.1
                info = "scroll(down)"

            elif action_name == "key":
                self.page.keyboard.press("Enter")
                reward = 0.1
                info = "key(Enter)"

            elif action_name == "done":
                reward = 0.0
                done = True
                info = "done"

            # Bonus for page changes
            if "google" in self.page.url.lower():
                reward += 0.5

        except Exception as e:
            reward = -0.1  # Penalty for errors
            info = f"error: {str(e)[:50]}"

        return reward, done, info


def run_online_learning(
    max_episodes: int = 100,
    max_steps_per_episode: int = 50,
    headless: bool = False,
    save_path: str = "online_lnn.pt"
):
    """Run online learning loop"""
    print("=" * 60)
    print("LNN ONLINE LEARNING")
    print("=" * 60)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = OnlineLNN(num_actions=6, hidden_dim=256, num_layers=4)
    learner = OnlineLearner(model, lr=3e-4, update_freq=10, device=device)

    env = BrowserEnvironment(headless=headless)
    env.setup()

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    try:
        for episode in range(max_episodes):
            # Reset
            model.reset_hidden(1, device)
            env.page.goto("about:blank")
            episode_reward = 0
            episode_info = []

            print(f"\n[Episode {episode + 1}/{max_episodes}]")

            for step in range(max_steps_per_episode):
                # Get state
                state = env.get_screenshot().to(device)

                # Get action
                action, x, y, log_prob, info = model.get_action(state)

                # Execute
                reward, done, action_info = env.execute_action(action, x, y)
                episode_reward += reward

                # Store experience
                learner.store(state, action, log_prob, reward, info["value"], done)

                # Log
                tau_str = ",".join([f"{t:.2f}" for t in info["tau"][:4]])
                print(f"  Step {step+1}: {action_info:20s} | R={reward:+.2f} | τ=[{tau_str}]")

                if done:
                    break

                time.sleep(0.1)

            # Episode summary
            print(f"  Episode reward: {episode_reward:.2f}")
            print(f"  Total updates: {learner.total_updates}")
            learner.episode_rewards.append(episode_reward)

            # Save periodically
            if (episode + 1) % 10 == 0:
                learner.save(save_path)
                avg_reward = np.mean(learner.episode_rewards[-10:])
                print(f"  [Saved] Avg reward (last 10): {avg_reward:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        learner.save(save_path)
        env.cleanup()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Total steps: {learner.total_steps}")
    print(f"Total updates: {learner.total_updates}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    run_online_learning(
        max_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        headless=args.headless
    )
