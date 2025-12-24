#!/usr/bin/env python3
"""
LNN Distributed Learning (Swarm/Botnet Style)

여러 브라우저 에이전트가 동시에 학습하고 가중치 공유
- 각 노드: 독립적으로 브라우저 조작 & 학습
- 중앙 서버: 가중치 수집 & 평균화 (Federated Averaging)
- 주기적 동기화: 모든 노드가 최신 가중치 공유

Architecture:
  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ Agent 1 │     │ Agent 2 │     │ Agent N │
  │ Browser │     │ Browser │     │ Browser │
  │   LNN   │     │   LNN   │     │   LNN   │
  └────┬────┘     └────┬────┘     └────┬────┘
       │               │               │
       └───────────────┼───────────────┘
                       │
                ┌──────┴──────┐
                │   Central   │
                │   Server    │
                │  (Weights)  │
                └─────────────┘
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Manager
import numpy as np
import time
import os
from typing import Dict, List, Optional
from collections import deque
import threading
import socket
import pickle
import struct

from online_learner import OnlineLNN, OnlineLearner, BrowserEnvironment


class WeightServer:
    """
    중앙 가중치 서버

    - 각 에이전트로부터 가중치 수집
    - Federated Averaging 수행
    - 평균 가중치 배포
    """

    def __init__(self, model: nn.Module, port: int = 5555):
        self.port = port
        self.global_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.collected_weights: List[Dict] = []
        self.num_agents = 0
        self.sync_count = 0
        self.lock = threading.Lock()

        # Stats
        self.total_episodes = 0
        self.total_rewards = []

    def receive_weights(self, agent_id: int, weights: Dict, stats: Dict):
        """에이전트로부터 가중치 수신"""
        with self.lock:
            self.collected_weights.append(weights)
            self.total_episodes += stats.get("episodes", 0)
            if "reward" in stats:
                self.total_rewards.append(stats["reward"])

            print(f"  [Server] Received from Agent {agent_id} | "
                  f"Total collected: {len(self.collected_weights)}")

    def federated_average(self, min_agents: int = 2):
        """Federated Averaging 수행"""
        with self.lock:
            if len(self.collected_weights) < min_agents:
                return False

            print(f"  [Server] Federated Averaging ({len(self.collected_weights)} agents)...")

            # Average weights
            avg_weights = {}
            for key in self.global_weights.keys():
                stacked = torch.stack([w[key].float() for w in self.collected_weights])
                avg_weights[key] = stacked.mean(dim=0)

            self.global_weights = avg_weights
            self.collected_weights = []
            self.sync_count += 1

            avg_reward = np.mean(self.total_rewards[-10:]) if self.total_rewards else 0
            print(f"  [Server] Sync #{self.sync_count} complete | "
                  f"Avg reward: {avg_reward:.2f}")

            return True

    def get_global_weights(self) -> Dict:
        """현재 글로벌 가중치 반환"""
        with self.lock:
            return {k: v.clone() for k, v in self.global_weights.items()}


class DistributedAgent:
    """
    분산 학습 에이전트

    독립적으로 브라우저 조작하며 학습
    주기적으로 서버와 가중치 동기화
    """

    def __init__(
        self,
        agent_id: int,
        server: WeightServer,
        device: str = "cuda",
        sync_freq: int = 5  # Sync every N episodes
    ):
        self.agent_id = agent_id
        self.server = server
        self.device = device
        self.sync_freq = sync_freq

        # Create model and learner
        self.model = OnlineLNN(num_actions=6, hidden_dim=256, num_layers=4).to(device)
        self.learner = OnlineLearner(self.model, lr=3e-4, device=device)

        # Load global weights
        self.sync_from_server()

        # Stats
        self.episodes_done = 0
        self.local_rewards = []

    def sync_from_server(self):
        """서버에서 가중치 가져오기"""
        global_weights = self.server.get_global_weights()
        self.model.load_state_dict({k: v.to(self.device) for k, v in global_weights.items()})
        print(f"  [Agent {self.agent_id}] Synced from server")

    def sync_to_server(self):
        """서버로 가중치 전송"""
        weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
        stats = {
            "episodes": self.episodes_done,
            "reward": np.mean(self.local_rewards[-5:]) if self.local_rewards else 0
        }
        self.server.receive_weights(self.agent_id, weights, stats)

    def run_episode(self, env: BrowserEnvironment, max_steps: int = 20) -> float:
        """단일 에피소드 실행"""
        self.model.reset_hidden(1, self.device)
        env.page.goto("about:blank")

        episode_reward = 0

        for step in range(max_steps):
            # Get state
            state = env.get_screenshot().to(self.device)

            # Get action
            action, x, y, log_prob, info = self.model.get_action(state)

            # Execute
            reward, done, action_info = env.execute_action(action, x, y)
            episode_reward += reward

            # Store & learn
            self.learner.store(state, action, log_prob, reward, info["value"], done)

            if done:
                break

            time.sleep(0.05)

        self.episodes_done += 1
        self.local_rewards.append(episode_reward)

        return episode_reward


def run_agent_process(
    agent_id: int,
    weight_queue: Queue,
    result_queue: Queue,
    num_episodes: int,
    sync_freq: int,
    headless: bool
):
    """
    에이전트 프로세스 (멀티프로세싱용)
    """
    print(f"[Agent {agent_id}] Starting...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = OnlineLNN(num_actions=6, hidden_dim=256, num_layers=4).to(device)
    learner = OnlineLearner(model, lr=3e-4, device=device)

    # Setup browser
    env = BrowserEnvironment(headless=headless)
    env.setup()

    episodes_done = 0
    local_rewards = []

    try:
        for episode in range(num_episodes):
            # Check for weight updates from server
            while not weight_queue.empty():
                try:
                    new_weights = weight_queue.get_nowait()
                    model.load_state_dict({k: v.to(device) for k, v in new_weights.items()})
                    print(f"[Agent {agent_id}] Received new weights")
                except:
                    break

            # Run episode
            model.reset_hidden(1, device)
            env.page.goto("about:blank")
            episode_reward = 0

            for step in range(20):
                state = env.get_screenshot().to(device)
                action, x, y, log_prob, info = model.get_action(state)
                reward, done, _ = env.execute_action(action, x, y)
                episode_reward += reward
                learner.store(state, action, log_prob, reward, info["value"], done)

                if done:
                    break
                time.sleep(0.05)

            episodes_done += 1
            local_rewards.append(episode_reward)

            # Send results to server periodically
            if episodes_done % sync_freq == 0:
                weights = {k: v.cpu() for k, v in model.state_dict().items()}
                result_queue.put({
                    "agent_id": agent_id,
                    "weights": weights,
                    "reward": np.mean(local_rewards[-sync_freq:]),
                    "episodes": episodes_done
                })
                print(f"[Agent {agent_id}] Episode {episodes_done} | "
                      f"Reward: {episode_reward:.2f} | Sent to server")

    except KeyboardInterrupt:
        print(f"[Agent {agent_id}] Interrupted")
    finally:
        env.cleanup()

    print(f"[Agent {agent_id}] Done. Total episodes: {episodes_done}")


def run_server_process(
    result_queue: Queue,
    weight_queues: List[Queue],
    model_template: nn.Module,
    min_agents_for_sync: int = 2
):
    """
    서버 프로세스 (Federated Averaging)
    """
    print("[Server] Starting...")

    global_weights = {k: v.cpu().clone() for k, v in model_template.state_dict().items()}
    collected = []
    sync_count = 0

    while True:
        try:
            # Collect results
            result = result_queue.get(timeout=1)
            collected.append(result)
            print(f"[Server] Received from Agent {result['agent_id']} | "
                  f"Reward: {result['reward']:.2f}")

            # Federated average when enough agents reported
            if len(collected) >= min_agents_for_sync:
                print(f"[Server] Averaging {len(collected)} agents...")

                # Average weights
                avg_weights = {}
                for key in global_weights.keys():
                    stacked = torch.stack([c["weights"][key].float() for c in collected])
                    avg_weights[key] = stacked.mean(dim=0)

                global_weights = avg_weights
                collected = []
                sync_count += 1

                # Broadcast to all agents
                for q in weight_queues:
                    q.put(global_weights)

                print(f"[Server] Sync #{sync_count} complete, broadcasted to {len(weight_queues)} agents")

        except:
            continue


def run_distributed_training(
    num_agents: int = 3,
    num_episodes: int = 50,
    sync_freq: int = 5,
    headless: bool = True
):
    """
    분산 학습 실행

    Args:
        num_agents: 에이전트 수
        num_episodes: 에이전트당 에피소드 수
        sync_freq: 동기화 주기
        headless: 브라우저 숨김 여부
    """
    print("=" * 60)
    print(f"LNN DISTRIBUTED LEARNING")
    print(f"  Agents: {num_agents}")
    print(f"  Episodes per agent: {num_episodes}")
    print(f"  Sync frequency: every {sync_freq} episodes")
    print("=" * 60)

    mp.set_start_method('spawn', force=True)

    # Create queues
    result_queue = Queue()
    weight_queues = [Queue() for _ in range(num_agents)]

    # Template model for weight initialization
    model_template = OnlineLNN(num_actions=6, hidden_dim=256, num_layers=4)

    # Start server process
    server_proc = Process(
        target=run_server_process,
        args=(result_queue, weight_queues, model_template, min(2, num_agents))
    )
    server_proc.start()

    # Start agent processes
    agent_procs = []
    for i in range(num_agents):
        p = Process(
            target=run_agent_process,
            args=(i, weight_queues[i], result_queue, num_episodes, sync_freq, headless)
        )
        p.start()
        agent_procs.append(p)
        time.sleep(1)  # Stagger starts

    # Wait for agents
    for p in agent_procs:
        p.join()

    # Stop server
    server_proc.terminate()

    print("\n" + "=" * 60)
    print("DISTRIBUTED TRAINING COMPLETE")
    print("=" * 60)


class SimpleSwarm:
    """
    간단한 스웜 학습 (싱글 브라우저, 멀티 탭)

    하나의 브라우저에서 여러 탭으로 학습
    """

    def __init__(
        self,
        num_agents: int = 3,
        device: str = "cuda"
    ):
        self.num_agents = num_agents
        self.device = device

        # Shared model
        self.model = OnlineLNN(num_actions=6, hidden_dim=256, num_layers=4).to(device)
        self.learner = OnlineLearner(self.model, lr=3e-4, device=device)

        # Single browser, multiple pages
        self._pw = None
        self._browser = None
        self.pages: List = []

    def setup(self, headless: bool = True):
        """Setup browser with multiple tabs"""
        from playwright.sync_api import sync_playwright

        print(f"Setting up browser with {self.num_agents} tabs...")
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)

        for i in range(self.num_agents):
            context = self._browser.new_context(
                viewport={"width": 1280, "height": 720}
            )
            page = context.new_page()
            self.pages.append(page)
            print(f"  Tab {i} ready")

    def cleanup(self):
        """Cleanup browser"""
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def get_screenshot(self, page) -> torch.Tensor:
        """Get screenshot from page as tensor"""
        from PIL import Image
        import io
        screenshot_bytes = page.screenshot()
        img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).unsqueeze(0)

    def execute_action(self, page, action_idx: int, x: float, y: float):
        """Execute action on page"""
        action_names = ["click", "type", "navigate", "scroll", "key", "done"]
        action_name = action_names[action_idx]
        px, py = int(x * 1280), int(y * 720)
        reward, done = 0.0, False

        try:
            if action_name == "click":
                page.mouse.click(px, py)
                reward = 0.1
            elif action_name == "type":
                page.keyboard.type("a")
                reward = 0.1
            elif action_name == "navigate":
                sites = ["google.com", "github.com", "wikipedia.org"]
                site = np.random.choice(sites)
                page.goto(f"https://{site}", wait_until="domcontentloaded", timeout=10000)
                reward = 1.0
            elif action_name == "scroll":
                page.mouse.wheel(0, 300)
                reward = 0.1
            elif action_name == "key":
                page.keyboard.press("Enter")
                reward = 0.1
            elif action_name == "done":
                done = True

            if "google" in page.url.lower():
                reward += 0.5
        except:
            reward = -0.1

        return reward, done

    def run_round_robin(self, num_rounds: int = 10, steps_per_agent: int = 5):
        """
        라운드 로빈 방식으로 학습

        각 라운드마다 모든 탭이 순서대로 몇 스텝씩 실행
        """
        print(f"\n[Swarm] Round-robin training: {num_rounds} rounds, {len(self.pages)} tabs")

        total_rewards = []

        for round_idx in range(num_rounds):
            round_reward = 0

            for agent_idx, page in enumerate(self.pages):
                # Reset if first round
                if round_idx == 0:
                    page.goto("about:blank")
                    self.model.reset_hidden(1, self.device)

                # Run steps
                for step in range(steps_per_agent):
                    state = self.get_screenshot(page).to(self.device)
                    action, x, y, log_prob, info = self.model.get_action(state)
                    reward, done = self.execute_action(page, action, x, y)
                    round_reward += reward

                    self.learner.store(state, action, log_prob, reward, info["value"], done)

                    if done:
                        page.goto("about:blank")
                        break

                    time.sleep(0.05)

            total_rewards.append(round_reward)

            # Get tau info
            tau_str = ""
            if self.model.hidden_states:
                taus = [h.abs().mean().item() for h in self.model.hidden_states]
                tau_str = f" | τ=[{','.join([f'{t:.2f}' for t in taus])}]"

            print(f"  Round {round_idx+1}/{num_rounds} | "
                  f"Reward: {round_reward:.2f} | "
                  f"Updates: {self.learner.total_updates}{tau_str}")

        return total_rewards

    def save(self, path: str):
        self.learner.save(path)


def run_simple_swarm(
    num_agents: int = 2,
    num_rounds: int = 20,
    steps_per_agent: int = 5,
    headless: bool = False
):
    """간단한 스웜 학습 실행"""
    print("=" * 60)
    print(f"LNN SIMPLE SWARM")
    print(f"  Agents: {num_agents}")
    print(f"  Rounds: {num_rounds}")
    print("=" * 60)

    swarm = SimpleSwarm(num_agents=num_agents)

    try:
        swarm.setup(headless=headless)
        rewards = swarm.run_round_robin(num_rounds, steps_per_agent)
        swarm.save("swarm_lnn.pt")

        print(f"\nTotal reward: {sum(rewards):.2f}")
        print(f"Avg reward per round: {np.mean(rewards):.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        swarm.cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["distributed", "swarm"], default="swarm")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.mode == "distributed":
        run_distributed_training(
            num_agents=args.agents,
            num_episodes=args.rounds,
            headless=args.headless
        )
    else:
        run_simple_swarm(
            num_agents=args.agents,
            num_rounds=args.rounds,
            headless=args.headless
        )
