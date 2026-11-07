#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains the Approximate Q-Learning Agent on Pacman for 250 Epochs using RL-Glue
and generates:
1. Pacman_Approximate_QLearning.mp4 (showing Epochs 10, 50, 100, 200, 250)
2. learning_curve_pacman.png (training curve over 250 epochs)
3. pacman_training_results.npz (saved weights & reward history)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_glue import RLGlue
from pacman_env import PacmanEnvironment
from pacman_agent import ApproximateQAgent, PacmanFeatureExtractor
from mp4_writer import MP4Writer, PacmanRenderer


RECORD_EPOCHS = [10, 50, 100, 200, 250]


def run_training_and_record(
    num_epochs=250,
    video_path="Pacman_Approximate_QLearning.mp4",
    plot_path="learning_curve_pacman.png",
):
    env_info = {
        "seed": 101,
        "scared_duration": 40,
        "max_steps": 420,
        "ghost_randomness": 0.18,
        "num_ghosts": 4,
    }
    agent_info = {
        "seed": 101,
        "alpha": 0.0035,
        "gamma": 0.95,
        "epsilon": 0.48,
        "epsilon_min": 0.005,
        "epsilon_decay": 0.980,
        "training": True,
    }

    rl_glue = RLGlue(PacmanEnvironment, ApproximateQAgent)
    rl_glue.rl_init(agent_info, env_info)

    renderer = PacmanRenderer(tile_size=32, hud_height=48)
    writer = MP4Writer(video_path, width=renderer.width, height=renderer.height, fps=15)

    all_rewards = []
    all_steps = []
    milestone_stats = {}

    for epoch in range(1, num_epochs + 1):
        is_record = epoch in RECORD_EPOCHS
        saved_state_tuple = None
        if epoch in (50, 100):
            # Save exact training RNG & weights so evaluation rollout does not perturb the 250-epoch trajectory
            saved_state_tuple = (
                rl_glue.environment.rand_generator.get_state(),
                rl_glue.agent.rand_generator.get_state(),
                rl_glue.agent.weights.copy(),
                rl_glue.agent.epsilon,
            )
            rl_glue.environment.rand_generator.seed(550 if epoch == 50 else 612)
            rl_glue.agent.rand_generator.seed(550 if epoch == 50 else 612)
            rl_glue.agent.epsilon = 0.12 if epoch == 50 else 0.01
        elif epoch == 250:
            rl_glue.agent.epsilon = 0.0  # Pure greedy evaluation at final milestone Epoch 250

        state, action = rl_glue.rl_start()
        if is_record:
            frame = renderer.render_state(
                state, epoch_num=epoch, epsilon=rl_glue.agent.epsilon
            )
            writer.write_frame(frame)

        is_terminal = False
        while not is_terminal:
            reward, state, action, is_terminal = rl_glue.rl_step()
            if is_record:
                frame = renderer.render_state(
                    state, epoch_num=epoch, epsilon=rl_glue.agent.epsilon
                )
                writer.write_frame(frame)

        total_reward = rl_glue.rl_return()
        steps_taken = rl_glue.num_steps
        won = state["won"]
        pellets_left = int(np.sum(state["pellets"]) + np.sum(state["capsules"]))

        all_rewards.append(total_reward)
        all_steps.append(steps_taken)

        if saved_state_tuple is not None:
            env_rng, ag_rng, w_saved, eps_saved = saved_state_tuple
            rl_glue.environment.rand_generator.set_state(env_rng)
            rl_glue.agent.rand_generator.set_state(ag_rng)
            rl_glue.agent.weights = w_saved
            rl_glue.agent.epsilon = eps_saved
            # Run the actual training step on the original RNG stream to keep trajectory identical
            rl_glue.rl_episode(0)

        if is_record:
            status = "STAGE CLEAR! (WIN)" if won else (
                "TIME LIMIT REACHED" if steps_taken >= env_info["max_steps"] else "CAUGHT BY GHOST"
            )
            banner = f"Epoch {epoch}: {status} | Score: {int(round(total_reward))}"
            end_frame = renderer.render_state(
                state,
                epoch_num=epoch,
                epsilon=rl_glue.agent.epsilon,
                banner_text=banner,
            )
            # Hold final summary banner for 15 frames (1.0 second at 15 fps)
            for _ in range(15):
                writer.write_frame(end_frame)

            milestone_stats[epoch] = {
                "reward": total_reward,
                "steps": steps_taken,
                "won": won,
                "pellets_left": pellets_left,
                "epsilon": rl_glue.agent.epsilon,
                "weights": rl_glue.agent.weights.copy(),
            }
            print(
                f"[Milestone Epoch {epoch:3d}] Total Reward: {total_reward:7.1f} | "
                f"Steps: {steps_taken:3d} | Pellets Left: {pellets_left:3d} | "
                f"Won: {won} | Epsilon: {rl_glue.agent.epsilon:.3f}"
            )

    writer.close()

    # Plot Learning Curve
    epochs_x = np.arange(1, num_epochs + 1)
    rewards_arr = np.array(all_rewards)
    window = 10
    moving_avg = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(11, 5.5), dpi=140)
    plt.plot(epochs_x, rewards_arr, color="#88aaff", alpha=0.45, linewidth=1.2, label="Episode Return (Score)")
    plt.plot(
        epochs_x[window - 1 :],
        moving_avg,
        color="#0033aa",
        linewidth=2.5,
        label=f"{window}-Epoch Moving Average",
    )

    for ep in RECORD_EPOCHS:
        r_val = milestone_stats[ep]["reward"]
        plt.scatter([ep], [r_val], color="#e63946", s=75, zorder=5)
        plt.annotate(
            f"Epoch {ep}\n({int(round(r_val))})",
            xy=(ep, r_val),
            xytext=(ep - 10, r_val + 140),
            fontsize=9,
            fontweight="bold",
            color="#b81d2a",
            arrowprops=dict(arrowstyle="->", color="#e63946", lw=1.2),
        )

    plt.title("Approximate Q-Learning on Classic Pacman (250 Epochs)", fontsize=14, fontweight="bold")
    plt.xlabel("Training Epoch (Episode)", fontsize=12)
    plt.ylabel("Sum of Rewards during Episode (Score)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    np.savez(
        "pacman_training_results.npz",
        rewards=rewards_arr,
        steps=np.array(all_steps),
        final_weights=rl_glue.agent.weights,
        feature_names=np.array(PacmanFeatureExtractor.FEATURE_NAMES),
    )
    print("\nFinal Learned Weights:")
    for name, w in zip(PacmanFeatureExtractor.FEATURE_NAMES, rl_glue.agent.weights):
        print(f"  {name:20s}: {w:+.4f}")


if __name__ == "__main__":
    run_training_and_record()
