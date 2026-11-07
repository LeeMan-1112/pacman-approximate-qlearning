#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Approximate Q-Learning Agent for Pacman (RL-Glue BaseAgent compatible).

Implements Linear Function Approximation over state-action features f(s, a):
    Q(s, a) = w^T f(s, a)
    delta   = R + gamma * max_{a'} Q(s', a') - Q(s, a)
    w      <- w + alpha * delta * f(s, a)
"""

import numpy as np
from agent import BaseAgent
from pacman_env import ACTION_DELTAS, OPPOSITE_ACTION


class PacmanFeatureExtractor:
    """
    Extracts normalized state-action feature vector f(s, a) in R^9
    following standard Pacman Approximate Q-Learning design.
    """

    FEATURE_NAMES = [
        "bias",
        "eats_food",
        "closest_food_inv",
        "active_ghost_dist1",
        "active_ghost_dist2",
        "scared_ghost_hunt",
        "power_capsule_seek",
        "trapped_danger",
        "stop_or_reverse",
    ]

    def __init__(self):
        self.num_features = len(self.FEATURE_NAMES)

    @staticmethod
    def get_legal_actions(state, pos=None):
        if pos is None:
            pos = state["pacman_pos"]
        walls = state["walls"]
        height, width = state["height"], state["width"]
        actions = []
        for a, (dr, dc) in ACTION_DELTAS.items():
            nr = pos[0] + dr
            nc = (pos[1] + dc) % width
            if 0 <= nr < height and not walls[nr, nc]:
                if (nr == 8 and nc == 9) or (nr == 9 and 8 <= nc <= 10):
                    continue
                actions.append(a)
        return actions if actions else [0]

    @staticmethod
    def get_next_pos(state, pos, action):
        legal = PacmanFeatureExtractor.get_legal_actions(state, pos)
        if action not in legal:
            return pos
        dr, dc = ACTION_DELTAS[action]
        nr = pos[0] + dr
        nc = (pos[1] + dc) % state["width"]
        return (nr, nc)

    @staticmethod
    def maze_dist(dist_map, p1, p2):
        return dist_map.get(p1, {}).get(p2, abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))

    def extract(self, state, action):
        pos = state["pacman_pos"]
        next_pos = self.get_next_pos(state, pos, action)
        nr, nc = next_pos
        dist_map = state["dist_map"]
        pellets = state["pellets"]
        capsules = state["capsules"]
        ghosts = state["ghosts"]
        steps = state["steps"]
        max_maze_dist = 32.0

        features = np.zeros(self.num_features, dtype=np.float64)
        # 0. Bias
        features[0] = 1.0

        # Analyze ghosts relative to next_pos
        active_dists = []
        scared_dists = []
        for g in ghosts:
            # Ignore ghosts still waiting inside the ghost house with high release delay
            if steps < g["release_delay"] and (g["pos"][0] == 9 and 8 <= g["pos"][1] <= 10):
                continue
            d = self.maze_dist(dist_map, next_pos, g["pos"])
            if g["scared_timer"] > 1:
                scared_dists.append((d, g["scared_timer"]))
            else:
                active_dists.append(d)

        min_active_dist = min(active_dists) if active_dists else 99.0

        # 3 & 4. Active ghost danger at 1 step and 2 steps
        ghosts_at_1 = sum(1 for d in active_dists if d <= 1)
        ghosts_at_2 = sum(1 for d in active_dists if d == 2)
        features[3] = float(ghosts_at_1)
        features[4] = float(ghosts_at_2) * 0.75

        # 7. Trapped / Corridor pinch danger (3-step pursuit-evasion lookahead)
        if min_active_dist <= 6:
            legal_step1 = self.get_legal_actions(state, next_pos)
            safe_escape_paths = 0
            for a1 in legal_step1:
                pos1 = self.get_next_pos(state, next_pos, a1)
                if any(
                    self.maze_dist(dist_map, pos1, g["pos"]) <= 2
                    for g in ghosts
                    if g["scared_timer"] <= 2 and not (steps < g["release_delay"] and g["pos"][0] == 9)
                ):
                    continue
                for a2 in self.get_legal_actions(state, pos1):
                    pos2 = self.get_next_pos(state, pos1, a2)
                    if pos2 == next_pos:
                        continue
                    if any(
                        self.maze_dist(dist_map, pos2, g["pos"]) <= 3
                        for g in ghosts
                        if g["scared_timer"] <= 3 and not (steps < g["release_delay"] and g["pos"][0] == 9)
                    ):
                        continue
                    for a3 in self.get_legal_actions(state, pos2):
                        pos3 = self.get_next_pos(state, pos2, a3)
                        if pos3 == pos1:
                            continue
                        if all(
                            self.maze_dist(dist_map, pos3, g["pos"]) > 4
                            for g in ghosts
                            if g["scared_timer"] <= 4 and not (steps < g["release_delay"] and g["pos"][0] == 9)
                        ):
                            safe_escape_paths += 1
            if safe_escape_paths == 0:
                features[7] = 1.0
            elif safe_escape_paths <= 2:
                features[7] = 0.35

        is_safe = (ghosts_at_1 == 0) and (features[7] < 0.9)

        # 1. Eats food (pellet or capsule) when safe from ghost trap
        if is_safe and (pellets[nr, nc] or capsules[nr, nc]):
            features[1] = 1.0

        # 2. Inverse distance to closest food (pellet or capsule) - constant slope across map
        d_from_next = dist_map.get(next_pos, {})
        food_positions = np.argwhere(pellets | capsules)
        if is_safe and len(food_positions) > 0:
            min_food_d = min(
                d_from_next.get((int(fr), int(fc)), 30) for fr, fc in food_positions
            )
            features[2] = max(0.0, 1.0 - (min_food_d / 28.0))

        # 5. Scared ghost hunting (only when safe from active ghosts)
        if is_safe and ghosts_at_2 == 0 and scared_dists:
            min_scared_d, timer = min(scared_dists, key=lambda x: x[0])
            if min_scared_d == 0:
                features[5] = 1.0
            elif min_scared_d <= min(timer - 2, 10):
                features[5] = max(0.0, 1.0 - (min_scared_d / 12.0))

        # 6. Power capsule seek (important when active ghosts are nearby)
        cap_positions = np.argwhere(capsules)
        if is_safe and len(cap_positions) > 0:
            min_cap_d = min(
                d_from_next.get((int(cr), int(cc)), 30) for cr, cc in cap_positions
            )
            urgency = 1.0 if min_active_dist <= 6 else 0.45
            features[6] = urgency * max(0.0, 1.0 - (min_cap_d / max_maze_dist))

        # 8. Reverse / Stop penalty (discourages back-and-forth jittering when safe)
        if min_active_dist > 3 and action == OPPOSITE_ACTION.get(state["pacman_dir"], -1):
            features[8] = 1.0

        return features


class ApproximateQAgent(BaseAgent):
    """
    Approximate Q-Learning Agent for Pacman following Coursera RL-Glue interface.
    """

    def __init__(self):
        super().__init__()
        self.extractor = PacmanFeatureExtractor()
        self.weights = np.zeros(self.extractor.num_features, dtype=np.float64)

    def agent_init(self, agent_info={}):
        seed = agent_info.get("seed", 42)
        self.rand_generator = np.random.RandomState(seed)
        self.alpha = agent_info.get("alpha", 0.002)
        self.gamma = agent_info.get("gamma", 0.95)
        self.epsilon = agent_info.get("epsilon", 0.30)
        self.epsilon_min = agent_info.get("epsilon_min", 0.01)
        self.epsilon_decay = agent_info.get("epsilon_decay", 0.982)
        self.training = agent_info.get("training", True)

        if "initial_weights" in agent_info and agent_info["initial_weights"] is not None:
            self.weights = np.array(agent_info["initial_weights"], dtype=np.float64).copy()
        else:
            # Small random symmetry-breaking initialization
            self.weights = self.rand_generator.uniform(
                -0.05, 0.05, size=self.extractor.num_features
            )

        self.prev_state = None
        self.prev_action = None
        self.prev_features = None
        self.epoch_count = 0

    def get_q_value(self, state, action):
        features = self.extractor.extract(state, action)
        return float(np.dot(self.weights, features)), features

    def select_action(self, state):
        legal_actions = self.extractor.get_legal_actions(state)
        if self.training and self.rand_generator.rand() < self.epsilon:
            idx = self.rand_generator.randint(len(legal_actions))
            chosen_action = legal_actions[idx]
            _, chosen_features = self.get_q_value(state, chosen_action)
            return chosen_action, chosen_features

        q_vals = []
        feat_list = []
        for a in legal_actions:
            q, f = self.get_q_value(state, a)
            q_vals.append(q)
            feat_list.append(f)

        q_vals = np.array(q_vals)
        max_q = np.max(q_vals)
        best_indices = np.flatnonzero(np.isclose(q_vals, max_q, atol=1e-6))
        chosen_idx = int(self.rand_generator.choice(best_indices))
        return legal_actions[chosen_idx], feat_list[chosen_idx]

    def agent_start(self, state):
        action, features = self.select_action(state)
        self.prev_state = state
        self.prev_action = action
        self.prev_features = features
        return action

    def agent_step(self, reward, state):
        if self.training and self.prev_features is not None:
            legal_actions = self.extractor.get_legal_actions(state)
            max_next_q = max(self.get_q_value(state, a)[0] for a in legal_actions)
            current_q = float(np.dot(self.weights, self.prev_features))
            # Scale reward by 1/100 for numerical stability in linear Q-learning
            scaled_reward = reward / 100.0
            if self.prev_features[8] > 0.5 and reward < 0:
                # Extra penalty for unnecessary reverse without eating food or escaping ghosts
                scaled_reward -= 0.35
            td_error = scaled_reward + self.gamma * max_next_q - current_q
            td_error = np.clip(td_error, -5.0, 5.0)
            self.weights += self.alpha * td_error * self.prev_features

        action, features = self.select_action(state)
        self.prev_state = state
        self.prev_action = action
        self.prev_features = features
        return action

    def agent_end(self, reward):
        if self.training and self.prev_features is not None:
            current_q = float(np.dot(self.weights, self.prev_features))
            scaled_reward = reward / 100.0
            td_error = scaled_reward - current_q
            td_error = np.clip(td_error, -5.0, 5.0)
            self.weights += self.alpha * td_error * self.prev_features

        self.epoch_count += 1
        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.prev_state = None
        self.prev_action = None
        self.prev_features = None

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        if message == "get_weights":
            return self.weights.copy()
        elif message == "get_epsilon":
            return self.epsilon
        elif isinstance(message, dict) and "set_training" in message:
            self.training = bool(message["set_training"])
        return None
