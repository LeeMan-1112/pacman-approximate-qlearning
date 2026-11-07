#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pacman Environment for RL-Glue (Course 3 & 4 Assignment Structure).
Implements classic Pacman rules:
- Classic Arcade Maze with walls (#), pellets (.), power pellets/energizers (o), and ghost house.
- 4 Ghosts with classic personalities:
  * Blinky (Red): Direct chaser
  * Pinky (Pink): Ambush (targets ahead of Pacman)
  * Inky (Cyan): Flanker
  * Clyde (Orange): Chases when far, scatters to corner when close
- Standard scoring:
  * Pellet: +10
  * Power Pellet (Energizer): +50 (sets ghosts to Scared state for 35 steps)
  * Eating Scared Ghost: +200
  * Time-step penalty: -1
  * Eaten by Active Ghost: -500 (Terminal)
  * Clearing all Pellets: +500 (Terminal - WIN)
"""

from collections import deque
import numpy as np
from environment import BaseEnvironment


CLASSIC_MAZE = [
    "###################",
    "#o.......#.......o#",
    "#.##.###.#.###.##.#",
    "#.................#",
    "#.##.#.#####.#.##.#",
    "#....#...#...#....#",
    "####.###.#.###.####",
    "   #.#.......#.#   ",
    "####.#.##-##.#.####",
    ".......#GGG#.......",
    "####.#.#####.#.####",
    "   #.#.......#.#   ",
    "####.#.#####.#.####",
    "#........#........#",
    "#.##.###.#.###.##.#",
    "#o.#.....P.....#.o#",
    "##.#.#.#####.#.#.##",
    "#....#...#...#....#",
    "#.######.#.######.#",
    "#.................#",
    "###################",
]

# Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
ACTION_DELTAS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
}

OPPOSITE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2}


class PacmanEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.maze_lines = CLASSIC_MAZE
        self.height = len(self.maze_lines)
        self.width = len(self.maze_lines[0])
        self.rand_generator = np.random.RandomState(0)
        self._parse_static_maze()
        self._precompute_all_pairs_bfs()

    def _parse_static_maze(self):
        self.walls = np.zeros((self.height, self.width), dtype=bool)
        self.initial_pellets = np.zeros((self.height, self.width), dtype=bool)
        self.initial_capsules = np.zeros((self.height, self.width), dtype=bool)
        self.ghost_spawns = []
        self.pacman_spawn = (15, 9)
        self.ghost_door = (8, 9)

        for r, row in enumerate(self.maze_lines):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.walls[r, c] = True
                elif ch == ".":
                    self.initial_pellets[r, c] = True
                elif ch == "o":
                    self.initial_capsules[r, c] = True
                elif ch == "P":
                    self.pacman_spawn = (r, c)
                elif ch == "G":
                    self.ghost_spawns.append((r, c))
                elif ch == "-":
                    self.ghost_door = (r, c)

        # Ensure 4 ghosts
        while len(self.ghost_spawns) < 4:
            self.ghost_spawns.append((9, 9))
        # Also add a 4th spawn right above the ghost house
        if len(self.ghost_spawns) == 4:
            self.ghost_spawns = [
                (7, 9),   # Blinky starts outside above the house
                (9, 8),   # Pinky
                (9, 9),   # Inky
                (9, 10),  # Clyde
            ]

    def _neighbors(self, r, c, allow_ghost_house=True):
        res = []
        for a, (dr, dc) in ACTION_DELTAS.items():
            nr = r + dr
            nc = (c + dc) % self.width
            if 0 <= nr < self.height and not self.walls[nr, nc]:
                if not allow_ghost_house and (nr == 9 and 8 <= nc <= 10):
                    continue
                res.append((a, (nr, nc)))
        return res

    def _precompute_all_pairs_bfs(self):
        """Precompute exact shortest maze distances between all walkable cells."""
        self.dist_map = {}
        walkable = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if not self.walls[r, c]
        ]
        for start in walkable:
            d = {start: 0}
            q = deque([start])
            while q:
                curr = q.popleft()
                cd = d[curr]
                for _, nxt in self._neighbors(curr[0], curr[1], allow_ghost_house=True):
                    if nxt not in d:
                        d[nxt] = cd + 1
                        q.append(nxt)
            self.dist_map[start] = d

    def get_maze_distance(self, pos1, pos2):
        return self.dist_map.get(pos1, {}).get(pos2, abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    def env_init(self, env_info={}):
        seed = env_info.get("seed", 0)
        self.rand_generator = np.random.RandomState(seed)
        self.scared_duration = env_info.get("scared_duration", 35)
        self.max_steps = env_info.get("max_steps", 350)
        self.ghost_randomness = env_info.get("ghost_randomness", 0.15)
        self.num_ghosts = env_info.get("num_ghosts", 4)

    def env_start(self):
        self.pellets = self.initial_pellets.copy()
        self.capsules = self.initial_capsules.copy()
        self.pacman_pos = self.pacman_spawn
        self.pacman_dir = 3  # Facing RIGHT initially
        self.score = 0.0
        self.steps = 0
        self.won = False
        self.ghosts_eaten_in_power = 0

        self.ghosts = []
        ghost_names = ["Blinky", "Pinky", "Inky", "Clyde"]
        for i in range(self.num_ghosts):
            self.ghosts.append({
                "name": ghost_names[i],
                "pos": self.ghost_spawns[i],
                "spawn": self.ghost_spawns[i],
                "dir": 0,
                "scared_timer": 0,
                "release_delay": i * 3,
            })

        state = self.get_state_dict()
        self.reward_obs_term = (0.0, state, False)
        return state

    def get_state_dict(self):
        return {
            "pacman_pos": self.pacman_pos,
            "pacman_dir": self.pacman_dir,
            "pellets": self.pellets.copy(),
            "capsules": self.capsules.copy(),
            "ghosts": [g.copy() for g in self.ghosts],
            "score": self.score,
            "steps": self.steps,
            "won": self.won,
            "walls": self.walls,
            "dist_map": self.dist_map,
            "height": self.height,
            "width": self.width,
        }

    def get_legal_actions(self, pos=None):
        if pos is None:
            pos = self.pacman_pos
        actions = []
        for a, (dr, dc) in ACTION_DELTAS.items():
            nr = pos[0] + dr
            nc = (pos[1] + dc) % self.width
            if 0 <= nr < self.height and not self.walls[nr, nc]:
                # Pacman cannot enter the ghost house
                if (nr == 8 and nc == 9) or (nr == 9 and 8 <= nc <= 10):
                    continue
                actions.append(a)
        return actions if actions else [0]

    def _move_pacman(self, action):
        legal = self.get_legal_actions(self.pacman_pos)
        if action not in legal:
            # Try continuing in previous direction if valid, else stays put
            if self.pacman_dir in legal:
                action = self.pacman_dir
            else:
                return self.pacman_pos
        self.pacman_dir = action
        dr, dc = ACTION_DELTAS[action]
        nr = self.pacman_pos[0] + dr
        nc = (self.pacman_pos[1] + dc) % self.width
        self.pacman_pos = (nr, nc)
        return self.pacman_pos

    def _ghost_target(self, ghost_idx):
        pr, pc = self.pacman_pos
        pdir = self.pacman_dir
        dr, dc = ACTION_DELTAS.get(pdir, (0, 0))
        if ghost_idx == 0:  # Blinky: directly targets Pacman
            return (pr, pc)
        elif ghost_idx == 1:  # Pinky: targets 4 tiles ahead of Pacman
            tr = np.clip(pr + 4 * dr, 1, self.height - 2)
            tc = (pc + 4 * dc) % self.width
            return (int(tr), int(tc))
        elif ghost_idx == 2:  # Inky: flanks using Blinky's position
            br, bc = self.ghosts[0]["pos"]
            ar, ac = pr + 2 * dr, pc + 2 * dc
            tr = int(np.clip(2 * ar - br, 1, self.height - 2))
            tc = int(np.clip(2 * ac - bc, 1, self.width - 2))
            return (tr, tc)
        else:  # Clyde: chases when distance > 6, else scatters to bottom-left corner
            gpos = self.ghosts[ghost_idx]["pos"]
            if self.get_maze_distance(gpos, self.pacman_pos) > 6:
                return (pr, pc)
            return (self.height - 2, 1)

    def _move_ghost(self, idx, g):
        if self.steps < g["release_delay"]:
            return
        # If scared, ghosts move slightly slower (skip move every 3rd step)
        if g["scared_timer"] > 0 and (self.steps % 3 == 0):
            return

        r, c = g["pos"]
        # If inside ghost house, move towards exit (7, 9)
        if (r == 9 and 8 <= c <= 10) or (r == 8 and c == 9):
            target = (7, 9)
            neighbors = self._neighbors(r, c, allow_ghost_house=True)
            best_a, best_pos = min(
                neighbors, key=lambda item: self.get_maze_distance(item[1], target)
            )
            g["pos"] = best_pos
            g["dir"] = best_a
            return

        # Outside ghost house: cannot re-enter ghost house
        neighbors = [
            (a, pos)
            for a, pos in self._neighbors(r, c, allow_ghost_house=False)
            if pos != (8, 9)
        ]
        if not neighbors:
            neighbors = self._neighbors(r, c, allow_ghost_house=True)

        # Standard Pacman rule: ghosts do not immediately reverse unless no other choice
        non_reverse = [
            (a, pos) for a, pos in neighbors if a != OPPOSITE_ACTION.get(g["dir"], -1)
        ]
        candidates = non_reverse if non_reverse else neighbors

        if g["scared_timer"] > 0:
            # Scared ghosts flee from Pacman with some randomness
            if self.rand_generator.rand() < 0.35:
                choice_idx = self.rand_generator.randint(len(candidates))
                chosen_a, chosen_pos = candidates[choice_idx]
            else:
                chosen_a, chosen_pos = max(
                    candidates,
                    key=lambda item: self.get_maze_distance(item[1], self.pacman_pos),
                )
        else:
            if self.rand_generator.rand() < self.ghost_randomness:
                choice_idx = self.rand_generator.randint(len(candidates))
                chosen_a, chosen_pos = candidates[choice_idx]
            else:
                target = self._ghost_target(idx)
                chosen_a, chosen_pos = min(
                    candidates,
                    key=lambda item: (
                        self.get_maze_distance(item[1], target)
                        if target in self.dist_map.get(item[1], {})
                        else abs(item[1][0] - target[0]) + abs(item[1][1] - target[1])
                    ),
                )

        g["pos"] = chosen_pos
        g["dir"] = chosen_a

    def _check_ghost_collisions(self):
        """Returns (reward_delta, is_terminal_death)."""
        reward_delta = 0.0
        for g in self.ghosts:
            if g["pos"] == self.pacman_pos:
                if g["scared_timer"] > 0:
                    # Eat scared ghost!
                    self.ghosts_eaten_in_power += 1
                    reward_delta += 200.0
                    g["pos"] = (9, 9)
                    g["scared_timer"] = 0
                    g["release_delay"] = self.steps + 8
                else:
                    # Eaten by active ghost!
                    return -500.0, True
        return reward_delta, False

    def env_step(self, action):
        self.steps += 1
        reward = -1.0  # Living step penalty
        is_terminal = False

        # 1. Move Pacman
        pr, pc = self._move_pacman(int(action))

        # 2. Check immediate collision with ghosts
        r_col, dead = self._check_ghost_collisions()
        reward += r_col
        if dead:
            self.score += reward
            state = self.get_state_dict()
            self.reward_obs_term = (reward, state, True)
            return self.reward_obs_term

        # 3. Eat Pellet or Power Pellet (Energizer)
        if self.pellets[pr, pc]:
            self.pellets[pr, pc] = False
            reward += 10.0

        if self.capsules[pr, pc]:
            self.capsules[pr, pc] = False
            reward += 50.0
            self.ghosts_eaten_in_power = 0
            for g in self.ghosts:
                g["scared_timer"] = self.scared_duration

        # 4. Check win condition (all pellets and capsules eaten)
        if not np.any(self.pellets) and not np.any(self.capsules):
            reward += 500.0
            self.won = True
            self.score += reward
            state = self.get_state_dict()
            self.reward_obs_term = (reward, state, True)
            return self.reward_obs_term

        # 5. Move Ghosts & decrement scared timers
        for idx, g in enumerate(self.ghosts):
            self._move_ghost(idx, g)
            if g["scared_timer"] > 0:
                g["scared_timer"] -= 1

        # 6. Check post-ghost-move collisions
        r_col2, dead2 = self._check_ghost_collisions()
        reward += r_col2
        if dead2:
            is_terminal = True

        if self.steps >= self.max_steps:
            is_terminal = True

        self.score += reward
        state = self.get_state_dict()
        self.reward_obs_term = (reward, state, is_terminal)
        return self.reward_obs_term

    def env_cleanup(self):
        pass

    def env_message(self, message):
        if message == "get_score":
            return self.score
        elif message == "is_won":
            return self.won
        return None
