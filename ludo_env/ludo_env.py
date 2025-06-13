# ludo_env/ludo_env.py

import pygame
from gym import spaces
import gym
import numpy as np

class LudoEnv(gym.Env):
    def __init__(self):
        super(LudoEnv, self).__init__()

        self.num_players = 4
        self.tokens_per_player = 4
        self.total_tokens = self.num_players * self.tokens_per_player

        self.observation_space = spaces.Box(
            low=0,
            high=57,
            shape=(self.num_players, self.tokens_per_player),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(self.tokens_per_player)

        self.safe_positions = {1, 9, 14, 22, 27, 35, 40, 48}
        self._pygame_screen = None  # For reusing the screen across frames

        self.reset()

    def reset(self):
        self.state = np.zeros((self.num_players, self.tokens_per_player), dtype=np.int32)
        self.current_player = 0
        self.done_flags = [False] * self.num_players
        return self.state.copy()

    def _opponent_move(self, player_idx):
        dice_roll = np.random.randint(1, 7)
        player_tokens = self.state[player_idx]

        best_token = -1
        best_score = -float('inf')

        for token_id in range(self.tokens_per_player):
            pos = player_tokens[token_id]
            if pos >= 57:
                continue

            next_pos = min(pos + dice_roll, 57)
            score = next_pos

            if next_pos in self.safe_positions:
                score += 5

            for t in range(self.tokens_per_player):
                if self.state[self.current_player][t] == next_pos and next_pos not in self.safe_positions:
                    score += 10

            if score > best_score:
                best_score = score
                best_token = token_id

        if best_token != -1:
            next_pos = min(player_tokens[best_token] + dice_roll, 57)
            player_tokens[best_token] = next_pos

            for t in range(self.tokens_per_player):
                if self.state[self.current_player][t] == next_pos and next_pos not in self.safe_positions:
                    self.state[self.current_player][t] = 0

    def step(self, action):
        reward = 0
        dice_roll = np.random.randint(1, 7)
        current_tokens = self.state[self.current_player]

        if current_tokens[action] >= 57:
            return self.state.copy(), reward, all(self.done_flags), {"dice_roll": dice_roll}

        new_pos = min(current_tokens[action] + dice_roll, 57)
        current_tokens[action] = new_pos

        for p in range(self.num_players):
            if p == self.current_player:
                continue
            for t in range(self.tokens_per_player):
                if self.state[p][t] == new_pos and new_pos not in self.safe_positions:
                    self.state[p][t] = 0
                    reward += 0.5

        if new_pos in self.safe_positions:
            reward += 0.2

        if new_pos == 57:
            reward += 1

        if np.all(current_tokens == 57):
            self.done_flags[self.current_player] = True

        for p in range(1, self.num_players):
            opponent = (self.current_player + p) % self.num_players
            if not self.done_flags[opponent]:
                self._opponent_move(opponent)

        for _ in range(self.num_players):
            self.current_player = (self.current_player + 1) % self.num_players
            if not self.done_flags[self.current_player]:
                break

        done = all(self.done_flags)

        return self.state.copy(), reward, done, {"dice_roll": dice_roll}

    def render(self, mode='human'):
        print(f"\nCurrent Player: {self.current_player}")
        for i, tokens in enumerate(self.state):
            print(f"Player {i}: {tokens}")

    def render_pygame(self, screen=None):
        tile_size = 60
        margin = 5
        board_size = self.tokens_per_player  # simple 4x4 layout
        screen_size = (board_size * tile_size, self.num_players * tile_size)

        if self._pygame_screen is None:
            pygame.init()
            self._pygame_screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption("Ludo 2D")

        screen = self._pygame_screen
        screen.fill((255, 255, 255))  # white background

        colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 165, 0)]  # Red, Green, Blue, Orange
        font = pygame.font.SysFont(None, 24)

        for player in range(self.num_players):
            for token in range(self.tokens_per_player):
                x = token * tile_size
                y = player * tile_size
                rect = pygame.Rect(x + margin, y + margin, tile_size - 2*margin, tile_size - 2*margin)
                pygame.draw.rect(screen, colors[player], rect)
                token_value = self.state[player][token]
                txt = font.render(str(token_value), True, (255, 255, 255))
                screen.blit(txt, (x + tile_size // 3, y + tile_size // 3))

        pygame.display.flip()
