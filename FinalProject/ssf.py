import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
import pickle
import os

# --- ENTORNO PERSONALIZADO ---
class CityEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=(15, 15), render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=max(self.grid_size), shape=(4,), dtype=np.int32
        )
        self.render_mode = render_mode
        self.reset_city_map()
        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size[0]

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def reset_city_map(self):
        self.city_map = np.ones(self.grid_size, dtype=int)

        # --- CALLES (0) ---
        calles = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 5), (0, 6), (0, 10), (0, 11), (0, 12),
            (1, 0), (1, 2), (1, 4), (1, 6), (1, 7), (1, 8), (1, 10), (1, 12), (1, 13), (1, 14),
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (2, 10), (2, 11), (2, 14),
            (3, 0), (3, 2), (3, 6), (3, 8), (3, 10), (3, 12), (3, 13),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 14),
            (5, 2), (5, 4), (5, 6), (5, 7), (5, 8), (5, 10), (5, 12), (5, 13), (5, 14),
            (6, 0), (6, 1), (6, 3), (6, 4), (6, 5), (6, 8), (6, 10), (6, 11), (6, 12),
            (7, 1), (7, 3), (7, 5), (7, 7), (7, 9), (7, 11), (7, 13),
            (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 8), (8, 10), (8, 11), (8, 12), (8, 14),
            (9, 0), (9, 4), (9, 7), (9, 8), (9, 10), (9, 12), (9, 13), (9, 14),
            (10, 0), (10, 1), (10, 3), (10, 4), (10, 5), (10, 6), (10, 8), (10, 10), (10, 11), (10, 12), (10, 14),
            (11, 0), (11, 2), (11, 4), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11), (11, 12), (11, 14),
            (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 8), (12, 10), (12, 11), (12, 12), (12, 14),
            (13, 0), (13, 2), (13, 4), (13, 7), (13, 8), (13, 12), (13, 13), (13, 14),
            (14, 0), (14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 8), (14, 10), (14, 11), (14, 12), (14, 14)
        ]
        for (x, y) in calles:
            self.city_map[x, y] = 0

        # --- OBSTÁCULOS (1) ---
        obstaculos = [
            (0, 4), (0, 8), (0, 14), (2, 9), (2, 12), (3, 4), (3, 7), (3, 14),
            (5, 0), (5, 4), (6, 2), (6, 6), (6, 14), (7, 12),
            (9, 2), (9, 6), (11, 13), (13, 6), (13, 9), (13, 10)
        ]
        for (x, y) in obstaculos:
            self.city_map[x, y] = 1

        # --- EDIFICIOS (3) ---
        edificios = [
            (0, 7), (0, 9), (0, 13), (1, 1), (1, 3), (1, 5), (1, 9), (1, 11),
            (2, 7), (2, 13), (3, 1), (3, 3), (3, 5), (3, 9), (3, 11), (4, 7), (4, 13),
            (5, 1), (5, 3), (5, 5), (5, 9), (5, 11), (6, 7), (6, 9), (6, 13),
            (7, 0), (7, 2), (7, 4), (7, 6), (7, 8), (7, 10), (7, 14),
            (8, 7), (8, 9), (8, 13), (9, 1), (9, 3), (9, 5), (9, 9), (9, 11),
            (10, 7), (10, 9), (10, 13), (11, 1), (11, 3), (11, 5),
            (12, 7), (12, 9), (12, 13), (13, 1), (13, 3), (13, 5), (13, 11),
            (14, 7), (14, 9), (14, 13)
        ]
        for (x, y) in edificios:
            self.city_map[x, y] = 3

    def get_random_position(self):
        valid = np.argwhere(self.city_map == 0)
        return tuple(valid[np.random.choice(len(valid))])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.get_random_position()
        self.goal_pos = self.get_random_position()
        while self.goal_pos == self.agent_pos:
            self.goal_pos = self.get_random_position()

        obs = np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32)
        return obs, {}

    def step(self, action):
        x, y = self.agent_pos
        new_x, new_y = x, y

        if action == 0: new_x = max(x - 1, 0)
        elif action == 1: new_x = min(x + 1, self.grid_size[0] - 1)
        elif action == 2: new_y = max(y - 1, 0)
        elif action == 3: new_y = min(y + 1, self.grid_size[1] - 1)

        new_cell = self.city_map[new_x, new_y]
        reward = -1
        terminated = False

        if new_cell in [1, 3]:
            reward = -10
        else:
            self.agent_pos = (new_x, new_y)
            if self.agent_pos == self.goal_pos:
                reward = 500
                terminated = True

        obs = np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32)
        return obs, reward, terminated, False, {}

    def render(self):
        if self.render_mode != "human": return
        self.screen.fill((255, 255, 255))
        color_map = {0: (200, 200, 200), 1: (0, 0, 0), 3: (150, 0, 0)}
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color_map[self.city_map[i, j]], rect)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        pygame.draw.rect(self.screen, (0, 0, 255),
                         pygame.Rect(self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size,
                                     self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (0, 255, 0),
                         pygame.Rect(self.goal_pos[1] * self.cell_size, self.goal_pos[0] * self.cell_size,
                                     self.cell_size, self.cell_size))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

# --- ENTRENAMIENTO ---
usar_qtable_guardada = False
ruta_qtable = "q_table_dict.pkl"

env = CityEnv()
q_table = {}
episodes = 50000
alpha = 0.3
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.98
min_epsilon = 0.05
max_steps = 300
success_count = 0

for ep in range(episodes):
    obs, _ = env.reset()
    state = tuple(obs)
    done = False
    steps = 0

    while not done and steps < max_steps:
        if state not in q_table:
            q_table[state] = np.full(env.action_space.n, -1.0)

        action = env.action_space.sample() if random.random() < epsilon else np.argmax(q_table[state])
        next_obs, reward, done, _, _ = env.step(action)
        next_state = tuple(next_obs)

        # Recompensa por acercarse o alejarse
        if not done:
            old_dist = abs(state[0] - state[2]) + abs(state[1] - state[3])
            new_dist = abs(next_state[0] - next_state[2]) + abs(next_state[1] - next_state[3])
            if new_dist < old_dist:
                reward += 3
            else:
                reward -= 2

        if next_state not in q_table:
            q_table[next_state] = np.full(env.action_space.n, -1.0)

        q_table[state][action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state
        steps += 1

    if done:
        success_count += 1

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (ep + 1) % 2000 == 0:
        print(f"🎯 Episodio {ep+1} - Éxitos: {success_count}")

with open(ruta_qtable, "wb") as f:
    pickle.dump(q_table, f)
print("✅ Q-table guardada.")

# --- DEMOSTRACIÓN ---
demo_env = CityEnv(render_mode="human")
obs, _ = demo_env.reset()
state = tuple(obs)
done = False
print("🚶‍♂️ Mostrando el camino aprendido...")

# 🔷 Forzar ventana al frente (Windows)
try:
    import ctypes
    import time
    hwnd = pygame.display.get_wm_info()['window']
    time.sleep(0.5)  # Pequeño retardo para asegurarse de que se crea la ventana
    ctypes.windll.user32.ShowWindow(hwnd, 9)      # SW_RESTORE (por si está minimizada)
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    ctypes.windll.user32.BringWindowToTop(hwnd)
except Exception as e:
    print(f"(No se pudo traer al frente la ventana automáticamente: {e})")

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break
    if state in q_table:
        action = np.argmax(q_table[state])
    else:
        action = demo_env.action_space.sample()
    next_obs, _, done, _, _ = demo_env.step(action)
    state = tuple(next_obs)
    demo_env.render()
    pygame.time.delay(200)

demo_env.close()