
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


class SimpleUrbanEnv(gym.Env):
    def __init__(self):
        super(SimpleUrbanEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        # state = [speed, proximity, light_signal, time]
        self.action_space = gym.spaces.Discrete(3)  # [0: slow down, 1: maintain, 2: accelerate]

        self.reset()

    def reset(self, seed=None, options=None):
        self.speed = np.random.uniform(10, 50)
        self.proximity = np.random.uniform(5, 50)
        self.light_signal = np.random.choice([0, 1])  # 0: Red, 1: Green
        self.time = 0
        self.done = False
        self.steps = 0
        return np.array([self.speed, self.proximity, self.light_signal, self.time], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        reward = 0

        if action == 0:
            self.speed = max(0, self.speed - 10)
        elif action == 2:
            self.speed = min(100, self.speed + 10)

        self.time += 1
        self.proximity = max(0, self.proximity - np.random.uniform(1, 5))  # Simulate cars
        self.light_signal = np.random.choice([0, 1])  # Random signal

        if self.light_signal == 0 and self.speed > 0:
            reward -= 5  # Penalty for crossing red light
        if self.proximity < 5:
            reward -= 10  # Penalty for near collision
        if self.speed > 0 and self.light_signal == 1:
            reward += 5  # Positive for legal movement
        if self.speed == 0 and self.light_signal == 0:
            reward += 2  # Good for stopping at red

        if self.time >= 30:
            self.done = True

        obs = np.array([self.speed, self.proximity, self.light_signal, self.time], dtype=np.float32)
        return obs, reward, self.done, False, {}

    def render(self):
        print(f"Speed: {self.speed}, Proximity: {self.proximity}, Light: {self.light_signal}, Time: {self.time}")

def train_agent():
    env = SimpleUrbanEnv()
    monitored_env = Monitor(env)

    model = DQN(
        policy="MlpPolicy",
        env=monitored_env,
        verbose=1,
        learning_rate=0.0005,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        target_update_interval=250,
        train_freq=4,
        learning_starts=100,
        batch_size=32,
        gamma=0.99
    )

    eval_env = Monitor(SimpleUrbanEnv())
    eval_callback = EvalCallback(eval_env, best_model_save_path="./best_model", log_path="./logs", eval_freq=500)

    model.learn(total_timesteps=5000, callback=eval_callback)
    model.save("urban_dqn_agent")
    print("Training complete.")


def test_agent():
    model = DQN.load("urban_dqn_agent")
    env = SimpleUrbanEnv()

    obs, _ = env.reset()
    total_reward = 0

    for _ in range(30):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
        total_reward += reward
        if done:
            break

    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train_agent()
    else:
        test_agent()
