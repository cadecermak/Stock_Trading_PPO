import gym
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

class StockTradingEnv(gym.Env):
    def __init__(self, stock_data, initial_money):
        super(StockTradingEnv, self).__init__()
        self.stock_data = stock_data
        self.initial_money = initial_money
        self.current_money = initial_money
        self.current_step = 0
        self.previous_action = None
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(4 * 9 * 2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=1.0, shape=(9,), dtype=np.float32)

    def reset(self):
        self.current_money = self.initial_money
        self.current_step = 0
        self.previous_action = None
        return self._next_observation()

    def _next_observation(self):
        frame = np.array([
            self.stock_data.iloc[self.current_step, :].values,
            self.stock_data.iloc[self.current_step + 1, :].values,
            self.stock_data.iloc[self.current_step + 2, :].values,
            self.stock_data.iloc[self.current_step + 3, :].values
        ]).flatten()
        return frame

    def step(self, action):
        step_data = self.stock_data.iloc[self.current_step + 3].values
        current_open_prices = step_data[::2]
        current_close_prices = step_data[1::2]

        # Normalize action to ensure total investment <= 1.0 (100% of available money)
        investments = np.nan_to_num(action)
        total_investment = np.sum(investments)
        if total_investment > 1.0:
            investments /= total_investment

        # Calculate actual investment amount based on available funds
        total_money_to_invest = self.current_money
        invested_amounts = [total_money_to_invest * investments[i] for i in range(len(action))]

        reward = 0
        for i in range(len(action)):
            if current_open_prices[i] > 0:
                profit_from_stock = invested_amounts[i] * (current_close_prices[i] / current_open_prices[i]) - invested_amounts[i]
            else:
                profit_from_stock = 0.0
            self.current_money += profit_from_stock
            reward += profit_from_stock

        self.current_step += 1

        # Check if the current action is the same as the previous action
        if self.previous_action is not None and np.array_equal(action, self.previous_action):
            reward = -100  # Apply penalty for repeating the same action

        # Store the current action as the previous action for the next step
        self.previous_action = action

        done = self.current_step >= len(self.stock_data) - 4

        return self._next_observation(), reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]



tickers = ['SHOP', 'MSFT', 'PEP', 'AMZN', 'TSLA', 'META', 'NVDA', 'JNJ', 'V']
data = yf.download(tickers, start="2023-06-01", end="2024-06-01")
data = data[['Open', 'Adj Close']]
rows = []
for i in range(len(data)):
    row = []
    for ticker in tickers:
        row.append(data['Open'][ticker].iloc[i])
        row.append(data['Adj Close'][ticker].iloc[i])
    rows.append(row)
flattened_data = pd.DataFrame(rows)
flattened_data.columns = [f'{ticker}_Open' for ticker in tickers] + [f'{ticker}_Adj_Close' for ticker in tickers]

env = StockTradingEnv(flattened_data, initial_money=1000)
vec_env = make_vec_env(lambda: env, n_envs=1)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
eval_callback = EvalCallback(vec_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=10000, deterministic=True, render=False)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/', name_prefix='ppo_model')

policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
    activation_fn=nn.ReLU
)

model = PPO('MlpPolicy', vec_env, learning_rate=0.0001, n_steps=2048, batch_size=128, n_epochs=20, gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=0.02, verbose=1, tensorboard_log="./ppo_stock_trading_tensorboard/", policy_kwargs=policy_kwargs)

model.learn(total_timesteps=1000000, callback=[eval_callback, checkpoint_callback])
