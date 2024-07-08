import gym
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Define the StockTradingEnv class (same as before)

class StockTradingEnv(gym.Env):
    def __init__(self, stock_data, initial_money):
        super(StockTradingEnv, self).__init__()
        self.stock_data = stock_data
        self.initial_money = initial_money
        self.current_money = initial_money
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(4 * 9 * 2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=1.0, shape=(9,), dtype=np.float32)  # Action space in percentages

    def reset(self):
        self.current_money = self.initial_money
        self.current_step = 0
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
            investments /= total_investment  # Normalize investments to sum up to 1.0

        # Calculate actual investment amount based on available funds
        total_money_to_invest = self.current_money
        invested_amounts = [total_money_to_invest * investments[i] for i in range(len(action))]
        
        # Print investments for the current day
        for i, ticker in enumerate(tickers):
            if action[i] > 0:
                print(f"Invested {invested_amounts[i]:.2f} units of {ticker} stock.")
                print(current_open_prices[i])
                print(current_close_prices[i])

        reward = 0
        for i in range(len(action)):
            if current_open_prices[i] > 0:
                profit_from_stock = invested_amounts[i] * (current_close_prices[i] / current_open_prices[i]) - invested_amounts[i]
            else:
                profit_from_stock = 0.0
            self.current_money += profit_from_stock
            reward += profit_from_stock

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 3
        if done:
            print(f"Final Money: ${self.current_money:.2f}")
            return self.reset(), reward, done, {}

        return self._next_observation(), reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

# Function to download stock data
def download_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
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
    
    return flattened_data

# Function to load the best model
def load_best_model(model_path):
    return PPO.load(model_path)

# Function to run the trained model and show the stock picks with amounts
def run_model_and_show_picks(model, stock_data, initial_money):
    env = StockTradingEnv(stock_data, initial_money)
    obs = env.reset()
    done = False
    total_investments = np.zeros(len(tickers))
    while not done:
        print(f"Day {env.current_step + 1}:")
        print(f"Total current money: ${env.current_money:.2f}")
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
       
       
        
        # Print current total money after investments
        if not done:
            print(f"Total current money: ${env.current_money:.2f}")
        print("-" * 30)

# Example usage
if __name__ == "__main__":
    tickers = ['SHOP', 'MSFT', 'PEP', 'AMZN', 'META',  'JNJ', 'NVDA','TSLA','V']
    model_path = './logs/best_model.zip'  # Path to your best saved model
    start_date = "2023-06-19"
    end_date = "2024-06-19"
    initial_money = 1000

    # Download and process stock data
    stock_data = download_stock_data(tickers, start_date, end_date)

    # Load the best model
    model = load_best_model(model_path)

    # Run the model and show the stock picks with amounts for each day
    run_model_and_show_picks(model, stock_data, initial_money)
