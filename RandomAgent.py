import gym
from gym import spaces
import numpy as np
import yfinance as yf
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, stock_data, initial_money):
        super(StockTradingEnv, self).__init__()

        self.stock_data = stock_data
        self.initial_money = initial_money
        self.current_money = initial_money
        self.current_step = 0
        
        # Observation space: (3 days of open and adjusted close prices + today's open price) * 9 stocks
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4 * 9 * 2,), dtype=np.float32)
        
        # Action space: investments in each of the 9 stocks (percentages)
        self.action_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)

    def reset(self):
        self.current_money = self.initial_money
        self.current_step = 0
        return self._next_observation()
    
    def _next_observation(self):
        frame = np.array([
            self.stock_data.iloc[self.current_step, :].values,      # Day 1: Open and Close prices
            self.stock_data.iloc[self.current_step + 1, :].values,  # Day 2: Open and Close prices
            self.stock_data.iloc[self.current_step + 2, :].values,  # Day 3: Open and Close prices
            self.stock_data.iloc[self.current_step + 3, :].values   # Day 4: Open and Close prices
        ]).flatten()
        return frame
    
    def step(self, action):
        # Normalize action to ensure total percentage is 1
        action = action / np.sum(action)

        # Extract open and close prices correctly
        step_data = self.stock_data.iloc[self.current_step + 3].values
        current_open_prices = step_data[::2]   # Every other element starting from 0
        current_close_prices = step_data[1::2] # Every other element starting from 1

        investments = action * self.current_money

        # Print debug information
        # print(f"Step: {self.current_step}")
        # print("Current Open Prices:", current_open_prices)
        # print("Current Close Prices:", current_close_prices)
        # print("Investments:", investments)
        reward = 0
        for i in range(len(action)):
            profit_from_stock = investments[i] * (current_close_prices[i] / current_open_prices[i]) - investments[i]
            # print(f"  {tickers[i]}: Invested: {investments[i]:.2f}, Profit/Loss: {profit_from_stock:.2f}, Open: {current_open_prices[i]:.2f}, Close: {current_close_prices[i]:.2f}")
            self.current_money += profit_from_stock
            reward += profit_from_stock

        self.current_step += 1

       
        done = self.current_step >= len(self.stock_data) - 4

        # print(f"  Current Money: {self.current_money:.2f}, Reward {reward:.2f}")
        
        return self._next_observation(), reward, done, {}

# Define the random agent function
def random_agent(env, episodes=1):
    avg = 0
    high = 0
    low = 20000000
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        while True:
            action = env.action_space.sample()  # Random action
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        avg += env.current_money
        if(env.current_money>high):
            high = env.current_money
        if(env.current_money<low):
            low = env.current_money
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Final Money: {env.current_money}")
    print(f"Average: {avg / episodes:,.2f}")
    print(f"High: {high:,.2f}")
    print(f"Low: {low:,.2f}")


# Define the stock tickers
tickers = ['SHOP', 'MSFT', 'PEP', 'AMZN', 'TSLA', 'META', 'NVDA', 'JNJ', 'V']

# Download data for the last month
data = yf.download(tickers, start="2023-12-01", end="2024-06-01")

# Extract the required columns (Open and Adj Close)
data = data[['Open', 'Adj Close']]

# Flatten the DataFrame so each day is a row with all 9 stocks
rows = []

for i in range(len(data)):
    row = []
    for ticker in tickers:
        row.append(data['Open'][ticker].iloc[i])
        row.append(data['Adj Close'][ticker].iloc[i])
    rows.append(row)

flattened_data = pd.DataFrame(rows)

# Reset index and name the columns appropriately
flattened_data.columns = [f'{ticker}_Open' for ticker in tickers] + [f'{ticker}_Adj_Close' for ticker in tickers]

# Initialize environment with the processed stock data
env = StockTradingEnv(flattened_data, initial_money=1000)

# Run the random agent
random_agent(env, episodes=1000)

