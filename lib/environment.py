import numpy as np
import itertools
from enum import Enum
from lib.memory import ReplayBuffer
import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


class Actions(Enum):
    SELL = 0
    HOLD = 1
    BUY = 2

    def __str__(self):
        return self.name


class MultiStockEnv:
    """
  A n-stock trading environment.
  State: vector of size (n_stock + price details + 1)
    - # shares of stock 1..n owned
    - price details of stock 1..n (open, high, low, close)
    - cash owned (can be used to purchase more stocks)
  Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
  """

    def __init__(self, data, n_stock, price_indices=None, initial_investment=20000, commission=0.0):
        self.stock_price_history = data
        self.n_step = data.shape[0]
        self.n_stock = n_stock
        self.commission = commission
        self.price_indices = np.arange(data.shape[1]) if price_indices is None else price_indices

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment

        self.action_space = np.arange(len(Actions) ** self.n_stock)
        self.action_list = list(map(list, itertools.product([a for a in Actions], repeat=self.n_stock)))

        self.state_dim = self.n_stock + self.stock_price_history.shape[1] + 1

        self.reset()

    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def step(self, action):
        assert action in self.action_space

        # get current value before performing the action
        prev_val = self._get_val()

        # update price, i.e. go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]

        # perform the trade
        self._trade(action)

        # get the new value after taking the action
        cur_val = self._get_val()

        # reward is the increase in porfolio value
        reward = cur_val - prev_val

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # store the current value of the portfolio here
        info = {'cur_val': cur_val}

        # conform to the Gym API
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:-1] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        return self.stock_owned.dot(self.stock_price[self.price_indices]) + self.cash_in_hand

    def _trade(self, action):
        # index the action we want to perform
        # 0 = sell
        # 1 = hold
        # 2 = buy
        # e.g. [2,1,0] means:
        # buy first stock
        # hold second stock
        # sell third stock
        action_vec = self.action_list[action]

        # determine which stocks to buy or sell
        sell_index = []  # stores index of stocks we want to sell
        buy_index = []  # stores index of stocks we want to buy
        for i, a in enumerate(action_vec):
            if a == Actions.SELL:
                sell_index.append(i)
            elif a == Actions.BUY:
                buy_index.append(i)

        # sell any stocks we want to sell
        # then buy any stocks we want to buy
        if sell_index:
            # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
            for i in sell_index:
                stock_value = self.stock_price[self.price_indices[i]] * self.stock_owned[i]
                commission_value = stock_value * self.commission
                self.cash_in_hand += stock_value - commission_value
                self.stock_owned[i] = 0
        if buy_index:
            # NOTE: when buying, we will loop through each stock we want to buy,
            #       and buy one share at a time until we run out of cash
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[self.price_indices[i]]:
                        self.stock_owned[i] += 1  # buy one share
                        stock_value = self.stock_price[self.price_indices[i]]
                        commission_value = stock_value * self.commission
                        self.cash_in_hand -= stock_value + commission_value
                    else:
                        can_buy = False
