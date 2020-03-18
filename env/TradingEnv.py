import gym
from gym import spaces
import pandas as pd
import numpy as np
import math

from render.tradinggraph import TradingGraph

import logging
import loggingUtil

STEPS_IN_GRAPH = 80


class TradingEnv(gym.Env):
    """A trading environment for OpenAI gym"""
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None

    def __init__(self, data, feature_list, initial_account_balance=10000.0, margin_per_lot=3330.0, min_trade_size=0.01,
                 spread=0.00015, lookback_window_size=100, pip_value=97000, test=False):
        super(TradingEnv, self).__init__()
        self.logger = loggingUtil.get_new_logger('env')
        self.data = data
        self.feature_list = feature_list
        self.initial_account_balance = initial_account_balance
        self.margin_per_lot = margin_per_lot
        self.min_trade_size = min_trade_size
        self.lookback_window_size = lookback_window_size
        self.pip_value = pip_value
        self.spread = spread
        self.max_account_balance = self.initial_account_balance * 100.0

        # define action and observation space
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0, 1.0]), dtype=np.float16)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.lookback_window_size, len(self.feature_list)), dtype=np.float32)
        self.reward_range = (0.0, self.max_account_balance)

        self._reset_values()

        if test:
            self.logger.info("Testing mode - setting loglevel to debug")
            self.logger.setLevel(logging.DEBUG)

    def reset(self):
        # Reset the state of the environment to an initial state
        self._reset_values()
        return self._next_observation()

    def _reset_values(self):
        self.current_step = self.lookback_window_size+1
        self.total_profit = 0.0
        self.open_ratio = 0.0
        self.open_pl_ratio = 0.0
        self.trades = []
        self.balance = self.initial_account_balance
        self.net_worth = self.initial_account_balance
        self.max_net_worth = self.initial_account_balance
        self.last_sortino = 0.0
        self.finished_buy_trades = 0
        self.finished_sell_trades = 0
        self.returns = []
        self.inventory = []
        self.max_drawdown = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.losing_buy_trades = 0
        self.losing_sell_trades = 0
        self.winning_buy_trades = 0
        self.winning_sell_trades = 0
        self.max_position_size = (self.initial_account_balance / self.margin_per_lot) * 0.10
        self.cummulated_reward = 0.0

    def _next_observation(self):
        # Get the stock data points for the lookback period
        obs = self.data.loc[self.current_step-self.lookback_window_size:self.current_step-1, self.feature_list].values
        self.logger.debug(f'_next_observation: Next observation taken from '
                          f'{self.data.loc[self.current_step, "Date"]}')

        return obs

    def step(self, action):
        # Execute one time step within the environment
        reward = self._take_action(action)

        done = False

        margin_call = False

        if self.net_worth - (abs(self.open_ratio * self.max_position_size * self.margin_per_lot)) <= self.balance * 0.1:
            done = True
            margin_call = True
            if self.visualization is not None:
                self.logger.info(f"Margin Call: {self.net_worth}")

        if self.current_step+2 >= len(self.data):
            done = True
            if self.visualization is not None:
                self.logger.info("Full training batch done")

        if done:
            if margin_call:
                current_price = self.data.loc[self.current_step, "Open"]
                self._close_all_trades(current_price)
                self.total_profit = -self.initial_account_balance

            sortino = self._sortino_ratio(self.returns)

            reward = sortino * (self.total_profit / self.initial_account_balance)*100

            if self.total_profit < 0.0 or sortino < 0.0:
                reward = -abs(reward)

        reward = min(self.max_account_balance, max(0.0, reward))

        self.cummulated_reward += reward
        self.logger.debug(f"Reward for step {self.current_step}: {reward}")
        self.logger.debug(f"total cummulated reward: {self.cummulated_reward}")

        self.current_step += 1
        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        sum_turn_profit = 0.0
        reward = 0.0

        # get current price infos
        current_price = self.data.loc[self.current_step, "Open"]
        current_close_price = self.data.loc[self.current_step, "Close"]
        self.logger.debug(
            f'_take_action: current price info {self.data.loc[self.current_step, "Open"]} on {self.data.loc[self.current_step, "Date"]}')

        # extract amount to trade and action type (hold, buy, sell) from action
        amount = math.floor((action[0] * self.max_position_size) / self.min_trade_size) * self.min_trade_size
        amount = min(1.0, max(0.0, amount))
        current_action = np.argmax(action[1:])

        self.logger.debug(f'_take_action: returned action {action[1:]}')
        self.logger.debug(f'_take_action: selected action {current_action}')
        self.logger.debug(f'_take_action: returned amount {amount}')

        """
        if len(self.inventory) > 0:
            sum_turn_profit += self._check_open_trades(current_price)
        """

        # change inventory (open positions) according to selected action (buy, sell) and position size
        if current_action > 0 and action[-2] != action[-1]:
            # if sell then make amount negative
            if current_action > 1:
                amount = amount * -1
            sum_turn_profit += self._set_position_size(amount, current_price)

        # calculate open_ratio for current turn and save to dataset
        open_position_size = np.sum([trade[1] for trade in self.inventory])
        self.open_ratio = open_position_size / self.max_position_size
        self.data.loc[self.current_step, 'open_ratio'] = min(1.0, max(-1.0, self.open_ratio))
        self.logger.debug(f'_take_action: open_ratio = {self.open_ratio:.2f}')

        # calculate open_pl_ratio for current turn and save to dataset
        open_pl = np.sum([self._calculate_profit((current_close_price - trade[0]), trade[1]) for trade in self.inventory])
        self.open_pl_ratio = (open_pl / self.initial_account_balance)
        self.data.loc[self.current_step, 'open_pl_ratio'] = min(1.0, max(-1.0, self.open_pl_ratio))
        self.logger.debug(f'_take_action: open_pl = {open_pl:.2f}')

        # add this turns profit to total profit so far, update balance, drawdown and networth
        self.total_profit += sum_turn_profit
        self.balance = self.initial_account_balance + self.total_profit
        self.net_worth = self.balance + open_pl
        current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth

        self.data.loc[self.current_step, 'profit_ratio'] = (self.total_profit / self.max_account_balance) * 100

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # if there were any closed positions in this turn update the sortino ration for reward
        if sum_turn_profit != 0.0:
            sortino = self._sortino_ratio(self.returns)
            sortino_change = sortino - self.last_sortino
            reward = sortino_change
            if sum_turn_profit > 0.0:
                reward = abs(reward)
            self.last_sortino = sortino

        # log current values for debugging
        self.logger.debug(f'_take_action: balance = {self.balance:.2f}')
        self.logger.debug(f'_take_action: net_worth = {self.net_worth:.2f}')
        self.logger.debug(f'_take_action: size of open positions = {open_position_size}')
        self.logger.debug(f'_take_action: finished buy trades = {self.finished_buy_trades}')
        self.logger.debug(f'_take_action: finished sell trades = {self.finished_sell_trades}')
        self.logger.debug(f'_take_action: winning trades = {self.winning_trades}')
        self.logger.debug(f'_take_action: losing trades = {self.losing_trades}')
        pct_winning_buy = 0.0
        sum_buy = self.winning_buy_trades + self.losing_buy_trades
        if sum_buy > 0:
            pct_winning_buy = self.winning_buy_trades / (self.winning_buy_trades + self.losing_buy_trades)
        pct_winning_sell = 0.0
        sum_sell = self.winning_sell_trades + self.losing_sell_trades
        if sum_sell > 0:
            pct_winning_sell = self.winning_sell_trades / (self.winning_sell_trades + self.losing_sell_trades)
        pct_total_trades = 0.0
        if len(self.returns) > 0:
            pct_total_trades = self.winning_trades / (self.winning_trades + self.losing_trades)
        self.logger.debug(f'_take_action: % winning trades = {pct_total_trades:.2f}')
        self.logger.debug(f'_take_action: % winning buy trades = {pct_winning_buy:.2f}')
        self.logger.debug(f'_take_action: % winning sell trades = {pct_winning_sell:.2f}')
        self.logger.debug(f'_take_action: mean of returns: {np.nanmean(self.returns) if len(self.returns) > 0 else 0.0:.2f}')
        self.logger.debug(f'_take_action: max drawdown = {self.max_drawdown*100:.2f}')
        self.logger.debug(f'_take_action: Sortino Ratio = {self.last_sortino:.2f}')

        return reward

    def _set_position_size(self, amount, current_price):
        profit = 0.0
        current_position_sum = np.sum(trade[1] for trade in self.inventory)
        if (current_position_sum > 0.0 and amount < 0.0) or (current_position_sum < 0.0 and amount > 0.0) or amount == 0.0:
            profit += self._close_all_trades(current_price)

        current_position_sum = np.sum(trade[1] for trade in self.inventory)

        if current_position_sum != amount:
            if amount < 0.0:
                if current_position_sum > amount:
                    trade_size = amount - current_position_sum
                    self._open_trade(trade_size, current_price)
                else:
                    while current_position_sum < amount:
                        amount_to_close = current_position_sum - amount
                        if self.inventory[0][1] >= amount_to_close:
                            trade = self.inventory.pop(0)
                            trade_size = trade[1]
                        else:
                            trade = self.inventory[0]
                            self.inventory[0][1] = self.inventory[0][1] - amount_to_close
                            trade_size = amount_to_close

                        profit += self._close_trade(trade_size, trade[0], current_price)
                        current_position_sum = np.sum(trade[1] for trade in self.inventory)

            elif amount > 0.0:
                if current_position_sum < amount:
                    trade_size = amount - current_position_sum
                    self._open_trade(trade_size, current_price)
                else:
                    while current_position_sum > amount:
                        amount_to_close = current_position_sum - amount
                        if self.inventory[0][1] <= amount_to_close:
                            trade = self.inventory.pop(0)
                            trade_size = trade[1]
                        else:
                            trade = self.inventory[0]
                            self.inventory[0][1] = self.inventory[0][1] - amount_to_close
                            trade_size = amount_to_close

                        profit += self._close_trade(trade_size, trade[0], current_price)
                        current_position_sum = np.sum(trade[1] for trade in self.inventory)

        return profit

    def _check_open_trades(self, current_price):
        profit = 0.0
        counter = 0

        for trade in self.inventory:
            if self._calculate_profit((current_price - trade[0]), trade[1]) < -(self.initial_account_balance * 0.02):
                trade = self.inventory.pop(counter)
                profit += self._close_trade(trade[1], trade[0], current_price)
            else:
                counter +=1

        return profit

    def _close_all_trades(self, current_price):
        profit = 0.0
        while len(self.inventory) > 0:
            trade = self.inventory.pop(0)
            profit += self._close_trade(trade[1], trade[0], current_price)
        return profit

    def _close_trade(self, amount, open_price, close_price):
        pip_change = close_price - open_price
        trade_open_pl = pip_change * self.pip_value * amount
        if amount < 0.0:
            position_type = "close_sell"
            self.finished_sell_trades += 1
        else:
            position_type = "close_buy"
            self.finished_buy_trades += 1

        self.trades.append({'step': self.current_step,
                            'shares': amount, 'total': trade_open_pl,
                            'type': position_type})

        if trade_open_pl >= 0.0:
            self.winning_trades += 1
            if amount >= 0.0:
                self.winning_buy_trades += 1
            else:
                self.winning_sell_trades += 1
        else:
            self.losing_trades += 1
            if amount >= 0.0:
                self.losing_buy_trades += 1
            else:
                self.losing_sell_trades += 1

        self.returns.append(trade_open_pl)

        return trade_open_pl

    def _open_trade(self, amount, price):
        type = None
        position_price = 0.0
        if amount > 0.0:
            position_price = price+self.spread
            type = "buy"

        elif amount < 0.0:
            position_price = price - self.spread
            type = "sell"

        self.inventory.append([position_price, amount])
        self.trades.append({'step': self.current_step,
                'shares': amount, 'total': self.margin_per_lot * amount,
                'type': type})

        self.logger.debug(f'_take_action: open {type} position at {position_price}  with size {amount}')

    def _calculate_profit(self, pips, amount):
        profit = pips * self.pip_value * amount
        return profit

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'file':
            self._render_to_file(kwargs.get('filename', 'render.txt'))

        elif mode == 'live':
            if self.visualization == None:
                self.visualization = TradingGraph(self.data, initial_net_worth=self.initial_account_balance, title=kwargs.get('title', None))

            if self.current_step-1 > STEPS_IN_GRAPH:
                self.visualization.render(
                    self.current_step, self.balance, self.net_worth, self.trades, window_size=STEPS_IN_GRAPH)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None

    def _render_to_file(self, filename='render.txt'):
        profit = self.net_worth - self.initial_account_balance
        file = open(filename, 'a+')

        file.write(f'Step: {self.current_step}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(
            f'Open PL: {self.open_pl_ratio}\n')
        file.write(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})\n')
        file.write(f'Profit: {self.total_profit}\n\n')
        file.close()

    def _sortino_ratio(self, returns, risk_free_rate = 0.):
        sortino = 0.0

        if len(returns) > 0:
            downside_returns = [ret for ret in returns if ret < 0.0]

            expected_return = np.mean(returns)

            if len(downside_returns) < 1:
                ret_std = np.std(returns)
            else:
                ret_std = np.std(downside_returns)

            if ret_std != 0.0 and ret_std is not None:
                sortino = (expected_return - risk_free_rate) / ret_std
            else:
                sortino = np.tanh(expected_return)

        return sortino
