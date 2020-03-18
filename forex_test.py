import numpy as np
import pandas as pd
import json
import glob
import os

from multiprocessing import freeze_support

from configparser import ConfigParser

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.TradingEnv import TradingEnv

import loggingUtil

def make_env(data):

    # The algorithms require a vectorized environment to run
    env = TradingEnv(data, feature_columns, test=True)

    return env

if __name__ == '__main__':
    freeze_support()

    logger = loggingUtil.get_new_logger('test')

    config = ConfigParser()
    config.read('./config/config.ini')
    callback_configs = config['Callbacks']

    log_dir = f"./{callback_configs['checkpoint_path']}/"
    best_model_name = callback_configs['checkpoint_filename_best']
    model_name = callback_configs['checkpoint_filename']
    status_file = callback_configs['status_file_name']

    try:
        with open(log_dir + status_file) as json_file:
            data = json.load(json_file)
            best_mean_reward = data['best_mean_reward']
            logger.info(f'Best mean reward for model in training: {best_mean_reward}')
    except:
        best_mean_reward = -np.inf

    logger.info("Loading data...")
    df = pd.read_csv('data/training_data.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')

    data = df
    feature_columns = data.columns[6:]

    logger.info("Scaling the data...")
    logger.debug(f"Columns to scale: {feature_columns}")

    logger.debug(data[feature_columns].head())

    logger.info("Starting backtest...")
    # The algorithms require a vectorized environment to run
    test_data = data.iloc[-20000:].reset_index(drop=True)
    logger.info(f'Starting backtest from {test_data.iloc[0]["Date"]} to {test_data.iloc[-1]["Date"]}')
    env = TradingEnv(test_data, feature_columns, lookback_window_size=96, test=True)

    logger.info('Creating test environment...')
    # Create the vectorized environment
    env = DummyVecEnv([lambda: env])

    logger.info("Loading pretrained model...")
    list_of_files = glob.glob(log_dir + '*.pkl')
    latest_file = max(list_of_files, key=os.path.getmtime)
    logger.info(f"model file: {latest_file}")
    model = PPO2.load(latest_file, env=env)

    obs = env.reset()
    done = False
    stepcount = 0
    while not done:
        logger.info(f"step {stepcount + 1}")
        env.render(mode='live', title='EURUSD')
        action, _states = model.predict(obs)
        logger.debug(f"Received action: {action}")
        obs, rewards, done, info = env.step(action)
        stepcount += 1
