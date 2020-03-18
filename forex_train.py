import numpy as np
import pandas as pd
import os
import glob

import json
from multiprocessing import freeze_support

from callbacks.training import callback
from configparser import ConfigParser

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.bench import Monitor

from env.TradingEnv import TradingEnv

import loggingUtil


def make_env(data, feature_columns):
    # The algorithms require a vectorized environment to run
    env = TradingEnv(data, feature_columns, lookback_window_size=96)
    # Create the vectorized environment
    env = Monitor(env, log_dir, allow_early_resets=True)

    return env

if __name__ == '__main__':
    freeze_support()

    logger = loggingUtil.get_new_logger('main')

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
            logger.info(f'Best mean reward taken from pretraining: {best_mean_reward}')
    except:
        best_mean_reward = -np.inf

    logger.info("Loading data...")
    data = pd.read_csv('data/training_data.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d %H:%M:%S')

    logger.info(f'Found {len(data)} entries...')
    logger.debug(data.head())
    logger.debug(data.tail())

    feature_columns = data.columns[6:]

    logger.info(f"Feature columns: {feature_columns}")
    train_data = data.iloc[:-20000].reset_index(drop=True)
    logger.info(f'Length of training data: {len(train_data)}')

    logger.debug(train_data[feature_columns].head())

    logger.info('Creating training environment...')
    env = DummyVecEnv([lambda: make_env(train_data, feature_columns)])


    if np.isneginf(best_mean_reward):
        logger.info('Init model...')
        model = PPO2(MlpLnLstmPolicy, env, nminibatches=1, verbose=1, tensorboard_log='tb_logs')
    else:
        logger.info("Loading pretrained model...")
        list_of_files = glob.glob(log_dir + '*.pkl')
        latest_file = max(list_of_files, key=os.path.getmtime)
        logger.info(f"model file: {latest_file}")
        model = PPO2.load(latest_file, env=env, tensorboard_log='tb_logs')

    logger.info("Train model...")
    model.learn(total_timesteps=40000000, callback=callback)

    logger.info("Training finished.")
