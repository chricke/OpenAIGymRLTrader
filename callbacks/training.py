import numpy as np
import json
from configparser import ConfigParser

from stable_baselines.results_plotter import load_results, ts2xy

import loggingUtil

logger = loggingUtil.get_new_logger('callback')

config = ConfigParser()
config.read('./config/config.ini')
callback_configs = config['Callbacks']

n_steps = 0
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


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 100 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            logger.info(f"{x[-1]} timesteps")
            logger.info(f"Best mean reward: {best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                logger.info("Saving new best model")
                _locals['self'].save(log_dir + best_model_name)

                data = {}
                data['best_mean_reward'] = best_mean_reward

                with open(log_dir + status_file, 'w') as outfile:
                    json.dump(data, outfile)
            elif not np.isnan(mean_reward):
                logger.info("Saving model checkpoint")
                _locals['self'].save(log_dir + model_name)

        if (n_steps + 1) % 10000 == 0:
            logger.info(f"Saving model checkpoint for step {n_steps + 1}")
            _locals['self'].save(log_dir + model_name + '_' + str(n_steps + 1))

    n_steps += 1
    return True
