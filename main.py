import numpy as np
import csv
from datetime import datetime
import pickle
import logging
import warnings

from lib.configurations import Configurations, get_scale, remove_file
from lib.environment import MultiStockEnv
from lib.agent import DQRLAgent

DEBUG_MODE = False
logger = logging.getLogger("logger")


def play_one_episode(agent, env, mode, scale, batch_size, info_writer=None, print_results=True):
    state = env.reset()
    state = scale.transform([state])
    done = False

    id_step = 0
    info = {'cur_val': 0}
    info_steps = []

    while not done:
        action = agent.act(state, mode)
        next_state, reward, done, info = env.step(action)

        next_state = scale.transform([next_state])

        id_step += 1

        if mode == 'train':
            agent.update_replay_memory(state, action, reward, next_state, done)
            agent.replay(batch_size)
        elif mode == 'test' and print_results:
            logger.debug(f'{id_step}: For action {env.action_list[action]} the reward is {reward}.'
                          f' Total value: {info["cur_val"]}')
        state = next_state
        info_step = [id_step, reward, info['cur_val'], env.cash_in_hand] + \
                    [str(a) for a in env.action_list[action]] + \
                    list(env.stock_owned)
        info_steps.append(info_step)

        if info_writer is not None:
            info_writer.writerow(info_step)

    return info['cur_val'], info_steps


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    if DEBUG_MODE:
        # config
        configs = Configurations(output='model', episodes=2, train_directory='data/daily', data_filter="Date<20200100",
                                 join_columns='Date', prices='Close', technical_indicators=True, nn_layers=[16, 32])
    else:

        configs = Configurations.get_from_commandline()

    data = configs.data
    configs.make_directories()
    logger.debug(configs)
    configs.write_configs()
    # exit(0)
    env = MultiStockEnv(data, n_stock=configs.n_stocks, price_indices=configs.prices,
                        initial_investment=configs.investment,
                        commission=configs.commission)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQRLAgent(state_size, action_size, memory_size=configs.memory_size,
                      epsilon=configs.epsilon, epsilon_decay=configs.epsilon_decay, epsilon_min=configs.epsilon_min,
                      model_hidden_dims=configs.nn_layers, cnn=configs.cnn, a2c=configs.a2c,
                      batch_size=configs.batch_size, model_summary_dir=configs.output,
                      random_memory_sampling=configs.random_memory_sampling,
                      activation=configs.activation, activation_last_layer=configs.activation_last_layer,
                      loss=configs.loss, optimizer=configs.optimizer)

    # store the final value of the portfolio (end of episode)
    portfolio_value = []
    reward_writer = None
    reward_file = None

    if configs.mode == 'train':
        if configs.load_model is not None:
            with open(f'{configs.load_model}/scale.pkl', 'rb') as scale_file:
                scale = pickle.load(scale_file)
            agent.load(configs.load_model)
        else:
            scale = get_scale(env)
            # save the scale
            with open(f'{configs.model_directory}/scale.pkl', 'wb') as scale_file:
                pickle.dump(scale, scale_file)

            with open(f'{configs.model_directory}/best/scale.pkl', 'wb') as scale_file:
                pickle.dump(scale, scale_file)

    elif configs.mode == 'test':
        # then load the previous scale
        with open(f'{configs.load_model}/scale.pkl', 'rb') as scale_file:
            scale = pickle.load(scale_file)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        # agent.epsilon = 0.01

        # load trained weights
        agent.load(configs.load_model)

        reward_file = open(f'{configs.reward_directory}/rewards.csv', 'w+')
        reward_writer = csv.writer(reward_file, delimiter=';', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        header_row = ["Index", "Reward", "Value", "Cash"] + \
                     [f"Action_{i + 1}" for i in range(configs.n_stocks)] + \
                     [f"Stock_{i + 1}" for i in range(configs.n_stocks)]
        reward_writer.writerow(header_row)

    csv_file = open(f'{configs.reward_directory}/{configs.mode}.csv', 'w+')
    csv_writer = csv.writer(csv_file, delimiter=';', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    csv_writer.writerow(["Episode", "Duration", "Value", "Epsilon"])

    epsilon = configs.epsilon
    epsilon_start, epsilon_end, epsilon_steps = configs.epsilon_start, configs.epsilon_end, configs.epsilon_steps

    best_value = -1
    # configs.episodes = 1
    # play the game num_episodes times
    for e in range(configs.episodes):
        t0 = datetime.now()

        agent.epsilon = epsilon
        val, info = play_one_episode(agent, env, configs.mode, scale=scale,
                                     batch_size=configs.batch_size,
                                     info_writer=reward_writer)
        dt = datetime.now() - t0
        logger.info(f"episode: {e + 1}/{configs.episodes}, "
                     f"episode end value: {val:.2f}, duration: {dt}. epsilon at {epsilon}")
        csv_writer.writerow([e + 1, dt, val, epsilon])
        csv_file.flush()
        portfolio_value.append(val)  # append episode end portfolio value

        if epsilon > epsilon_end:
            epsilon -= (epsilon_start - epsilon_end) * 1 / epsilon_steps
        else:
            epsilon = epsilon_end

        # save the latest model after each iteration
        if configs.mode == 'train':
            # remove_file(f'{configs.model_directory}/model')
            # save the DQN
            agent.save(configs.model_directory)

            if val > best_value:
                best_value = val
                agent.save(f"{configs.model_directory}/best")

    csv_file.close()

    if reward_file is not None:
        reward_file.close()
    # save portfolio value for each episode
    np.save(f'{configs.reward_directory}/{configs.mode}.npy', portfolio_value)
