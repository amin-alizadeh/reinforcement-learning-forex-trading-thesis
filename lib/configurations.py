import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from yaml import dump as y_dump
from io import TextIOWrapper
import logging
import pathlib

from .agent import LOSS_FUNCTIONS, OPTIMIZER_FUNCTIONS, ACTIVATION_FUNCTIONS


def get_scale(env):
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here

    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scale = StandardScaler()
    scale.fit(states)
    return scale


def remove_file(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)
    elif os.path.isdir(filepath):
        for f in os.listdir(filepath):
            remove_file(os.path.join(filepath, f))


def make_dir(directory):
    logging.debug(f'Creating a directory at {directory}')
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_technical_indicators(df, open_column='Open', low_column='Low', high_column='High', close_column='Close',
                                 bars=10, suffix=''):
    assert open_column in df.columns.values
    assert low_column in df.columns.values
    assert high_column in df.columns.values
    assert close_column in df.columns.values
    assert type(bars) is int
    assert type(suffix) is str and len(suffix) < 10

    half_bars = int(bars / 2)
    initial_columns = list(df.columns.values)
    # indicators
    df['Ln'] = df[low_column].rolling(window=bars).min()
    df['Hn'] = df[high_column].rolling(window=bars).max()

    # Stochastic %K
    df['SK'] = df.apply(
        lambda x: 100 * (x[close_column] - x['Ln']) / (x['Hn'] - x['Ln']) if x['Hn'] - x['Ln'] != 0 else 0,
        axis='columns')
    # Stochastic %D
    df['SD'] = df['SK'].rolling(window=bars - 1).sum() / bars
    # Larry William's %R
    df['LWR'] = df.apply(
        lambda x: 100 * (x['Hn'] - x[close_column]) / (x['Hn'] - x['Ln']) if x['Hn'] - x['Ln'] != 0 else 0,
        axis='columns')
    # df['LWR'] = (df['Hn'] - df[close_column]) / (df['Hn'] - df['Ln']) * 100
    # Moving average n bars
    df['MA{}'.format(bars)] = df[close_column].rolling(window=bars).sum() / bars
    # Moving average n/2 bars
    df['MA{}'.format(half_bars)] = df[close_column].rolling(window=half_bars).sum() / half_bars
    # OSCP (price oscillator)
    df['OSCP'] = 1 - df['MA{}'.format(bars)] / df['MA{}'.format(half_bars)]
    # Index return
    df['SYt'] = 100 * (np.log(df[close_column]) - np.log(df[close_column].shift(1)))
    df['ASY{}'.format(half_bars)] = df['SYt'].rolling(window=half_bars).sum() / half_bars
    df['ASY{}'.format(bars)] = df['SYt'].rolling(window=bars).sum() / bars

    # Move
    df['Mt'] = (df[high_column] + df[low_column] + df[close_column]) / 3
    # Sum of move in n bars
    df['SMt{}'.format(half_bars)] = df['Mt'].rolling(window=half_bars).sum() / half_bars

    for i in range(half_bars):
        df['Dt_abs{}'.format(i)] = np.abs(df['Mt'].shift(i) - df['SMt{}'.format(half_bars)])

    df['Dt'] = df['Dt_abs{}'.format(0)]
    for i in range(1, half_bars):
        df['Dt'] += df['Dt_abs{}'.format(i)]
    df['Dt'] /= half_bars

    # Removing unnecessary columns
    df.drop(['Dt_abs{}'.format(i) for i in range(half_bars)] + ['Ln', 'Hn'], axis=1, inplace=True)

    df.rename(lambda n: '{}{}'.format(n, suffix) if n not in initial_columns else n, axis='columns', inplace=True)

    return df


class Configurations:
    def __init__(self, output=None, train_file=None, train_directory=None,
                 episodes=1000, batch_size=32, investment=20000,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05,
                 mode='train', model_directory='models',
                 reward_directory='rewards',
                 join_columns=None, ignore_columns=None, prices=None, commission=0.5,
                 load_model=None, data_as_df=False,
                 memory_size=500, technical_indicators=False, cnn=False, a2c=False,
                 bars=10, cpu_cores=6, gpu_cores=1,
                 run_size='0%:100%', nn_layers=None, data_filter=None,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_steps=800,
                 random_memory_sampling=True, activation=None,
                 activation_last_layer=None, loss=None, loss_critic=None, optimizer=None):

        self.output = output
        self.episodes = episodes
        self.batch_size = batch_size
        self.investment = investment
        self.mode = mode
        self.model_directory = output + os.sep + model_directory
        self.reward_directory = output + os.sep + reward_directory
        self.train_file = train_file
        self.train_directory = train_directory
        self.n_stocks = None
        self.commission = commission * 0.01
        self.load_model = load_model
        self.memory_size = memory_size
        self.technical_indicators = technical_indicators
        self.cnn = cnn
        self.a2c = a2c
        self.bars = bars
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon_steps = epsilon_steps
        self.cpu_cores = cpu_cores
        self.gpu_cores = gpu_cores
        self.data_filter = data_filter
        self.data_as_df = data_as_df
        self.random_memory_sampling = random_memory_sampling
        self.activation = activation
        self.activation_last_layer = activation_last_layer
        self.loss = loss
        self.loss_critic = loss_critic
        self.optimizer = optimizer

        if not self.random_memory_sampling:
            self.memory_size = self.batch_size

        if join_columns is None:
            self.join_columns = ['Date', 'Hour']
        elif type(join_columns) is str:
            self.join_columns = join_columns.split(',')
        else:
            self.join_columns = join_columns

        if ignore_columns is None:
            self.ignore_columns = []  # ['Open', 'Low', 'High', 'Volume']
        elif type(ignore_columns) is str:
            self.ignore_columns = ignore_columns.split(',')
        else:
            self.ignore_columns = ignore_columns

        if prices is None:
            self.prices = [3, 7, 11, 15, 19]
        elif type(prices) is str and len(prices.split(',')) > 1:
            self.prices = [int(p) for p in prices.split(',')]
        elif type(prices) is str and len(prices.split(',')) == 1:
            self._from_column = True
            self.prices = prices
        else:
            self.prices = prices

        if nn_layers is None:
            self.nn_layers = [32]
        elif type(nn_layers) is str:
            self.nn_layers = [int(n) for n in nn_layers.split(',')]
        elif type(nn_layers) is list:
            self.nn_layers = nn_layers
        else:
            raise ValueError('Wrong type. nn_layers must be str or list of int')

        run_size = run_size.strip().split(':')
        assert (len(run_size) == 2)
        self.__run_size = {"start": None, "end": None}
        self.__run_size_ratio = {"start": None, "end": None}

        if run_size[0][-1] == '%':
            self.__run_size_ratio["start"] = float(run_size[0][:-1]) / 100
        elif run_size[0].isnumeric():
            self.__run_size["start"] = int(run_size[0])
        else:
            raise NameError('Wrong run size format.')

        if run_size[1][-1] == '%':
            self.__run_size_ratio["end"] = float(run_size[1][:-1]) / 100
        elif run_size[1].isnumeric():
            self.__run_size["end"] = int(run_size[1])
        else:
            raise NameError('Wrong format.')

        self.join_columns_data = None
        self.data = self.__get_training_data()

    def make_directories(self):
        make_dir(self.model_directory)
        make_dir(f"{self.model_directory}/best")
        make_dir(self.reward_directory)

    @classmethod
    def get_from_commandline(cls):
        def str_to_bool(s):
            if isinstance(s, str):
                s = s.lower()
                if s == 'true' or s == 'y' or s == 't' or s == 'y':
                    return True
            elif isinstance(s, int) and s != 0:
                return True

            return False

        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('-m', '--mode', type=str, required=False, default='train',
                            help='either "train" or "test". Default: train')
        parser.add_argument('-o', '--output', type=str, required=False, default='model',
                            help='result folder. Default: model')
        parser.add_argument('-td', '--train_directory', type=str, required=False,
                            help='directory of training files')
        parser.add_argument('-tf', '--train_file', type=str, required=False,
                            help='the single training file')
        parser.add_argument('-e', '--episodes', type=int, required=False, default=100,
                            help='number of episodes. Default: 100')
        parser.add_argument('-inv', '--investment', type=int, required=False, default=20000,
                            help='initial investment. Default: 20000')
        parser.add_argument('-bs', '--batch_size', type=int, required=False, default=32,
                            help='training batch size. Default: 32')
        parser.add_argument('-md', '--model_directory', type=str, required=False, default='models',
                            help='where model is stored. Default: models')
        parser.add_argument('-rd', '--reward_directory', type=str, required=False, default='rewards',
                            help='where rewards are stored. Default: rewards')
        parser.add_argument('-jc', '--join_columns', type=str, required=False, default='Date',
                            help='Join columns that are shared between data files (comma, separated). Default: Date')
        parser.add_argument('-ic', '--ignore_columns', type=str, required=False,
                            help='Columns to be ignored that are not a part of training or test (comma, separated)\n' +
                                 '. Default: Open,Low,High,Volume')
        parser.add_argument('-ps', '--prices', type=str, required=False, default='Close',
                            help='the prices on which trades take place (comma, separated) or the name of the column, '
                                 'e.g. Close or Open. Default: Close')
        parser.add_argument('-cm', '--commission', type=float, required=False, default=0.1,
                            help='commission in percentage. Default: 0.1')
        parser.add_argument('-lm', '--load_model', type=str, required=False, default=None,
                            help='Continue training on an existing model')
        parser.add_argument('-eps', '--epsilon', type=float, required=False, default=1.0,
                            help='Epsilon value of epsilon-greedy for the agent. Default: 1.0')
        parser.add_argument('-ed', '--epsilon_decay', type=float, required=False, default=0.995,
                            help='Epsilon value''s decay. Default: 0.995')
        parser.add_argument('-em', '--epsilon_min', type=float, required=False, default=0.05,
                            help='Epsilon value''s decay. Default: 0.05')
        parser.add_argument('-ms', '--memory_size', type=int, required=False, default=500,
                            help='Replay memory size. Default: 500')
        parser.add_argument('-ti', '--technical_indicators', action="store_true", required=False, default=False,
                            help='Whether to use the technical indicators or not. Boolean type FLAG. '
                                 'Technical indicators are SK,SD,LWR,MA10,MA5,OSCP,SYt,ASY5,ASY10,Mt,SMt5,Dt')
        parser.add_argument('-cnn', '--cnn', action="store_true", required=False, default=False,
                            help='Whether to use convolutional neural network. Boolean type FLAG')
        parser.add_argument('-a2c', '--a2c', action="store_true", required=False, default=False,
                            help='Whether to use Actor Critic Modeling. Boolean type FLAG')
        parser.add_argument('-s', '--run_size', type=str, required=False, default='0%:100%',
                            help='Use this if you want a part of the data instead of the whole. Options. \n' +
                                 'N  M items N through M. N and M can be negative. N less than M.\n' +
                                 'N M items N through M. N less than M. Default: 0 - 100 (all)')
        parser.add_argument('-nn', '--nn_layers', type=str, required=False, default=None,
                            help="Hidden layers of the agent's neural network(comma separated int values). Default: 32")
        parser.add_argument('-cpu', '--cpu_cores', type=int, required=False, default=6,
                            help="Number of CPU cores. Default: 6")
        parser.add_argument('-gpu', '--gpu_cores', type=int, required=False, default=1,
                            help="Number of CPU cores. Default: 1")
        parser.add_argument('-f', '--data_filter', type=str, required=False, default='',
                            help='Apply a filter on each of the loaded data frames. e.g. "Date<20190101"')
        parser.add_argument('-estart', '--epsilon_start', type=float, required=False, default=1.0,
                            help='Starting value of epsilon of the main episode loop. Default: 1.0')
        parser.add_argument('-eend', '--epsilon_end', type=float, required=False, default=0.1,
                            help='Ending value of epsilon of the main episode loop. Default: 0.1')
        parser.add_argument('-esteps', '--epsilon_steps', type=int, required=False, default=100,
                            help='The number of steps that epsilon_start should decay steadily to epsilon_end. '
                                 'Default: 100')
        parser.add_argument('-rms', '--random_memory_sampling', required=False, default=True,
                            type=str_to_bool, nargs='?',
                            help='if set to false, replay memory bugger won''t be used')
        parser.add_argument('-a', '--activation', type=str, required=False,
                            help=f"activation function. one of {', '.join(ACTIVATION_FUNCTIONS)}")
        parser.add_argument('-al', '--activation_last_layer', type=str, required=False,
                            help=f"activation function. one of {', '.join(ACTIVATION_FUNCTIONS)}")
        parser.add_argument('-l', '--loss', type=str, required=False,
                            help=f"loss function. one of {', '.join(LOSS_FUNCTIONS)}")
        parser.add_argument('-lc', '--loss_critic', type=str, required=False,
                            help=f"loss function. one of {', '.join(LOSS_FUNCTIONS)}")
        parser.add_argument('-oz', '--optimizer', type=str, required=False,
                            help=f"optimizer function. one of {', '.join(OPTIMIZER_FUNCTIONS)}")

        args = vars(parser.parse_args())

        return cls(**args)

    def __get_training_data(self):
        if self.train_directory is not None:
            self.n_stocks = 0
            dfs = None
            content_list = sorted(os.listdir(self.train_directory))
            for f in content_list:
                fpath = os.path.join(self.train_directory, f)
                if os.path.isfile(fpath) and pathlib.Path(fpath).suffix == '.csv':
                    self.n_stocks += 1
                    df = pd.read_csv(fpath, sep=';')
                    if self.data_filter is not None and len(self.data_filter) > 0:
                        df.query(self.data_filter, inplace=True)
                    indicator = f[:-4] if len(f.split('_')) == 1 else f.split('_')[0]
                    if self.technical_indicators:
                        df = process_technical_indicators(df, bars=self.bars, suffix='')

                    ignore_columns = [c for c in self.ignore_columns if c in list(df.columns.values)]
                    df.drop(columns=ignore_columns, inplace=True)
                    columns = [c for c in list(df.columns.values) if c not in self.join_columns]
                    new_columns = [f'{_}_{indicator}' for _ in columns]
                    df.rename(columns=dict(zip(columns, new_columns)), inplace=True)

                    if dfs is None:
                        dfs = df.copy()
                    else:
                        dfs = pd.merge(dfs, df, how='inner',
                                       left_on=self.join_columns, right_on=self.join_columns,
                                       suffixes=('', f'_{indicator}'))
            if self._from_column:
                c = list(dfs.columns.values)
                self.prices = np.array([i for i in range(len(c)) if self.prices in c[i]]) - len(self.join_columns)
            self.join_columns_data = dfs[self.join_columns]
            dfs.drop(columns=self.join_columns, inplace=True)
            dfs.dropna(inplace=True)

            if self.__run_size_ratio["start"] is not None:
                self.__run_size["start"] = int(self.__run_size_ratio["start"] * dfs.values.shape[0])
            if self.__run_size_ratio["end"] is not None:
                self.__run_size["end"] = int(self.__run_size_ratio["end"] * dfs.values.shape[0])

            if self.data_as_df:
                return dfs.iloc[self.__run_size["start"]:self.__run_size["end"], :]

            return dfs.values[self.__run_size["start"]:self.__run_size["end"], :]

        self.n_stocks = 1
        df = pd.read_csv(self.train_file, sep=';')
        return df

    def write_configs(self, output=None):
        output = output or f'{self.output}/configurations.txt'
        if isinstance(output, str):
            output = open(output, 'w+')
        assert isinstance(output, TextIOWrapper)

        y_dump(self.__get_self_dict(), output)
        output.close()

    def __get_self_dict(self):
        d = dict()
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = f'numpy array of size {v.shape}'
                if np.prod(v.shape) < 10:
                    d[k] += f'\n{v}'
            elif isinstance(v, pd.DataFrame):
                d[k] = f'Dataframe with columns {v.columns} and size {v.values.shape}'
            else:
                d[k] = v
        return d

    def __str__(self):
        return y_dump(self.__get_self_dict())
