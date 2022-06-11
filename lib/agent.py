from lib.memory import ReplayBuffer
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Conv1D, Conv2D, MaxPool1D, Activation, Flatten, BatchNormalization, Add
from keras.regularizers import l2
from keras.initializers import RandomUniform
from keras.optimizers import Adam, Nadam, Adadelta, Optimizer
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.utils import to_categorical
import logging
import warnings


logger = logging.getLogger("logger")
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


ACTIVATION_FUNCTIONS = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign']
LOSS_FUNCTIONS = ['mse', 'msle', 'mape', 'categorical_crossentropy', 'sparse_categorical_crossentropy']
OPTIMIZER_FUNCTIONS = ['adam', 'nadam', 'adadelta']


def ann_model(state_size, action_size, hidden_dims=None,
              activation=None, activation_last_layer=None,
              loss=None, optimizer=None):
    """
    activation: one of 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign'
    https://keras.io/api/layers/activations/
    loss: one of 'mse', 'msle', 'mape', 'categorical_crossentropy'
    https://keras.io/api/losses/
    optimizer: one of 'adam', 'nadam', 'adadelta'
    https://keras.io/api/optimizers/
    """
    hidden_dims = [32] if hidden_dims is None else hidden_dims
    activation = activation or 'relu'
    activation_last_layer = activation_last_layer or 'sigmoid'
    loss = loss or 'mse'
    optimizer = optimizer or 'adam'
    activation = activation.lower()
    activation_last_layer = activation_last_layer.lower()
    loss = loss.lower()

    assert isinstance(hidden_dims, list)
    assert activation in ACTIVATION_FUNCTIONS
    assert activation_last_layer in ACTIVATION_FUNCTIONS
    assert loss in LOSS_FUNCTIONS

    model = Sequential()
    model.add(Dense(units=hidden_dims[0], kernel_initializer='uniform', activation=activation, input_dim=state_size,
                    name='input'))
    for i in range(1, len(hidden_dims)):
        model.add(Dense(units=hidden_dims[i], kernel_initializer='uniform',
                        activation='relu', name=f'layer_{i}'))

    # activation = 'sigmoid'
    model.add(Dense(units=action_size, kernel_initializer='uniform', activation=activation_last_layer, name='output'))

    metrics = None
    if isinstance(loss, str) and \
            (loss.lower() == 'categorical_crossentropy' or loss.lower() == 'sparse_categorical_crossentropy'):
        metrics = ['accuracy']

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    logger.debug(model.summary())
    return model


def get_net(input_size, hidden_dims=None, name='', activation=None):
    """
    activation: one of 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign'
    https://keras.io/api/layers/activations/
    """
    hidden_dims = [32] if hidden_dims is None else hidden_dims
    activation = 'relu' or activation
    activation = activation.lower()

    assert isinstance(hidden_dims, list)
    assert activation in ACTIVATION_FUNCTIONS

    net_in = Input(shape=(input_size,), name=f'{name}_input_states')

    net = net_in
    for i in range(len(hidden_dims)):
        net = Dense(units=hidden_dims[i], kernel_regularizer=l2(1e-6),
                    name=f'{name}_dense_layer_{i + 1}')(net)
        net = BatchNormalization(name=f'{name}_batch_normalization_{i + 1}')(net)
        net = Activation(activation, name=f'{name}_activation_{i + 1}')(net)

    return net_in, net


def actor_model(state_size, action_size, hidden_dims=None, activation=None,
                loss=None, optimizer=None):
    """
    activation: one of 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign'
    https://keras.io/api/layers/activations/
    loss: one of 'mse', 'msle', 'mape', 'categorical_crossentropy'
    https://keras.io/api/losses/
    optimizer: one of 'adam', 'nadam', 'adadelta'
    https://keras.io/api/optimizers/
    """
    hidden_dims = [32] if hidden_dims is None else hidden_dims
    activation = activation or 'relu'
    loss = loss or 'mse'
    optimizer = optimizer or 'adam'
    activation = activation.lower()
    loss = loss.lower()

    assert isinstance(hidden_dims, list)
    assert activation in ACTIVATION_FUNCTIONS
    assert loss in LOSS_FUNCTIONS
    states, net = get_net(state_size, hidden_dims, name='actor', activation=activation)

    # activation: softmax
    actions = Dense(units=action_size, activation=activation, name='actor_actions')(net)

    model = Model(inputs=states, outputs=actions, name='actor')

    action_gradients = Input(shape=(action_size,), name='action_gradient')
    loss = K.mean(-action_gradients * actions)

    if isinstance(optimizer, str) and optimizer.lower() == 'adam':
        optimizer = Adam(lr=.00001)
    elif isinstance(optimizer, str) and optimizer.lower() == 'nadam':
        optimizer = Nadam(lr=.00002)
    elif isinstance(optimizer, str) and optimizer.lower() == 'adadelta':
        optimizer = Adadelta(lr=0.9)

    updates_op = optimizer.get_updates(params=model.trainable_weights, loss=loss)
    train_fn = K.function(
        inputs=[model.input, action_gradients, K.learning_phase()],
        outputs=[],
        updates=updates_op)

    logger.debug(model.summary())
    return model, train_fn


def critic_model(state_size, action_size, hidden_dims=None, activation=None,
                 loss=None, optimizer=None):
    """
    activation: one of 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign'
    https://keras.io/api/layers/activations/
    loss: one of 'mse', 'msle', 'mape', 'categorical_crossentropy'
    https://keras.io/api/losses/
    optimizer: one of 'adam', 'nadam', 'adadelta'
    https://keras.io/api/optimizers/
    """
    hidden_dims = [32] if hidden_dims is None else hidden_dims
    activation = activation or 'relu'
    loss = loss or 'mse'
    optimizer = optimizer or 'adam'
    activation = activation.lower()
    loss = loss.lower()

    assert isinstance(hidden_dims, list)
    assert activation in ACTIVATION_FUNCTIONS
    assert loss in LOSS_FUNCTIONS

    actions = Input(shape=(action_size,), name='critic_input_actions')

    states, net_states = get_net(state_size, hidden_dims[0:-1], name='critic', activation=activation)

    net_states = Dense(units=hidden_dims[-1], kernel_regularizer=l2(1e-6),
                       name=f'states_dense_layer_{len(hidden_dims)}')(net_states)

    net_actions = Dense(units=hidden_dims[-1], kernel_regularizer=l2(1e-6), name='actions_dense_layer_1')(actions)

    net = Add(name='critic_add')([net_states, net_actions])
    net = Activation(activation, name='critic_activation')(net)

    q_values = Dense(units=1, name='q_values',
                     kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003))(net)

    model = Model(inputs=[states, actions], outputs=q_values, name='critic_model')

    model.compile(optimizer=optimizer, loss=loss)

    action_gradients = K.gradients(q_values, actions)

    get_action_gradients = K.function(
        inputs=[*model.input, K.learning_phase()],
        outputs=action_gradients)

    logger.debug(model.summary())

    return model, get_action_gradients


def cnn_model(state_size, action_size, hidden_dims=None, batch_size=32,
              activation=None, activation_last_layer=None,
              loss=None, optimizer=None):
    """
    activation: one of 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign'
    https://keras.io/api/layers/activations/
    loss: one of 'mse', 'msle', 'mape', 'categorical_crossentropy'
    https://keras.io/api/losses/
    optimizer: one of 'adam', 'nadam', 'adadelta'
    https://keras.io/api/optimizers/
    """
    hidden_dims = [32] if hidden_dims is None else hidden_dims
    activation = activation or 'relu'
    activation_last_layer = activation_last_layer or 'sigmoid'
    loss = loss or 'mse'
    optimizer = optimizer or 'adam'
    activation = activation.lower()
    activation_last_layer = activation_last_layer.lower()
    loss = loss.lower()

    assert isinstance(hidden_dims, list)
    assert activation in ACTIVATION_FUNCTIONS
    assert activation_last_layer in ACTIVATION_FUNCTIONS
    assert loss in LOSS_FUNCTIONS

    model = Sequential()
    model.add(Conv1D(filters=int(batch_size/2), kernel_size=8, input_shape=(state_size, 1),
                     kernel_initializer='uniform',
                     # activation: 'relu'
                     activation=activation, name='conv_input'))

    # activation: 'softmax'
    model.add(Activation(activation, name='activation'))
    model.add(MaxPool1D(pool_size=5, name='max_pooling'))
    model.add(Flatten())

    model.add(Dense(units=hidden_dims[0], kernel_initializer='uniform',
                    name='hidden_in', activation=activation))
    for i in range(1, len(hidden_dims)):
        model.add(Dense(units=hidden_dims[i], kernel_initializer='uniform',
                        activation=activation, name=f'layer_{i}'))

    # activation: 'sigmoid'
    model.add(Dense(units=action_size, kernel_initializer='uniform', activation=activation_last_layer, name='output'))
    # optimizer = Adam(lr=1e04)
    metrics = None
    if isinstance(loss, str) and \
            (loss.lower() == 'categorical_crossentropy' or loss.lower() == 'sparse_categorical_crossentropy'):
        metrics = ['accuracy']

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    logger.debug(model.summary())
    return model


class DQRLAgent(object):
    def __init__(self, state_size, action_size, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.001,
                 model_hidden_dims=None, memory_size=500,
                 cpu_cores=6, gpu_cores=1, cnn=False, a2c=False, batch_size=32,
                 model_summary_dir=None, random_memory_sampling=True,
                 activation=None, activation_last_layer=None,
                 loss=None, loss_critic=None, optimizer=None):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount rate
        self.tau = 0.001
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.cnn = cnn
        self.a2c = a2c
        self.memory = ReplayBuffer(state_size=state_size, size=memory_size, a2c=a2c,
                                   action_size=action_size, random_memory_sampling=random_memory_sampling)

        num_cores = cpu_cores + gpu_cores
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                                          inter_op_parallelism_threads=num_cores,
                                          allow_soft_placement=True,
                                          device_count={'CPU': cpu_cores,
                                                        'GPU': gpu_cores}
                                          )
        session = tf.compat.v1.Session(config=config)
        K.set_session(session)
        if self.cnn:
            self.model = cnn_model(state_size, action_size, model_hidden_dims, batch_size=batch_size,
                                   activation=activation, activation_last_layer=activation_last_layer,
                                   loss=loss, optimizer=optimizer)
        elif self.a2c:
            self.local_actor_model, self.local_train_fn = actor_model(self.state_size, self.action_size,
                                                                      model_hidden_dims,
                                                                      activation=activation,
                                                                      loss=loss, optimizer=optimizer)
            self.target_actor_model, self.target_train_fn = actor_model(self.state_size, self.action_size,
                                                                        model_hidden_dims,
                                                                        activation=activation,
                                                                        loss=loss, optimizer=optimizer)
            if isinstance(loss_critic, str) and \
                    (loss_critic.lower() == 'categorical_crossentropy' or
                     loss_critic.lower() == 'sparse_categorical_crossentropy'):
                loss_critic = "mse"
            self.local_critic_model, self.local_get_action_gradients = critic_model(self.state_size, self.action_size,
                                                                                    model_hidden_dims,
                                                                                    activation=activation,
                                                                                    loss=loss_critic,
                                                                                    optimizer=optimizer)
            self.target_critic_model, self.target_get_action_gradients = critic_model(self.state_size, self.action_size,
                                                                                      model_hidden_dims,
                                                                                      activation=activation,
                                                                                      loss=loss_critic,
                                                                                      optimizer=optimizer)

            self.target_critic_model.set_weights(self.local_critic_model.get_weights())
            self.target_actor_model.set_weights(self.local_actor_model.get_weights())
        else:
            self.model = ann_model(state_size, action_size, model_hidden_dims,
                                   activation=activation, activation_last_layer=activation_last_layer,
                                   loss=loss, optimizer=optimizer)

        if model_summary_dir is not None:
            model_file = open(f"{model_summary_dir}/model.txt", 'w+')

            def write_model(line):
                model_file.write(line + "\n")

            if self.a2c:
                model_file.write("Local Actor:\n")
                self.local_actor_model.summary(line_length=120, print_fn=write_model)
                model_file.write("\n\n\nLocal Critic:\n")
                self.local_critic_model.summary(line_length=120, print_fn=write_model)
                model_file.write("\n\n\nTarget Actor:\n")
                self.target_actor_model.summary(line_length=120, print_fn=write_model)
                model_file.write("\n\n\nTarget Critic:\n")
                self.target_critic_model.summary(line_length=120, print_fn=write_model)
            else:
                model_file.write("Model:\n")
                self.model.summary(line_length=120, print_fn=write_model)

            model_file.close()

    def update_replay_memory(self, state, action, reward, next_state, done):
        if self.a2c:
            action = self.local_actor_model.predict(state)

        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, mode='train'):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        if self.cnn:
            state = state.reshape(1, self.state_size, 1)

        if self.a2c:
            if mode == 'train':
                act_values = self.local_actor_model.predict(state)
            else:
                # act_values = self.target_actor_model.predict(state)
                act_values = self.local_actor_model.predict(state)
            # return act_values
        else:
            act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size=32):
        # first check if replay buffer contains enough data
        if self.memory.size < batch_size:
            return

        # sample a batch of data from the replay memory
        states, actions, rewards, next_states, done = self.memory.sample_batch(batch_size)
        # actions = to_categorical(actions, num_classes=self.action_size)
        if self.a2c:
            a2c_actions = actions.reshape(-1, self.action_size)
            a2c_states = states.reshape(-1, self.state_size)
            a2c_rewards = rewards.reshape(-1, 1)
            a2c_done = done.reshape(-1, 1)
            a2c_next_states = next_states.reshape(-1, self.state_size)

            a2c_actions_next = self.target_actor_model.predict_on_batch(a2c_next_states)
            a2c_q_targets_next = self.target_critic_model.predict_on_batch([a2c_next_states, a2c_actions_next])

            q_targets = a2c_rewards + self.gamma * a2c_q_targets_next * (1 - a2c_done)

            # print("states:", a2c_states.shape, "\n", a2c_states, "\nactions: ", a2c_actions.shape, "\n", a2c_actions,
            #       "\ntargets:", q_targets.shape, "\n", q_targets)

            self.local_critic_model.train_on_batch(x=[a2c_states, a2c_actions], y=q_targets)

            action_gradients = np.reshape(self.local_get_action_gradients([a2c_states, a2c_actions, 0]),
                                          (-1, self.action_size))
            self.local_train_fn([a2c_states, action_gradients, 1])
            self.soft_update(self.local_critic_model, self.target_critic_model)
            self.soft_update(self.local_actor_model, self.target_actor_model)
        else:
            # Calculate the tentative target: Q(s',a)
            if self.cnn:
                next_states = next_states.reshape(batch_size, self.state_size, 1)
                states = states.reshape(batch_size, self.state_size, 1)

            target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)

            # The value of terminal states is zero
            # so set the target to be the reward only
            target[done] = rewards[done]

            # We only need to update the network for the actions
            # which were actually taken.
            # We can accomplish this by setting the target to be equal to
            # the prediction for all values.
            # Then, only change the targets for the actions taken.
            # Q(s,a)
            target_full = self.model.predict(states)
            target_full[np.arange(batch_size), actions] = target
            # print("actions:", actions.shape, actions,
            #       "\ntarget:", target.shape, target,
            #       "\ntarget full:", target_full.shape, target_full)
            # target_full = to_categorical(target_full, num_classes=self.action_size)
            # print("categorical target:", target_full.shape, target_full)
            self.model.train_on_batch(states, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights)

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def load(self, path):
        if self.a2c:
            self.local_actor_model.load_weights(filepath=f'{path}/local_actor.h5')
            self.target_actor_model.load_weights(filepath=f'{path}/target_actor.h5')
            self.local_critic_model.load_weights(filepath=f'{path}/local_critic.h5')
            self.target_critic_model.load_weights(filepath=f'{path}/target_critic.h5')
        else:
            self.model.load_weights(f'{path}/dqn.h5')

    def save(self, path):
        if self.a2c:
            self.local_actor_model.save_weights(filepath=f'{path}/local_actor.h5', overwrite=True)
            self.target_actor_model.save_weights(filepath=f'{path}/target_actor.h5', overwrite=True)
            self.local_critic_model.save_weights(filepath=f'{path}/local_critic.h5', overwrite=True)
            self.target_critic_model.save_weights(filepath=f'{path}/target_critic.h5', overwrite=True)
        else:
            self.model.save(f'{path}/dqn.h5')
