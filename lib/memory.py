import numpy as np


class ReplayBuffer:
    """
    The experience replay memory
    """

    def __init__(self, state_size, size, a2c=False,
                 action_size=None, random_memory_sampling=True):
        assert ((a2c and isinstance(action_size, int)) or not a2c)
        self._states_buffer = np.zeros([size, state_size], dtype=np.float_)
        self._states_next_buffer = np.zeros([size, state_size], dtype=np.float_)
        self._rewards_buffer = np.zeros(size, dtype=np.float_)
        self._done_buffer = np.zeros(size, dtype=np.uint8)
        self._pointer, self._max_size = -1, size

        self.a2c = a2c

        self.random_memory_sampling = random_memory_sampling

        if self.a2c:
            self._actions_buffer = np.zeros([size, action_size], dtype=np.float_)
        else:
            self._actions_buffer = np.zeros(size, dtype=np.uint8)

    def add(self, state, action, reward, next_state, done):
        if self._pointer < (self._max_size - 1):
            self._pointer += 1
        else:
            self._pointer = self._max_size - 1

            self._states_buffer = np.roll(self._states_buffer, -1, axis=0)
            self._states_next_buffer = np.roll(self._states_next_buffer, -1, axis=0)
            self._rewards_buffer = np.roll(self._rewards_buffer, -1, axis=0)
            self._done_buffer = np.roll(self._done_buffer, -1, axis=0)
            self._actions_buffer = np.roll(self._actions_buffer, -1, axis=0)

        self._states_buffer[self._pointer, :] = state
        self._states_next_buffer[self._pointer, :] = next_state
        self._rewards_buffer[self._pointer] = reward
        self._done_buffer[self._pointer] = done

        if self.a2c:
            self._actions_buffer[self._pointer, :] = action
        else:
            self._actions_buffer[self._pointer] = action

        self._pointer = self._pointer % self._max_size

    def sample_batch(self, batch_size=32):
        assert (batch_size <= self._max_size)

        if batch_size > self.size:
            batch_size = self.size

        if self.random_memory_sampling:
            ids = np.random.randint(0, self.size, size=batch_size)
        else:
            ids = range(batch_size)

        return (self._states_buffer[ids],
                self._actions_buffer[ids],
                self._rewards_buffer[ids],
                self._states_next_buffer[ids],
                self._done_buffer[ids])

    @property
    def size(self):
        return self._pointer + 1
