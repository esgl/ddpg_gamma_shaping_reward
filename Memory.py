import numpy as np

class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype="float32"):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen, ) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx > self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)

class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observation0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, (1,))
        self.terminals1 = RingBuffer(limit, (1,))
        self.observation1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observation0.get_batch(batch_idxs)
        obs1_batch = self.observation1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            "obs0": array_min2d(obs0_batch),
            "obs1": array_min2d(obs1_batch),
            "rewards": array_min2d(reward_batch),
            "actions": array_min2d(action_batch),
            "terminals1": array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return

        self.observation0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observation1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observation0)