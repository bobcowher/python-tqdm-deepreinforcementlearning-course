import numpy as np

class ReplayBuffer(object):

    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size

    def add(self, transition):
        if len(self.storage) < self.max_size:
            self.storage.append(transition)
        else:
            self.storage.remove(self.storage[0])
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        batch_states = np.array(batch_states)
        batch_next_states = np.array(batch_next_states)
        batch_actions = np.array(batch_actions)
        batch_rewards = np.array(batch_rewards).reshape(-1, 1)
        batch_dones = np.array(batch_dones)

        return batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones

    def can_sample(self, batch_size):
        if len(self.storage) > batch_size * 10:
            return True
        else:
            return False