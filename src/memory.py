import random
from collections import deque

class Memory:
    def __init__(self, max_size):
        self._samples = deque(maxlen=max_size)

    def add_sample(self, sample):
        self._samples.append(sample)

    def get_samples(self, n):
        n = min(n, len(self._samples))
        return random.sample(list(self._samples), n)

    def __len__(self):
        return len(self._samples)