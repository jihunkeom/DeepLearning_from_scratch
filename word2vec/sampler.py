from collections import Counter
import numpy as np


class UnigramSampler:
    def __init__(self, corpus, sample_size):
        self.sample_size = sample_size
        counter = Counter()
        for w in corpus:
            counter[w] += 1

        self.vocab_size = len(counter)
        self.word_p = np.zeros(self.vocab_size)
        for i in range(self.vocab_size):
            self.word_p[i] = counter[i]

        self.word_p = np.power(self.word_p, 0.75)
        self.word_p /= np.sum(self.word_p)

    def get_negative_samples(self, target):
        batch_size = target.shape[0]
        ns = np.random.choice(
            self.vocab_size, (batch_size, self.sample_size), p=self.word_p)
        return ns
