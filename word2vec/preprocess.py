import numpy as np


def preprocess(x):
    x = x.lower()
    x = x.replace(".", " .")
    x = x.split(" ")

    w2id, id2w = {}, {}
    for w in x:
        if w not in w2id.keys():
            idx = len(w2id)
            w2id[w] = idx
            id2w[idx] = w

    corpus = np.array([w2id[w] for w in x])
    return corpus, w2id, id2w


def create_contexts_target(corpus, window_size):
    target = corpus[window_size: -window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t != 0:
                cs.append(corpus[idx+t])

        contexts.append(cs)

    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    N = corpus.shape[0]

    if corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size))
        for i, word_ids in enumerate(corpus):
            for j, word_id in enumerate(word_ids):
                one_hot[i, j, word_id] = 1

    else:
        one_hot = np.zeros((N, vocab_size))
        for i, word_id in enumerate(corpus):
            one_hot[i, word_id] = 1

    return one_hot
