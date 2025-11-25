import numpy as np
from multiprocessing import Process, Queue
import random

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    

    def sample(uid):
        while len(user_train[uid]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


def sample_random_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    # 1) Create *local* RNGs
    np_rng = np.random.RandomState(SEED)
    py_rng = random.Random(SEED)

    def random_neq(low, high, banned_set):
        """Draw from [low, high) but avoid items in banned_set."""
        while True:
            x = py_rng.randrange(low, high)
            if x not in banned_set:
                return x

    def sample(uid):
        # If user too short, pick another uid
        while len(user_train[uid]) <= 1:
            uid = np_rng.randint(1, usernum + 1)

        user_seq = user_train[uid][:-1]
        seq_len  = len(user_seq)

        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)

        ts = set(user_train[uid])

        if seq_len >= maxlen:
            # draw maxlen distinct indices, keep order
            sampled_idxs = np_rng.choice(seq_len, maxlen, replace=False)
            sampled_idxs.sort()
            for i, idx in enumerate(sampled_idxs):
                seq[i] = user_seq[idx]
        else:
            # pad front
            start_idx = maxlen - seq_len
            for i, item in enumerate(user_seq):
                seq[start_idx + i] = item

        # build pos/neg
        for i in range(maxlen - 1):
            pos[i] = seq[i+1]
            if pos[i] != 0:
                neg[i] = random_neq(1, itemnum + 1, ts)
        pos[-1] = 0

        return uid, seq, pos, neg

    # 2) Use that same np_rng for shuffling your uid array
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0

    while True:
        if counter % usernum == 0:
            np_rng.shuffle(uids)

        one_batch = []
        for _ in range(batch_size):
            uid = uids[counter % usernum]
            one_batch.append(sample(uid))
            counter += 1

        # unzip and put into queue
        result_queue.put(zip(*one_batch))

def sample_random_chrono_function(User, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    # 1) dedicated RNGs
    np_rng = np.random.RandomState(SEED)
    py_rng = random.Random(SEED)

    def random_neq(low, high, banned_set):
        """Draw from [low, high) but avoid items in banned_set."""
        while True:
            x = py_rng.randrange(low, high)
            if x not in banned_set:
                return x

    def sample(uid):
        user_seq = User[uid]
        seq_len  = len(user_seq)

        # 2) pick window of size maxlen+1
        window_size = maxlen + 1
        if seq_len <= window_size:
            window = user_seq[-window_size:] if seq_len > window_size else user_seq[:]
        else:
            max_start = seq_len - window_size
            start     = np_rng.randint(0, max_start + 1)
            window    = user_seq[start : start + window_size]

        # 3) left-pad if needed
        if len(window) < window_size:
            pad_len = window_size - len(window)
            window = [0]*pad_len + window

        # 4) build seq, pos, neg arrays
        seq = np.zeros(maxlen, dtype=np.int32)
        pos = np.zeros(maxlen, dtype=np.int32)
        neg = np.zeros(maxlen, dtype=np.int32)

        ts = set(user_seq)
        for i in range(maxlen):
            # tokens & next-item
            seq[i] = window[i]
            pos[i] = window[i+1] if i+1 < len(window) else 0

            # negative sampling
            neg[i] = random_neq(1, itemnum+1, ts)

        return uid, seq, pos, neg

    # 5) main loop, deterministic uid sampling
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0

    while True:
        # reshuffle uids each epoch
        if counter % usernum == 0:
            np_rng.shuffle(uids)

        batch = []
        for _ in range(batch_size):
            uid = uids[counter % usernum]
            batch.append(sample(uid))
            counter += 1

        u, s, p, n = zip(*batch)
        result_queue.put((np.array(u), np.stack(s), np.stack(p), np.stack(n)))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1,sampler_func = None):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sampler_func, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
