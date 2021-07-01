import numpy as np


def get_impulse_memory_data(n_train=10000, n_test=2000):
    rng = np.random.RandomState(1234)
    seq_length = 16

    rng_train = np.random.RandomState(rng.randint(low=0, high=10000))
    rng_test = np.random.RandomState(rng.randint(low=0, high=10000))

    def make_dataset(n: int, rng_this: np.random.RandomState):
        dataset = []
        for _ in range(n):
            inputs = [np.array([1.0, 0.0], dtype=np.float32) for _ in range(seq_length)]
            gt = 0

            for i in range(seq_length):
                impulse_this_idx = rng_this.choice(2, p=[(1/16), 1 - (1/16)])
                if impulse_this_idx:
                    inputs[i] = np.array([0.0, 1.0])
                    if i >= 8:
                        gt = 1

            dataset.append((inputs, gt))

    return {
        'train': make_dataset(n_train, rng_train),
        'test': make_dataset(n_test, rng_test),
    }





