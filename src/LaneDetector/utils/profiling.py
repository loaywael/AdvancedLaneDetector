from time import time


def timer(func):
    def timming_wrapper(*args, **kwargs):
        t1 = time()
        out = func(*args, **kwargs)
        t2 = time()
        func.n_calls += 1
        print(f">>> {func.__qualname__} runtime: {t2 - t1:.5f}s, no. calls: {func.n_calls}")
        return out
    func.n_calls = 0
    return timming_wrapper