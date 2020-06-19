from time import perf_counter


def timer(func):
    def timming_wrapper(*args, **kwargs):
        t1 = perf_counter()
        out = func(*args, **kwargs)
        t2 = perf_counter()
        func.n_calls += 1
        print(f">>> {func.__qualname__} runtime: {t2 - t1:.3f}s, no. calls: {func.n_calls}")
        print("-"*75)
        return out
    func.n_calls = 0
    return timming_wrapper