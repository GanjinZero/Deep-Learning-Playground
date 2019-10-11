import numpy as np


def generate_01_data(n, p, use_n, threshold=0.8, method='linear'):
    x = np.random.randint(0, 2, (n, p))
    use = np.random.choice(p, use_n, replace=False)
    beta = np.random.rand(use_n)
    y_original = np.sum(x[:, use] * beta, axis=1)
    if method=='linear':
        y = y_original + np.random.randn(n) * 0.01
        y_min = min(y)
        y_max = max(y)
        y = [(y0 - y_min) / (y_max - y_min) for y0 in y]
        y = [min(1, y0 / threshold) for y0 in y]
    return x, y

def generate_continue_data(n, p, use_n, threshold=0.8, method='linear'):
    x = np.random.rand(n, p)
    use = np.random.choice(p, use_n, replace=False)
    beta = np.random.rand(use_n)
    y_original = np.sum(x[:, use] * beta, axis=1)
    if method=='linear':
        y = y_original + np.random.randn(n) * 0.01
        y_min = min(y)
        y_max = max(y)
        y = [(y0 - y_min) / (y_max - y_min) for y0 in y]
        y = [min(1, y0 / threshold) for y0 in y]
    return x, y

if __name__ == "__main__":
    x, y = generate_01_data(10, 5, 3)
    print(x)
    print(y)
    x, y = generate_continue_data(10, 5, 3)
    print(x)
    print(y)

