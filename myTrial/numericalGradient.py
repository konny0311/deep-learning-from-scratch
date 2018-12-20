import numpy as np

def function_2(x):
    return np.sum(x**2)

def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)

        x[i] = tmp -h
        fxh2 = f(x)

        grad[i] = (fxh1-fxh2) / (2*h)
        x[i] = tmp

    return grad
