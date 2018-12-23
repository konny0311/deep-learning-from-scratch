import numpy as np

def function_2(x):
    return np.sum(x**2)

def numerical_gradient(f,x):
    """
    数値微分を行う。あるxの値の前後h(微小)での変化量を返す
    """
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    #np.nditer(flags=['multi_index'])は多次元の各要素に注目する際に、全ての次元でforする必要無く総当たりしてくれる。
    #要素を変更する時はop_flags=['readwrite']が必要。readonlyだと変更できない
    while not it.finished:
        i = it.multi_index
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)

        x[i] = tmp -h
        fxh2 = f(x)

        grad[i] = (fxh1-fxh2) / (2*h)
        x[i] = tmp

        it.iternext()

    return grad
