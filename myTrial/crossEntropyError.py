import numpy as np

def cross_entropy_error(y,t):
    """
    出力結果が正解データと比較してどれだけ「悪いか」を数値化する。
    tはone-hotで正解インデックスのみが1,他は0。
    従って、正解インデックスをiとすると-logy[i](>0)が出力される。
    他のyの要素は無視される。
    0に近い方が正解に近い。
    Parameters
    ---------
    y : np.ndarray
        出力結果
    t : np.ndarray
        正解データ

    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7 #オーバーフローを防ぐために微小値を足す
    return -np.sum(t * np.log(y + delta)) / batch_size

    # return -np.sum(t*np.log(y+delta))
