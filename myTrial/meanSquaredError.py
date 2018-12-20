import numpy as np

def mean_squared_error(y,t):
    """
    出力結果が正解データと比較してどれだけ「悪いか」を数値化する。
    1/2してる理由:http://yaju3d.hatenablog.jp/entry/2016/12/09/073050
    Parameters
    ---------
    y : np.ndarray
        出力結果
    t : np.ndarray
        正解データ

    """
    return 0.5 * np.sum((y-t)**2)
