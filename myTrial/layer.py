from crossEntropyError import cross_entropy_error
import numpy as np
from softmax import softmax
from common.util import img2col, col2im


class MulLayer:

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:

    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        #逆伝播してきた値がそのまま入力元へ遡る
        dx = dout * 1
        dy = dout * 1

        return dx, dy

class Relu:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        入力値xから<=0の箇所のみ0としてxを返す
        """
        self.mask = (x <= 0) #x<=0だとTrueになる
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self ,dout):
        """
        forwardの時にmaskに値入ってるのでそれを使う。x<=0なら0になる
        """
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):

        return dout * self.out * (1 - self.out)

class Affine:
    def __init__(self, W,b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):

        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x  = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T) #Tは転置
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx

class SoftmaxWithLoss: # TODO: 付録Aに詳細説明
    """
    各出力の総和が1になるように、出力を調整するのがsoftmax(出力=確率となる)。
    それをもとに、cross_entropy_errorを用いて、正解データとどれだけ離れているかを計算する。
    """
    def __init__(self):
        self.loss = None
        self.y = None #softmaxの出力
        self.t = None #教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: #教師データがone-hot
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

class Convolution:

    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape #filter
        N, C, H, W = x.shape
        out_h = int(1+(H+2*self.pad-FH)/self.stride)
        out_w = int(1+(W+2*self.pad-FW)/self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        #numpy.reshape()で-1を使った次元の長さは他の次元を考慮して自動的に決定される
        #今回の場合だと片方の次元がFNに決まると残りの次元が自動的に決まる
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2) #transposeは配列順番の入れ替え
        #今回の例では(N,H,W,C)0,1,2,3=>(N,C,H,W)0,3,1,2元の形に戻す

        return out

class Pooling:

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N,C,H,W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        return out

    def backward(self, dout):
        dout = dout.transpose(0,2,3,1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
