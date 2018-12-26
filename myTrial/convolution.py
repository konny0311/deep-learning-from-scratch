import sys, os
sys.path.append(os.pardir)
from common.util import img2col

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
