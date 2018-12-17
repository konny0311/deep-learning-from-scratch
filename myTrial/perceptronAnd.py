import sys
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
# def AND(x1,x2):
#     w1,w2,theta = 0.5,0.5,0.7
#     tmp = w1*x1 + w2*x2
#     if tmp <= theta:
#         return 0
#     else:
#         return 1

if len(sys.argv) > 2:
    print(AND(int(sys.argv[1]),int(sys.argv[2])))
