import sys
from perceptronAnd import AND
from perceptronNand import NAND
from perceptronOr import OR

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    return AND(s1,s2)

if len(sys.argv) > 2:
    print(XOR(int(sys.argv[1]),int(sys.argv[2])))
