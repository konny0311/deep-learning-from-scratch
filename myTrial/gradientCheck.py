import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

gradNumerical = network.numerical_gradient(x_batch, t_batch)
gradBackprop = network.gradient(x_batch, t_batch)

for key in gradNumerical.keys():
    diff = np.average(np.abs(gradBackprop[key] - gradNumerical[key]))
    print(key + ':' + str(diff))
