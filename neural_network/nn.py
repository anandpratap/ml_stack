import numpy as np
import adolc as ad
import pickle
from utils import calc_jacobian
from activation import activation_functions

class NeuralNetwork(object):
    """
    A simple feed forward neural network.
    Input:
    sizes: list containing number of neurons in each layer.
            Example: [1, 10, 1] contains input layer with 1 inputs,
            a hidder layer with 10 neurons, and a output layer with 1 output.
    activation_function: default = tanh
                         options: sigmoid, tanh, atan, identity
    
    """
    def __init__(self, sizes = [2, 10, 1], activation_function="tanh"):
        self.nlayer = len(sizes)
        self.sizes = sizes
        
        self.weights = []
        self.biases = []

        self.dweights = []
        self.dbiases = []

        self.nw = 0
        self.nb = 0
        for i in range(1, self.nlayer):
            weights = np.random.randn(sizes[i], sizes[i-1])
            bias = np.random.randn(sizes[i])

            self.nw += weights.size
            self.nb += bias.size
            self.weights.append(weights)
            self.biases.append(bias)
        self.n = self.nw + self.nb
        self.activation_function = activation_function

    def get_nhyperparameters(self):
        return self.n

    def get_nweights(self):
        return self.nw

    def get_nbiases(self):
        return self.nb
        
    def veval(self, *args):
        """Vectorize version of the function eval to be
           used with numpy array as inputs
        """
        np_veval = np.vectorize(self.eval)
        return np_veval(*args)

    def activation(self, x):
        return activation_functions[self.activation_function](x)
    
    def eval(self, *args):
        """Evaluate the neural network.
        Example: nn.eval(x1, x2, x3)
        """
        x = np.array(args).T
        assert x.size == self.sizes[0]
        for i in range(1, self.nlayer):
            y = np.dot(self.weights[i-1], x) + self.biases[i-1]
            if i == self.nlayer-1:
                x = y
            else:
                x = self.activation(y)
        return x[0]
    
    def set_from_array(self, gamma):
        """Set weights and biases from a numpy array of size self.n.
           All weights are assigned first followed by biases.
        """
        assert gamma.size == self.get_nhyperparameters()
        start = 0
        for i in range(1, self.nlayer):
            end = start + self.weights[i-1].size
            self.weights[i-1] = np.reshape(gamma[start:end], self.weights[i-1].shape)
            start = end

        for i in range(1, self.nlayer):
            end = start + self.biases[i-1].size
            self.biases[i-1] = np.reshape(gamma[start:end], self.biases[i-1].shape)
            start = end
    def get_array(self):
        gamma = np.zeros(self.n)
        start = 0
        for i in range(1, self.nlayer):
            end = start + self.weights[i-1].size
            gamma[start:end] = np.reshape(self.weights[i-1], (self.weights[i-1].size,))
            start = end

        for i in range(1, self.nlayer):
            end = start + self.biases[i-1].size
            gamma[start:end] = np.reshape(self.biases[i-1], (self.biases[i-1].size,))
            start = end
        return gamma
            
    def dydgamma(self, x, gamma):
        """
        Calculate derivative of the neural network output with respect to the
        hyperparameters gamma.
        """
        gamma_c = gamma.copy()
        gamma = ad.adouble(gamma)
        tag = 11
        ad.trace_on(tag)
        ad.independent(gamma)
        self.set_from_array(gamma)
        y = self.eval(x)
        ad.dependent(y)
        ad.trace_off()
        gamma = gamma_c
        self.set_from_array(gamma_c)
        dJdgamma = calc_jacobian(gamma, tag=tag, sparse=False)
        return dJdgamma.reshape(gamma_c.shape)


    def save(self, filename="network.nn"):
        """Save neural network to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)


    def load(self, filename="network.nn"):
        """Load neural network from a pickle file."""
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict) 
        assert self.n == self.nb + self.nw
        assert self.nlayer == len(self.sizes)
        

        
if __name__ == "__main__":
    def func(x):
        return 2.0*x + 10.0

    xd = np.random.randn(100)
    yd = func(xd)

    from pylab import *

    #figure()
    #plot(x, y, 'x')
    #show()
    
    nn = NeuralNetwork(sizes=[1, 1])
    y = nn.eval(np.array([1.0]))
    print y
    gamma = np.random.randn(nn.n)*0.02

    for j in range(10000):
        nn.set_from_array(gamma)
        dJdgamma = np.zeros_like(gamma)
        J = 0.0
        for i in range(len(xd)):
            xin = xd[i]
            yeval = nn.eval(xin)
            #print yeval
            J += (yeval - yd[i])**2
            dydgamma = nn.dydgamma(xin, gamma)
            dJdgamma += 2.0*(yeval - yd[i])*dydgamma
            
        gamma = gamma - dJdgamma/np.abs(dJdgamma).max()*0.01
        if j%100 == 0:
            print j, J
        #print gamma

    yeval = []
    for i in range(len(xd)):
        xin = np.array([xd[i]])
        yeval.append(nn.eval(xin))
    figure()
    plot(xd, yd, 'b.')
    plot(xd, yeval, 'rx')

    nn.save()
    nn.load()

    yeval = []
    for i in range(len(xd)):
        xin = np.array([xd[i]])
        yeval.append(nn.eval(xin))
    plot(xd, yeval, 'g.')

    show()
        
