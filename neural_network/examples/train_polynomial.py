import sys
sys.path.append("../")
from nn import NeuralNetwork
from pylab import *

np.random.seed(101)

def func(x, form=0):
    if form == 0:
        return x + 1.0
    if form == 1:
        return x*x + x + 1.0

xd = np.random.randn(100)
yd = func(xd, form=1)



nn = NeuralNetwork(sizes=[1, 5, 1], activation_function="sigmoid")
y = nn.eval(1.0)
print y
gamma = np.random.randn(nn.n)*0.2

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
    gamma = gamma - dJdgamma/np.abs(dJdgamma)*0.001
    if j%100 == 0:
        print j, J
        #print gamma
        
yeval = nn.veval(xd)
figure()
plot(xd, yd, 'b.')
plot(xd, yeval, 'rx')

nn.save()
nn.load()

yeval = nn.veval(xd)
plot(xd, yeval, 'g.')

show()
        

