import math
from matplotlib import pyplot as plt
import numpy as np

def integrand(x):
    return (math.exp(x)-1)/(math.exp(1)-1)


#a

# x = np.arange(0,1,0.001)
# result = map(integrand, x)
# y = [i for i in result]
# plt.plot(x,y)
# plt.legend()
# plt.plot(x,x)
# plt.xlim(xmin=0,xmax=1)
# plt.ylim(ymin=0,ymax=1)
# plt.show()


#b
vIntegrand = np.vectorize(integrand, otypes=[np.float])

n=1000000
half = n//2
crudeIn = np.random.rand(1,n)
antithIn1 = np.random.rand(1,half)
antithIn2 = np.subtract(1,antithIn1)



crudeMC = vIntegrand(crudeIn)
crudeEst = np.mean(crudeMC)
crudeVar = np.var(crudeMC)
seCrude = math.sqrt(crudeVar/n)
leftCrude = crudeEst - seCrude
rightCrude = crudeEst + seCrude

antithMC = (vIntegrand(antithIn1) + vIntegrand(antithIn2))/2
antithEst = np.mean(antithMC)
antithVar = np.var(antithMC)
seAntith = math.sqrt(antithVar/n)
leftAntith = antithEst - seAntith
rightAntith = antithEst + seAntith

efficiency = crudeVar/antithVar

print("The Crude Monte Carlo Estimator is {}".format(crudeEst))
print("The Antithetic Monte Carlo Estimator is {}".format(antithEst))
print("The Crude Monte Carlo Variance is {}".format(crudeVar))
print("The Antithetic Monte Carlo Variance is {}".format(antithVar))
print("The efficiency is {}".format(efficiency))
print("The Crude Monte Carlo 95% CI is ({},{})".format(leftCrude,rightCrude))
print("The Antithetic Monte Carlo 95% CI is ({},{})".format(leftAntith,rightAntith))
