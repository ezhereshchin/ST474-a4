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

n=1000
half = n//2
crudeIn = np.random.rand(1,n)
antithIn1 = np.random.rand(1,half)
antithIn2 = np.subtract(1,antithIn1)
antithIn = np.concatenate((antithIn1,antithIn2),axis=0)


crudeMC = vIntegrand(crudeIn)
antithMC = vIntegrand(antithIn)
curdeEst = np.mean(crudeMC)
antithEst = np.mean(antithMC)
crudeVar = np.var(crudeMC)
antithVar = np.var(antithMC)
efficiency = antithVar/crudeVar

print("The Crude Monte Carlo Estimator is {}".format(curdeEst))
print("The Antithetic Monte Carlo Estimator is {}".format(antithEst))
print("The Crude Monte Carlo Variance is {}".format(crudeVar))
print("The Antithetic Monte Carlo Variance is {}".format(antithVar))
print("The efficiency is {}".format(efficiency))


%Antithetic Estimator
u=rand(1,500000);
Ansim=0.5.*(fn_call_integrand(u)+fn_call_integrand(1-u));
pr_An=mean(Ansim);
var_pr_An=var(Ansim);
se_pr_An=sqrt(var(Ansim)./length(u));
CI_An=[pr_An-1.96*se_pr_An pr_An+1.96*se_pr_An];
Eff_An=var_pr_CR./(var_pr_An);
