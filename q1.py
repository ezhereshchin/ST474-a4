import math
from matplotlib import pyplot as plt
import numpy as np

N = 10000

def integrand(x):
    return (math.exp(x)-1)/(math.exp(1)-1)


vIntegrand = np.vectorize(integrand, otypes=[np.float])

# a


def plotIntegrand():
    x = np.arange(0, 1, 0.001)
    result = map(integrand, x)
    y = [i for i in result]
    plt.plot(x, y, label = "f(x)")
    plt.plot(x, x, label = "g(x)=x")
    plt.xlim(xmin=0, xmax=1)
    plt.ylim(ymin=0, ymax=1)
    plt.title("Integrand vs g(x)=x")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend( )
    plt.show()
    return
# plotIntegrand()
# b


def crude():
    global N
    n = N
    crudeIn = np.random.rand(1, n)
    crudeMC = vIntegrand(crudeIn)
    crudeEst = np.mean(crudeMC)
    crudeVar = np.var(crudeMC)
    seCrude = 1.96*np.sqrt(crudeVar/n)
    leftCrude = crudeEst - seCrude
    rightCrude = crudeEst + seCrude
    crudeInterval = (leftCrude, rightCrude)
    # print("The Crude Monte Carlo Estimator is {}".format(crudeEst))
    # print("The Crude Monte Carlo Variance is {}".format(crudeVar))
    # print("The Crude Monte Carlo 95% CI is ({},{})".format(leftCrude,rightCrude))
    return (crudeEst, crudeVar, crudeInterval)

#c
def anti():
    global N
    n = N
    half = n//2

    antithIn1 = np.random.rand(1, half)
    antithIn2 = np.subtract(1, antithIn1)
    antithMC = (vIntegrand(antithIn1) + vIntegrand(antithIn2))/2
    antithEst = np.mean(antithMC)
    antithVar = np.var(antithMC)
    seAntith = 1.96*np.sqrt(antithVar/n)
    leftAntith = antithEst - seAntith
    rightAntith = antithEst + seAntith
    antithInterval = (leftAntith, rightAntith)

    # print("The Antithetic Monte Carlo Estimator is {}".format(antithEst))
    # print("The Antithetic Monte Carlo Variance is {}".format(antithVar))
    # print("The Antithetic Monte Carlo 95% CI is ({},{})".format(leftAntith,rightAntith))

    return (antithEst, antithVar, antithInterval)

# d


def strat(a, split=0.5):  # works only with equal split
    global N
    n = N
    first = int(n*split)
    second = n-first

    stratIn1 = np.random.rand(1, first)
    stratIn1 = stratIn1*a
    stratIn2 = np.random.rand(1, second)
    stratIn2 = a + (1-a)*stratIn2

    # Strsim=a*fn_call_integrand(a*rand(1,500000))+(1-a)*fn_call_integrand(a+(1-a)*rand(1,500000));

    # stratMC1 = a*vIntegrand(stratIn1)
    # stratMC2 = (1-a)*vIntegrand(stratIn2)
    # stratMC = np.concatenate([stratMC1,stratMC2],axis=1)
    stratMC = a*vIntegrand(stratIn1) + (1-a)*vIntegrand(stratIn2)
    stratEst = np.mean(stratMC)
    stratVar = np.var(stratMC)
    seStrat = 1.96*np.sqrt(stratVar/n)
    leftStrat = stratEst - seStrat
    rightStrat = stratEst + seStrat
    stratInterval = (leftStrat, rightStrat)
    # print("The Stratified Monte Carlo Estimator is {}".format(stratEst))
    # print("The Stratified Monte Carlo Variance is {}".format(stratVar))
    # print("The Stratified Monte Carlo 95% CI is ({},{})".format(leftStrat, rightStrat))
    return (stratEst, stratVar, stratInterval)


def gen_strat(a, sizes):
    global N
    n = N
    assert sum(sizes) == n

# % for i=1:length(n)
# %     xx=fn_call_integrand(a(i)+(a(i+1)-a(i))*rand(1,n(i)));
# %     pr_GStr=pr_GStr+(a(i+1)-a(i))*mean(xx);
# %     var_pr_GStr=var_pr_GStr + (a(i+1)-a(i))^2*var(xx)./n(i);
    gStratEst = 0
    gStratVar = 0
    for i in range(len(sizes)):
        evals = vIntegrand(a[i] + (a[i+1]-a[i])*np.random.rand(1, sizes[i]))
        gStratEst += (a[i+1]-a[i])*np.mean(evals)
        gStratVar += (a[i+1]-a[i])**2*np.var(evals)/sizes[i]
    seGStrat = 1.96*np.sqrt(gStratVar/n)
    leftGStrat = gStratEst - seGStrat
    rightGtrat = gStratEst + seGStrat
    return (gStratEst, gStratVar, (leftGStrat, rightGtrat))

# a = 0.8
# split = 0.5

# efficiencies = [(a,crude()[1]/strat(a,0.5)[1]) for a in np.arange(0.1,0.9,0.001)]
# print(max(efficiencies,key = lambda item:item[1]))


def get_best_n(a):
    n = np.empty(3, dtype=int)
    var = np.array([np.var(vIntegrand(a[i] + (a[i+1]-a[i]) * np.random.rand(1, 1000))) for i in range(len(n))])
    weights = np.array([a[i+1]-a[i] for i in range(len(n))])
    allWeightVars = np.sum(np.sqrt(var)*weights)

    for i in range(len(n)):
        if i != len(n)-1:
            n[i] = int(N*weights[i]*np.sqrt(var[i])/allWeightVars)
        else:
            n[i] = N-sum(n[0:i])
    return n

# result = gen_strat(a,get_best_n(a))
# print(result)
# print(crude()[1]/result[1])


# best a is [0,0.4,0.7,1], resulting best n of [35349, 28184, 36467]
def get_best_a():
    max_eff = 1
    best_a = None
    best_est = 0
    for i in np.arange(0.1, 0.9, 0.01):
        for j in np.arange(i, 0.95, 0.01):

            a = [0, i, j, 1]
            n = get_best_n(a)
            result = gen_strat(a, n)
            eff = crude()[1]/result[1]
            if eff > max_eff:
                max_eff = eff
                best_a = a
                best_est = result[0]
    return (best_a, max_eff, best_est)

# N = 100000
# a = [0,0.4,0.7,1]
# n = [35349, 28184, 36467]
# result = gen_strat(a,n)
# print(result)
# print(n)
# print(crude()[1]/result[1])


def comb():
    global N
    n = N
    half = n//2
    combIn1 = np.divide(np.random.rand(1, half), 2)
    combIn2 = np.subtract(1, combIn1)
    combMC = 1/2*(vIntegrand(combIn1)+vIntegrand(combIn2))
    combEst = np.mean(combMC)
    combVar = np.var(combMC)
    seComb = 1.96*np.sqrt(combVar/n)
    leftComb = combEst - seComb
    rightComb = combEst + seComb
    combInterval = (leftComb, rightComb)

    return (combEst, combVar, combInterval)

# result = comb()
# print(result[0],crude()[1]/result[1])


def control(beta):
    global N
    n = N
    half = n//2
    contrIn1 = np.random.rand(1, half)
    contrIn2 = np.subtract(contrIn1, 0.5)
    contrMC = vIntegrand(contrIn1) - beta*(contrIn2)
    contrEst = np.mean(contrMC)
    contrVar = np.var(contrMC)
    seContr = 1.96*np.sqrt(contrVar/n)
    leftContr = contrEst - seContr
    rightContr = contrEst + seContr
    contrInterval = (leftContr, rightContr)
    return (contrEst, contrVar, contrInterval)


# result = control(1)
# print(result[0],crude()[1]/result[1])


# def impMC():
#     global N
#     n = N
#     impIn = np.random.rand(1, n)
#     g = np.sqrt(impIn)
#     impMC = vIntegrand(g)/(g ** 2)
#     impEst = np.mean(impMC)
#     impVar = np.var(impMC)
#     seImp = 1.96*np.sqrt(impVar/n)
#     leftImp = impEst - seImp
#     rightImp = impEst + seImp
#     impInterval = (leftImp, rightImp)
#     return (impEst, impVar, impInterval)

# result = impMC()
# print(result[0],crude()[1]/result[1])   


def anMC(u):
    return 1/2 * (vIntegrand(u) + vIntegrand(1-u))

def stranMC(u):
    return 1/4* (vIntegrand(u/2)+vIntegrand((1-u)/2)+vIntegrand((1+u)/2)+vIntegrand(1-u/2))

def impMC(u): #using g(x)=2x
    g = np.sqrt(u)
    return vIntegrand(g)/u*2

def contrMC(u):
    return vIntegrand(u)-(u-0.5)


vAnMC = np.vectorize(anMC, otypes=[np.float])
vstranMC = np.vectorize(stranMC, otypes=[np.float])
vimpMC = np.vectorize(impMC, otypes=[np.float])
vcontrMC= np.vectorize(contrMC, otypes=[np.float])

def lin_comb():
    global N
    n = N
    linIn = np.random.rand(1, n)
    T1 = vAnMC(linIn)
    T2 = vstranMC(linIn)
    T3 = vimpMC(linIn)
    T4 = vcontrMC(linIn)
    X = np.squeeze(np.array([T1,T2,T3,T4]))
    V = np.cov(X)
    V1 = np.linalg.inv(V)
    ones = np.ones((X.shape[0],1))
    left = np.matmul(V1,ones)
    right = np.matmul(np.matrix.transpose(ones),V1)
    right = np.matmul(right,ones)
    b = np.divide(left,right)
    linMC = T1*b[0]+T2*b[1]+T3*b[2]+T4*b[3]
    linEst = np.mean(linMC)
    linVar = 1/right
    seLin = 1.96 * (np.sqrt(linVar/n))
    leftLin = linEst - seLin
    rightLin = linEst + seLin
    linInterval = (leftLin, rightLin)
    return (linEst, linVar, linInterval, b)

    
# result = lin_comb()
# print(result[0],result[3],crude()[1]/result[1])

