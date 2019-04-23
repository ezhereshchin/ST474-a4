import math
from matplotlib import pyplot as plt
import numpy as np

# Vectorizable functions
def bernoulli(x, p):
    if x <= p:
        return 1
    else:
        return 0

def antith(x):
    assert x >= 0 and x <= 1
    return 1 - x

# # #
# Vectorized forms
# # #

vBernoulli = np.vectorize(bernoulli, otypes=[np.float])
vAntith = np.vectorize(antith, otypes=[np.float])


# Non vectorizable functions
def mbModel(x0, u, d, p, th, vector = None):
    process = np.empty(th+1)
    process[0] = x0
    if vector is None:
        vector = vBernoulli(np.random.uniform(0, 1, th), p)
    for i in range(1, th+1):
        if vector[i - 1] == 1:
            process[i] = process[i - 1] * u
        else:
            process[i] = process[i - 1] * d
    return process


def runningMin(process, i, th):
    assert i >= 0 and i <= th
    subProcess = process[:i + 1]
    return np.min(subProcess)

def indicator(process):
    if runningMin(process, 10, th) <= 85:
        return max(80 - process[20], 0)
    else:
        return 0


n = 100000
p = 0.65
th = 20
u = 1.05
d = 1 / u
x0 = 100

# # #
# Crude Estimator
# # #

trials = np.empty(n)
for i in range(n):
    process = mbModel(x0, u, d, p, th)
    trials[i] = indicator(process)

expectedValue = np.mean(trials)
sampleVariance = np.var(trials, ddof=1)
leftSide = expectedValue - 1.96 * np.sqrt(sampleVariance / n)
rightSide = expectedValue + 1.96 * np.sqrt(sampleVariance / n)
confidenceInterval = (leftSide, rightSide)

crude = (expectedValue, sampleVariance, confidenceInterval)
print(crude)

# # #
# Antithetic Estimator
# # #

trials = np.empty(n//2)
for i in range(n//2):
    path = np.random.uniform(0, 1, th)
    process = mbModel(x0, u, d, p, th, vector = vBernoulli(path, p))
    antProcess = mbModel(x0, u, d, p, th, vector = vBernoulli(vAntith(path), p))
    trials[i] = (indicator(process) + indicator(antProcess))/2

expectedValue = np.mean(trials)
sampleVariance = np.var(trials, ddof=1)
leftSide = expectedValue - 1.96 * np.sqrt(sampleVariance / n)
rightSide = expectedValue + 1.96 * np.sqrt(sampleVariance / n)
confidenceInterval = (leftSide, rightSide)

antithetic = (expectedValue, sampleVariance, confidenceInterval)
print(antithetic)
antEfficiency = crude[1]/antithetic[1]
print(antEfficiency)


# # #
# Importance Estimators
# # #