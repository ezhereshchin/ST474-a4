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

# Plot 3 Antithetic paths
fig = plt.figure()
axes = (fig.add_subplot(121), fig.add_subplot(122))

axes[0].set_xlabel("Time")
axes[0].set_ylabel("X")

axes[1].set_xlabel("Time")
axes[1].set_ylabel("m")

markers = ["^", "^", "^"]
mMarkers = ["^", "^", "^"]
colours = [
    ("r", "c"),
    ("b", "y"),
    ("g", "m")
]
labels = [
    ("Path 1", "A-Path 1"),
    ("Path 2", "A-Path 2"),
    ("Path 3", "A-Path 3")
]
# for i in range(3):
#     path = np.random.uniform(0, 1, th)
#     process = mbModel(x0, u, d, p, th, vector = vBernoulli(path, p))
#     antProcess = mbModel(x0, u, d, p, th, vector = vBernoulli(vAntith(path), p))
#     axes[0].scatter(np.arange(0, th + 1), process, marker = markers[i], c = colours[i][0], label = labels[i][0])
#     axes[0].scatter(np.arange(0, th + 1), antProcess, marker = markers[i], c = colours[i][1], label = labels[i][1])
#     # Generate running min of path
#     mins = np.empty(th + 1)
#     antMins = np.empty(th + 1)
#     for j in range(0,th+1):
#         mins[j] = runningMin(process, j, th)
#         antMins[j] = runningMin(antProcess, j, th)
#     axes[1].scatter(np.arange(0, th + 1), mins, marker = mMarkers[i], c = colours[i][0], label = labels[i][0])
#     axes[1].scatter(np.arange(0, th + 1), antMins, marker = mMarkers[i], c = colours[i][1], label = labels[i][1])
    
# axes[0].legend(loc = "upper left")
# axes[1].legend(loc = "lower left")

# plt.show()
# plt.clf()




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

# Generating sample paths

# axes = (fig.add_subplot(121), fig.add_subplot(122))

path = np.random.uniform(0, 1, th)
ps = [0.65, 0.5, 0.35]
for i in range(3):
    # path = np.random.uniform(0, 1, th)
    process = mbModel(x0, u, d, p, th, vector = vBernoulli(path, ps[i]))
    # antProcess = mbModel(x0, u, d, p, th, vector = vBernoulli(vAntith(path), p))
    axes[0].scatter(np.arange(0, th + 1), process, marker = markers[i], c = colours[i][0], label = labels[i][0])
    # axes[0].scatter(np.arange(0, th + 1), antProcess, marker = markers[i], c = colours[i][1], label = labels[i][1])
    # Generate running min of path
    mins = np.empty(th + 1)
    antMins = np.empty(th + 1)
    for j in range(0,th+1):
        mins[j] = runningMin(process, j, th)
        # antMins[j] = runningMin(antProcess, j, th)
    axes[1].scatter(np.arange(0, th + 1), mins, marker = mMarkers[i], c = colours[i][0], label = labels[i][0])
    # axes[1].scatter(np.arange(0, th + 1), antMins, marker = mMarkers[i], c = colours[i][1], label = labels[i][1])

axes[0].legend(loc = "upper left")
axes[1].legend(loc = "upper right")

plt.show()

def importanceEstimator(x0, u, d, p, q, th, n):
    trials = np.empty(n)
    for i in range(n):
        uPath = np.random.uniform(0, 1, th)
        vector = vBernoulli(uPath, q)
        process = mbModel(x0, u, d, p, th, vector=vector)
        s = np.sum(vector)
        trials[i] = indicator(process) * ( (p/q) ** s) * ( ((1-p)/(1-q)) ** (th - s) )
    return trials

q = 0.5
trials = importanceEstimator(x0, u, d, p, q, th, n)
expectedValue = np.mean(trials)
sampleVariance = np.var(trials, ddof=1)
leftSide = expectedValue - 1.96 * np.sqrt(sampleVariance / n)
rightSide = expectedValue + 1.96 * np.sqrt(sampleVariance / n)
confidenceInterval = (leftSide, rightSide)
imp = (expectedValue, sampleVariance, confidenceInterval)
impEfficiency = crude[1]/imp[1]
print(imp)
print(impEfficiency)

q = 0.35
trials = importanceEstimator(x0, u, d, p, q, th, n)
expectedValue = np.mean(trials)
sampleVariance = np.var(trials, ddof=1)
leftSide = expectedValue - 1.96 * np.sqrt(sampleVariance / n)
rightSide = expectedValue + 1.96 * np.sqrt(sampleVariance / n)
confidenceInterval = (leftSide, rightSide)
imp = (expectedValue, sampleVariance, confidenceInterval)
impEfficiency = crude[1]/imp[1]
print(imp)
print(impEfficiency)
