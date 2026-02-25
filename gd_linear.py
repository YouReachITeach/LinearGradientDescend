import numpy as np
# Loss: SUM((y-yhat)^2)
#1-dimensional. yhat = wx +b
#Difference Batch & SGD: Looking at all datapoints every time -> takes longer per descend but better overview
#   vs looking at only one datapoint each descend -> less overview but better performance
#   the Gradient stays two-dimensional for this example.

x = np.random.randn(10, 1)

#y = f(x) + e   where e is totally random
#f(x) = bx     where it could also have a constant shift e.g. +2
b = 2
y = (b * x) + 0.5 * np.random.randn(10, 1)

#f(x) != yhat = cx + d  we want to predict f(x) without e, because you can't predict e
#L(c,d)= SUM((y-yhat)^2) = SUM((y-(cx+d))^2) has to be minimized, so derive and put in descend method.
c = 0.0
d = 0.0

#params
descendRate = 0.01
tol = 0.0001
runs = 400

def descendBatch(x, y, c, d, descendRate):
    betaC = 0
    betaD = 0
    for i in range(x.shape[0]):
        betaC += x[i] * (y[i] - (c * x[i] + d))
        betaD += y[i] - (c * x[i] + d)
    betaC *= -2 / x.shape[0]
    betaD *= -2 / x.shape[0]
    return (c - descendRate * betaC), (d - descendRate * betaD)

#Batch GD
epCount = 0
for epoch in range(runs):
    epCount += 1
    state = descendBatch(x, y, c, d, descendRate)
    c_new = state[0]
    d_new = state[1]
    if abs(c - c_new) + abs(d - d_new) < tol:
        break
    else:
        c = c_new
        d = d_new
c_batch = c
d_batch = d

#Stochastic Gradient Descent for better CPU optimization
c = 0
d = 0
for epoch in range(runs):
    for i in np.random.permutation(x.shape[0]):
        state = descendBatch(x[i], y[i], c, d, descendRate)
        c = state[0]
        d = state[1]

print(f"actual slope: {b}, actual offset: {0}")
print(f"batch: c = {c_batch}, d = {d_batch}, error = {b-c_batch}")
print(f"SGD: c = {c}, b = {d}, error = {b-c}")
print(f"x0 = {x[0]}")
print(f"y0 = {y[0]}")
print(f"y0hat batch = {(c_batch * x[0] + d_batch)}")
print(f"y0hat mini = {(c * x[0] + d)}")


