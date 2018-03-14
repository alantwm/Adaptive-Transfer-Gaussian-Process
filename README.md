# Adaptive-Transfer-Gaussian-Process

ATGP is a gaussian process-based supervised transfer learning algorithm based on a paper by Cao et. al. This implementation  utilizes the squared exponential kernel function and the associated hyperparameters are optimized using global search algorithm (CMAES).

ATGP takes as inputs (x_target,y_target,x_source,y_source).

Accepted inputs are shaped (n,d); n = # of instances, d = dimensions. Accepted outputs are shaped (n,1)

Example Use:
```matlab
model = ATGP(x_target,y_target,x_source,y_source)
yhat = model.predict(x_test)
```

Note: Scripts were tested in Matlab R2015b.

### Described in detail in:
Cao, Bin, Sinno Jialin Pan, Yu Zhang, Dit-Yan Yeung, and Qiang Yang. "Adaptive Transfer Learning." In AAAI, vol. 2, no. 5, p. 7. 2010.
