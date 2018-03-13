# Adaptive-Transfer-Gaussian-Process

ATGP is a gaussian process-based supervised transfer learning algorithm based on a paper by Cao et. al.

ATGP takes as inputs (x_target,y_target,x_source,y_source)
Accepted inputs are shaped (n,d); n = # of instances, d = dimensions. Accepted outputs are shaped (n,1)

Example Use:
'''
model = ATGP(x_target,y_target,x_source,y_source)
yhat = model.predict(x_test)'''
