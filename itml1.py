import numpy as np
from metric_learn import ITML
from sklearn.datasets import load_iris

iris_data = load_iris()
X = iris_data['data']
Y = iris_data['target']

print 'Y is ',Y.shape
print 'X.shape is ',X.shape
itml = ITML()

num_constraints = 200
C = ITML.prepare_constraints(Y, X.shape[0], num_constraints)
itml.fit(X, C, verbose=False)
x2=itml.transform(X)

print 'x2 is ',x2
l=itml.transformer()
print '\n\n\nafter transforming is ',np.dot(X,l.T)
