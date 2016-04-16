from sklearn.manifold import TSNE
import numpy as np
from metric_learn import ITML
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

print 'starting'
iris_data = load_iris()
X = iris_data['data']
Y = iris_data['target']

randomState=13204
projectedDigits=TSNE(random_state=randomState).fit_transform(X)

itml = ITML()
plt.subplot(1,2,1)
plt.scatter(projectedDigits[:,0],projectedDigits[:,1],c=Y)
plt.title('Original Data Without ITML')

num_constraints = 200
C = ITML.prepare_constraints(Y, X.shape[0], num_constraints)
itml.fit(X, C, verbose=False)
x2=itml.transform(X)
projectedDigits=TSNE(random_state=randomState).fit_transform(x2)
plt.subplot(1,2,2)
plt.scatter(projectedDigits[:,0],projectedDigits[:,1],c=Y)
plt.title('Original Data with ITML')
plt.show()
