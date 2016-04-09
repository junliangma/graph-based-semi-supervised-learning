import numpy as np
from metric_learn import ITML
from sklearn.datasets import load_iris
from scipy.sparse import rand
x = rand(10, 10)

print 'S is ',x.todense()

x=x.todense()

mat1=np.zeros(x.shape)
for i in range(0,mat1.shape[0]):
    mat1[i,i]=112.0
    for j in range(0,mat1.shape[1]):
        if i==j:
            continue
        mat1[i,j]=x[i,j]
        
print 'mat1 is ',mat1
y=np.ones((10,))
y[5:]=0

itml=ITML()
print 'X is ',mat1.shape,' y is ',y.shape
num_constraints=5
C = ITML.prepare_constraints(y, mat1.shape[0], num_constraints)
itml.fit(mat1, C, verbose=False)
xl=itml.transform(mat1)
print 'xl is ',xl

