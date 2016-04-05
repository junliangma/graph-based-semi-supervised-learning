import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold                                                                    
from copy import copy



class GraphBasedLearning:
    
    def __init__(self,X_train,y_train,x_test,y_test):
        self.x_train=copy(X_train)
        self.y_train=copy(y_train)
        self.x_test=copy(x_test)
        self.y_test=copy(y_test)
        
        
    
    def constructSimilartyMatrixITML(self):
        
    

data=np.load('FullData.npy')
labels=np.load('Labels.npy')

print 'data.shape is ',data.shape
print 'labels.shape is ',labels.shape
#data=data.tolist()
labels=labels.tolist()

print 'labels are ',labels

#for graph based reasoning , replace every 0 with -1

newLabels=[]
for label in labels:
    if label==1:
        newLabels.append(label)
    else:
        newLabels.append(-1)

print 'newLabels are ',newLabels

newLabels=np.asarray(newLabels)
np.save('NewLabels',newLabels)

    

skf=StratifiedKFold(newLabels,n_folds=2,shuffle=True)




for train_index,test_index in skf:
    X_train,X_test=data[train_index],data[test_index]
    print 'X_train is ',X_train.shape
    print 'X_test is ',X_test.shape
