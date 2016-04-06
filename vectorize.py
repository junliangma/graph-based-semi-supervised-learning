import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold   
from nltk.stem.porter import PorterStemmer                                                                 
from copy import copy
from nltk import word_tokenize
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import write_dot
from sklearn import svm

data=np.load('FullData.npy')
labels=np.load('NewLabels.npy')

print 'data.shape is ',data.shape
print 'labels.shape is ',labels.shape
#data=data.tolist()
#labels=labels.tolist()

print 'labels are ',labels
class StemmerTokenizer(object):

    def __init__(self): 
        self.stemmer = PorterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc)]
        
class GraphBasedLearning:
    
    def __init__(self,X_train,y_train,x_test,y_test):
        self.x_train=copy(X_train)
        self.y_train=copy(y_train)
        self.x_test=copy(x_test)
        self.y_test=copy(y_test)
        self.data=copy(np.vstack((self.x_train,self.x_test)))
        self.labels=copy(np.hstack((self.y_train,self.y_test)))
        print 'labels are ',self.labels
        self.Vectorize()
        self.constructSimilartyMatrixCosine()
        print 'now doing label propogation\n'
        self.labelPropogation()
        self.compareWithSvm()
        
    def Vectorize(self):
        self.vectorizer = TfidfVectorizer(decode_error='replace',analyzer='word',stop_words='english',lowercase=True,tokenizer=StemmerTokenizer())
     
        self.x2=[]
        for doc in self.x_train:
            #print 'doc is ',doc[0],'\n\n'
            self.x2.append(doc[0])
        self.data2=[]
        for doc in self.data:   
            self.data2.append(doc[0])
        #print 'self.data[0:2] is ',self.data
        self.xtest2=[]
        for doc in self.x_test:  
            self.xtest2.append(doc[0])
        
        self.vectorizer.fit(self.x2)
        #print 'self.x2 is ',len(self.x2)
        self.trainVectors=self.vectorizer.transform(self.x2)
        print 'train vectors are ',self.trainVectors.shape
        self.testVectors=self.vectorizer.transform(self.xtest2)
        self.allVectors=self.vectorizer.transform(self.data2)
        print 'allVectors are ',self.allVectors.shape
        
    
    def constructSimilartyMatrixCosine(self,k=20):
        #This is a simpole k nearest neighbour approach based on the cosine distance
        #for this take the full dataset and find the pairwise_distances between each of the nodes
        #then find the k nearest neighbours for each node 
        self.pwdis=pairwise_distances(self.allVectors,metric='cosine')
        #now we have all the pairwise cosine distances between all the sentences
        #now we need to do a knnNeighbour search
        #now we can construct the diagonal weight marix , which has the sum of all the weights
        self.D=np.zeros(self.pwdis.shape)
        for i in range(0,self.pwdis.shape[0]):
            l1=self.pwdis[i].tolist()
            #print 'l1 is ',l1,'\n\n'
            allnearestNeighbours=sorted(range(len(l1)),key=lambda i : l1[i])
            #now set the all the weights except for k+1 to 0
            self.pwdis[i,allnearestNeighbours[k:]]=0
            self.D[i,i]=sum(self.pwdis[i])
            
            
        
        #now we have the weight matrix graph based on the cosine distance
        #print 'self.D is ',self.D
    
    
    def checkAccuracy(self,predicted,goldset):
        predicted=predicted.tolist()
        goldset=goldset.tolist()
        correct=0
        for i in range(0,len(predicted)):
            #print 'predicted is ',predicted[i],' goldset is ',goldset[i]
            if goldset[i]==predicted[i]:
                correct+=1
        
        return (float(correct)/len(predicted))*100
        
        
    def labelPropogation(self):
        #Algorithm 11.1 Label propagation (Zhu and Ghahramani, 2002)
        self.y_test=self.y_test.reshape(-1,1)
        self.y_train=self.y_train.reshape(-1,1)
        
        self.yUnlabeled=np.zeros(self.y_test.shape)
        self.y_labeled=copy(self.y_train)
        
        self.ypred=copy(np.vstack((self.y_labeled,self.yUnlabeled)))
        
        #now to do the label propogation 
        
        for i in range(0,50):   
            
            self.ypred=np.dot(np.linalg.inv(self.D),np.dot(self.pwdis,self.ypred))
            #now need to relabel all the labeled points
            for i in range(0,self.y_labeled.shape[0]):
                self.ypred[i,0]=self.y_labeled[i,0]
            
            
            
        #now label propogation is complete
        numTrain=self.y_train.shape[0]
        self.predicted1=self.ypred[numTrain:,0]
        #now we need to rethreshold them to 1 and -1
        
        for i in range(self.predicted1.shape[0]):
            if self.predicted1[i]>0:
                self.predicted1[i]=1
            else:
                self.predicted1[i]=-1
        
        self.predicted1=self.predicted1.reshape(-1,)
        self.y_test=self.y_test.reshape(-1) 
        print 'self.predicted1 is ',self.predicted1.shape
        print 'self.y_test is ',self.y_test.shape       
        print 'the accuracy is ',self.checkAccuracy(self.predicted1,self.y_test)
                    
            
        
    def compareWithSvm(self):
        C=[0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
        self.y_train=self.y_train.reshape(-1,)
        for c in C:
            self.Svm=svm.LinearSVC(C=c)
            self.Svm.fit(self.trainVectors,self.y_train)
            labels=self.Svm.predict(self.testVectors)
            print 'accuracy with c=',c,'  is  ',self.checkAccuracy(labels,self.y_test),'% ','\n\n'   
        
            
            
            
    
    #def constructSimilartyMatrixITML(self):
        
    



#for graph based reasoning , replace every 0 with -1

newLabels=[]
for label in labels:
    if label==1:
        newLabels.append(label)
    else:
        newLabels.append(-1)

#print 'newLabels are ',newLabels

newLabels=np.asarray(newLabels)
np.save('NewLabels',newLabels)

    

skf=StratifiedKFold(newLabels,n_folds=4,shuffle=True)

for train_index,test_index in skf:
    X_train,X_test=data[test_index],data[train_index]
    y_train,y_test=labels[test_index],labels[train_index]
    X_train=copy(X_train.reshape(-1,1))
    X_test=copy(X_test.reshape(-1,1))
    #y_train=copy(y_train.reshape(-1,1))
    #y_test=copy(y_test.reshape(-1,1))
    ob1=GraphBasedLearning(X_train,y_train,X_test,y_test)
    print 'X_train is ',y_train.shape
    print 'X_test is ',y_test.shape
    
    

