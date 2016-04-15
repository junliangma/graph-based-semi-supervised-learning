import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold   
from nltk.stem.porter import PorterStemmer                                                                 
from copy import copy
from sklearn.manifold import TSNE
from nltk import word_tokenize
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import write_dot
from sklearn import svm
from metric_learn import LMNN
import sys
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
#from modshogun import RealFeatures,BinaryLabels,LMNN,MulticlassLabels
from metric_learn import ITML


pp=PdfPages('PlotPdf.pdf')

randomState=13204
data=np.load('FullData.npy')
labels=np.load('NewLabels.npy')

print 'data.shape is ',data.shape
print 'labels.shape is ',labels.shape
#data=data.tolist()
#labels=labels.tolist()
from sklearn.manifold import TSNE
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
        #self.constructSimilartyMatrixCosine()
        #print 'now doing label propogation\n'
        #self.labelPropogation()
        #self.compareWithSvm()
        #self.constructSimilartyMatrixLMNN()
        #self.constructSimilartyMatrixITML()
        self.convertToDenseMAtrix()
        print 'before PCA SVM Accuracy is ',self.compareWithSvm(self.trainVectors,self.testVectors)
        print 'now computing pca ',self.computePca()
        print 'after PCA svm accuracy is ',self.compareWithSvm(self.trainVectorsPCA,self.testVectorsPCA)
    
    
    def convertToDenseMAtrix(self):
        self.trainVectors=self.trainVectors.todense()
        temp=copy(np.zeros(self.trainVectors.shape))
        for i in range(0,self.trainVectors.shape[0]):
            for j in range(0,self.trainVectors.shape[1]):
                temp[i,j]=self.trainVectors[i,j]
        
        self.trainVectors=copy(temp)
        self.testVectors=self.testVectors.todense()
        temp=copy(np.zeros(self.testVectors.shape))
        for i in range(0,self.testVectors.shape[0]):
            for j in range(0,self.testVectors.shape[1]):
                temp[i,j]=self.testVectors[i,j]    
        self.testVectors=copy(temp)
        
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
        #projectedDigits = TSNE(random_state=randomState).fit_transform(self.allVectors.todense())
        #plt.scatter(projectedDigits[:,0],projectedDigits[:,1],c=self.labels)
        #plt.title('All Datas Set projected into 2D by TSNE')
        #plt.savefig(pp,format='pdf')
        #plt.show()
        
    
    def constructSimilartyMatrixCosine(self,k=20):
        #This is a simpole k nearest neighbour approach based on the cosine distance
        #for this takefrom modshogun import RealFeatures, MulticlassLabels
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
    
    def computePca(self):
        
        pca=PCA(n_components=self.trainVectors.shape[0])
        pca.fit(self.trainVectors)
        self.trainVectorsPCA=copy(pca.transform(self.trainVectors))
        self.testVectorsPCA=copy(pca.transform(self.testVectors))
        #print 'the explained variance is ',np.cumsum(pca.explained_variance_ratio_)
        
         
    
    def constructSimilartyMatrixLMNN(self):
        print 'Now doing LMNN'
        #self.y_train=self.y_train.reshape()
        print 'self.y_train is ',self.y_train.shape
        
        x1=copy(self.trainVectors.todense())
        mat1=np.zeros(x1.shape)
        for i in range(0,mat1.shape[0]):
            for j in range(0,mat1.shape[1]):
                mat1[i,j]=x1[i,j]
        print 'x.shape is ',x1
        #self.shogun_X_train=RealFeatures(mat1.T)
        #print 'shogun xtrain is ',self.shogun_X_train
        #self.shogun_y_train=MulticlassLabels(self.y_train[0:10].astype(np.float64))
        k=10
        lmnn=LMNN(k=20, learn_rate=1e-3,use_pca=False)
        #init_transform = np.eye(self.trainVectors.shape[1])
        #lmnn.set_maxiter(3)
        #lmnn.train(init_transform)
        lmnn.fit(mat1, self.y_train, verbose=False)
        self.L = lmnn.transformer()
        
        #self.M = np.matrix(np.dot(L.T,L))
        np.save('LMNN transformer',self.L)

        
        
        
        print 'L.shape is ',self.L.shape,'\n\n'
    
    
    def constructSimilartyMatrixITML(self):
        print 'Now doing itml'
        #self.y_train=self.y_train.reshape(-1,)
        self.y_train2=copy(self.y_train)
        rows=np.where(self.y_train==-1)
        self.y_train2[rows]=0
        print 'self.y_train is ',self.y_train.shape
        x1=copy(self.trainVectors.todense()[0:10]) 
        y1=copy(self.y_train2[0:10])
        print 'x1 is ',x1,' y1 is ',y1
        itml = ITML()
        num_constraints = 1000
        mat1=np.zeros(x1.shape)
        for i in range(0,mat1.shape[0]):
            for j in range(0,mat1.shape[1]):
                mat1[i,j]=x1[i,j]
        C = ITML.prepare_constraints(y1, mat1.shape[0], num_constraints)
        itml.fit(mat1, C, verbose=True)
        xl=itml.transform(mat1)
        
        print 'xl is ',xl.shape
        
        
        
        
            
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
                    
            
         
    def compareWithSvm(self,datasetTrain,datasetTest):
        C=[0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
        print '\n'
        print 'dataset shape is ',datasetTrain.shape
        self.y_train=self.y_train.reshape(-1,)
        for c in C:
            self.Svm=svm.LinearSVC(C=c)
            self.Svm.fit(datasetTrain,self.y_train)
            labels=self.Svm.predict(datasetTest)
            print 'accuracy with c=',c,'  is  ',self.checkAccuracy(labels,self.y_test),'% ','\n'   
        
            
            
            
    
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
    
    
pp.close()

