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

from metric_learn import ITML,LMNN
import sys
#sys.path.append('/home/drishi/shogun-install/lib/python2.7/dist-packages/')

categories=['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),categories=categories)

class StemmerTokenizer(object):

    def __init__(self): 
        self.stemmer = PorterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc)]

vectorizer = TfidfVectorizer(decode_error='replace',analyzer='word',stop_words='english',lowercase=True,tokenizer=StemmerTokenizer())

data=newsgroups_train.data[0:100]
vectorizer.fit(data)

print 'the features are ',len(vectorizer.get_feature_names())
vectors = vectorizer.transform(data)

print 'vectorizer is  ' ,vectors[0].todense()
itml=ITML()
arr2=copy(vectors.todense())
arr=np.zeros((vectors.shape[0],vectors.shape[1]))

for i in range(0,vectors.shape[0]):
    for j in range(0,vectors.shape[1]):
        arr[i,j]=arr2[i,j]

print 'arr .shape is  ',arr.shape
target=newsgroups_train.target[0:100]
lab=[]
for i in target:
    lab.append(i)

lab=np.asarray(lab)
print 'lab is ',(lab)
print 'target is ',type(arr)
#C=itml.prepare_constraints(target,vectors.shape[0],200)

#itml.fit(arr,C,verbose=False)

lmnn = LMNN(k=20, learn_rate=1e-3,use_pca=True)
lmnn.fit(arr,target,verbose=False)

l=lmnn.transformer()
np.save('LMNN transformer',l)
