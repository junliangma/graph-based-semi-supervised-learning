from sklearn.datasets import fetch_20newsgroups

cats = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'),categories=cats)

print 'newsgroups_train.target is ',newsgroups_train.target

#print 'newsgroups_train is ',newsgroups_train.data[0:100]

d=newsgroups_train.data[0:50]
i=1
for line in d:
    print 'Line number i=',i,' = ',line,'\n\n\n'
    i+=1


