#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.sparse import csr_matrix,lil_matrix
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import vstack,hstack
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder


# In[16]:


data = pd.read_csv('Data/dataset_latest.csv')


# In[17]:


nor_weight=data.groupby(['s_timestamp', 's_group'])['weight'].sum()


# In[18]:


def norm_weight(df, normdf):
    x=df.values
    norm =[]
    for a in range(x.shape[0]):
        denom = normdf[df['s_timestamp'][a]][df['s_group'][a]]
        norm.append(x[a,5]/denom)
    return pd.Series(norm)
        


# In[19]:


data['norm'] = norm_weight(data, nor_weight)


# In[20]:


import pandas as pd 
class loader:
    
    def __init__(self):
        self.countID=0
        self.G={}
        self.co={}
        self.revco={}
        self.y=[]
        self.l=[]
        self.t=[]
        self.m=[]
        self.n=[]
        self.i=[]
    
    def nodeID(self,x, label_g, label_a, t):
        if x not in self.co:
            self.co[x]=self.countID
            self.i.append(self.countID)
            self.countID=self.countID+1
            self.revco[self.co[x]]=x
            self.y.append(label_g)
            self.l.append([label_a])
            self.t.append(t)
        return self.co[x] 
    
    def read(self,df):
        x=df.values
        for a in range(x.shape[0]):
            i=self.nodeID(x[a,1], x[a,7], x[a,2], x[a,8])
            j=self.nodeID(x[a,3], x[a,10], x[a,4], x[a,11])
            w=(x[a,12])
            self.addEdge((i,j), w)
        #self.fixG()
        
        
    
    def storeEmb(self,file,data):
        file1 = open(file, 'w') 
        for a in range(data.shape[0]):
            s=''+str(int(self.revco[a]))
            for b in range(data.shape[1]):
                s+=' '+str(data[a,b])
            file1.write(s+"\n")
        file1.close()
            
    
    def fixG(self):
        for g in range(len(self.G)):
            self.G[g]=np.array([x for x in self.G[g]])

    def addEdge(self,s, w):
        (l1,l2)=s
        if l1 not in self.G:
            self.G[l1]={}
        if l2 not in self.G:
            self.G[l2]={}
        self.G[l1][l2]=w
        self.G[l2][l1]=w
        
    
        


# In[21]:


from scipy.sparse import lil_matrix,csc_matrix
class node:
    def __init__(self):
        self.id=-1
        self.dictionary={}

class mappa:
    def __init__(self):
        self.node=node()
        self.count=0
    
    def get(self,X):
        (_,b)=X.nonzero()
        c=self.node
        for x in b:
            if x not in c.dictionary:
                c.dictionary[x]=node()
            c=c.dictionary[x]
            f=int(X[0,x])
            if f not in c.dictionary:
                c.dictionary[f]=node()
            c=c.dictionary[f]
        if c.id==-1:
            c.id=self.count
            self.count=self.count+1
            return (c.id,1)
        return (c.id,0)
    
    def load(self,X):
        l=[]
        k=[]
        for x in range(X.shape[0]):
            (a,b)=self.get(X[x,:])
            l.append(a)
            if b==1:
                k.append(x)
        return l,k

def fast(G,l,iter=100):
    res=[]
    nv=len(G) 
    size=0
    emb=csc_matrix((nv, 1), dtype=np.int8)
    for i in range(iter):
        print(i)
        print(emb.shape)
        ma=mappa()
        ll,_=ma.load(lil_matrix(hstack([emb,lil_matrix(l)])))
        emb=lil_matrix((nv, ma.count))
        print('emb: ', emb[0,:])
        for i in range(nv):
            for y in G[i]:
                j=ll[y]
                emb[i,j]+=G[i][y]
        if emb.shape[1]!=1 and emb.shape[1]==size:
            break
        else:
            size=emb.shape[1]
            res.append(emb)
        #current=hstack(res)
    print(iter)
    print(emb.shape)
    return hstack(res)
    #return emb


# In[22]:


l=loader()
l.read(data)


# In[26]:


l.l=np.array(l.l)


# In[27]:


l.l=OneHotEncoder().fit_transform(l.l)


# In[28]:


emb=fast(l.G,l.l,iter=100)


# In[29]:


svd = TruncatedSVD(n_components=min(17,emb.shape[1]-1),n_iter=10,random_state=1)
emb1=svd.fit_transform(emb)


# In[30]:


#key is (timestamp, group), object is nested dictionary where key is source, object is nodeID
# this method aggregates the nodes of a single group/timestamp together for graph classification.
dictionary = {}
labels=set()
data1 = data.values
for i in range(data1.shape[0]):
    label = data1[i,2]
    group = data1[i,7]
    timestamp = data1[i,8]
    nodeID = data1[i, 1]
    if((group, timestamp) not in dictionary):
        dictionary[(group, timestamp)]= {}
    dictionary[(group, timestamp)][label]= l.co[nodeID]
    labels.add(label)
X = []
for (group, timestamp) in dictionary:
    d1= dictionary[(group, timestamp)]
    ff=[group, timestamp]
    for label in labels:
        if(label in d1):
            ff.extend(emb1[d1[label], :].tolist())
        else:
            ff.extend(np.zeros(emb1.shape[1]).tolist())
    X.append(ff)
X=np.array(X)

        


# In[43]:


Y=[]
X2=[]
for i in range(X.shape[0]):
    Y.append(X[i][0])
    rows=[]
    for j in range(2, X.shape[1]):
          rows.append(X[i][j])
    X2.append(rows)  
X=np.array(X2)   

