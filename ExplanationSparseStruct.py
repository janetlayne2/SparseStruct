#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from scipy.sparse import csr_matrix,lil_matrix
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import vstack,hstack
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from scipy.special import logit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, mutual_info_classif


# In[2]:


data = pd.read_csv('dataset_latest.csv')


# In[3]:


nor_weight=data.groupby(['s_timestamp', 's_group'])['weight'].sum()


# In[4]:


def norm_weight(df, normdf):
    x=df.values
    norm =[]
    for a in range(x.shape[0]):
        denom = normdf[df['s_timestamp'][a]][df['s_group'][a]]
        norm.append(x[a,5]/denom)
    return pd.Series(norm)
        


# In[5]:


data['norm'] = norm_weight(data, nor_weight)


# In[6]:


import pandas as pd 
class loader:
    
    def __init__(self):
        self.countID=0
        self.relevantEdge=[]
        self.nedge=0
        self.G={}
        self.G2={}
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
        self.relevantEdge=self.relEdge()
        
        
        
    
    def storeEmb(self,file,data):
        file1 = open(file, 'w') 
        for a in range(data.shape[0]):
            s=''+str(int(self.revco[a]))
            for b in range(data.shape[1]):
                s+=' '+str(data[a,b])
            file1.write(s+"\n")
        file1.close()
            
    
    def fixG(self):
        for g in range(len(self.G2)):
            self.G[g]=np.array([x for x in self.G[g]])

    def addEdge(self,s, w):
        self.nedge+=1
        (l1,l2)=s
        if l1 not in self.G:
            self.G2[l1]=set()
            self.G[l1]={}
        if l2 not in self.G:
            self.G2[l2]=set()
            self.G[l2]={}
        self.G2[l1].add(l2)
        self.G2[l2].add(l1)
        self.G[l1][l2]=w
        self.G[l2][l1]=w
    
    def conv(self,e):
        (a,b)=e
        if a<b:
            return ''+str(a)+','+str(b)
        else:
            return ''+str(b)+','+str(a)
    
    def relEdge(self):
        visited = np.zeros(len(self.G2))
        relEdge= [None]*len(self.G2)
        queue = []
        i=0
        while (i<len(self.G2)):
            queue.append(i)
            this_graph = set()
            node_list= set()
            while (len(queue)>0):
                j= queue.pop(0)
                for x in (self.G2[j]):
                    if (visited[x] == 0):
                        if (x not in queue):
                            queue.append(x)
                            this_graph.add(self.conv((j, x)))
                            node_list.add(j)
                            node_list.add(x)
                visited[j]=1
            for g in node_list:
                relEdge[g]=this_graph
            i+=1
            
        return relEdge

        


# In[7]:


l = loader()
l.read(data)


# In[8]:


from scipy.sparse import lil_matrix,csc_matrix
from scipy.sparse import vstack,hstack
from sklearn.decomposition import TruncatedSVD
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
    explanationVal={}
    current=0
    for ii in range(iter):
        print(ii)
        print(emb.shape)
        ma=mappa()
        ll,_=ma.load(lil_matrix(hstack([emb,lil_matrix(l)])))
        emb=lil_matrix((nv, ma.count))
        
        for i in range(nv):
            for y in G[i]:
                j=ll[y]
                if (i,j+current) not in explanationVal:
                    explanationVal[(i,j+current)]=[]
                emb[i,j]+=G[i][y]
                explanationVal[(i,j+current)].append((y,ii))
                #nodes is y, stores what nodes are the neighbor, and at which level they are explored to obtain feature
                #want to know what the structure is responsible for the feature in j+current, to know that, need what are the neighbor nodes, and level they are explored, level is ii
        if emb.shape[1]!=1 and emb.shape[1]==size:
            break
        else:
            size=emb.shape[1]
            current+=size
            res.append(emb)
        #current=hstack(res)
    print(iter)
    print(emb.shape)
    return hstack(res),explanationVal
    #return emb
    
#call fast1, get embedding. Also returns what we need for explanation (dictionary) and the svd components
#have to return to the embedding responsible for the features that are in svd components
def fast1(G,l,n,iterq=100):  
    emb,explanation=fast(G,l,iter=iterq)
    svd = TruncatedSVD(n_components=min(n,emb.shape[1]-1),n_iter=10,random_state=1)
    return svd.fit_transform(emb),(explanation, svd.components_)


# In[9]:


l.l=np.array(l.l)


# In[10]:


l.l=OneHotEncoder().fit_transform(l.l)


# In[11]:


emb,(explanation,components)=fast1(l.G,l.l,17)


# In[25]:


# This class transfers the lime explanation score to actual edges. The lime score is calculated on features that are
# actually dimension reduced embedding items, not individual edges from the graph.
import networkx as nx

class explainer:
    def __init__(self,G,explanation,components):
        self.G=self.generate(G)
        self.G0=G
        self.explanation=explanation
        self.components=components
        self.edgescore={}
    
    def generate(self,g):
        G = nx.Graph()
        G.add_nodes_from([x for x in range(len(g))])
        l=[]
        for i in g:
            for j in g[i]:
                if i<=j:
                    l.append([i,j])
        G.add_edges_from(l)
        return G
        
    def update(self,e,v):
        (a,b)=self.conv(e)
        if (a,b) not in  self.edgescore:
            self.edgescore[(a,b)]=0
            self.edgescore[(b,a)]=0
        self.edgescore[(a,b)]=self.edgescore[(a,b)]+v
        self.edgescore[(b,a)]=self.edgescore[(b,a)]+v
        
    def conv(self,e):
        (a,b)=e
        if a<=b:
            return e
        else:
            return (b,a)
    
    def unfoldUpdate(self,node,aa,l,value): 
        #want to get the score for each edge to be proportion(based on weight), because this method follows all
        #neighbors, we need to accomodate their weights
        #ll=set([self.conv((node,aa))])
        ll=[self.conv((node,aa))]
        h = self.unfold(aa,l)
        #i = self.unfold(aa, l-1) (this is for number 6- remove 2 hops from 3 hop)
        #for (x,y) in i:
            #if (x,y) in h:
                #h.remove((x,y))
        #ll=ll+h            
        #for (x,y) in h:
         #   (i,j)=self.conv((x,y))
          #  if (i,j) not in ll:
           #     ll.add((i,j))
        ll=ll+(self.unfold(aa,l))            
        for (a,b) in ll:
            self.update((a,b),value)
  
    def unfold(self, node, l):
        if l==0:
            return []
        else:
            edges=[]
            for b in self.G0[node]:
                edges.append((node, b))
                h=self.unfold(b, l-1)
                edges.extend(h)
            return edges    
        

#create a dictionary where each edge has a score, shows which edge is most important for classification, we can
#select the subgraph that explains the classification, edges whose scores above a threshold, then we look at the
#distribution of the scores
    def explain(self,node, score):
        self.edgescore={}
        scores=np.matmul(score,self.components) #returns from svd back to original features
        #score is a vector of 130
        #scores is the size of sparsematrix.shape[1]
        for i in range(scores.shape[0]):
            if (node,i) in self.explanation:
                ll=self.explanation[(node,i)]
                #this is suggestion (fix) for how to distribute the individual weights that contribute,,
                #ll contains all the nodes participating in a structure, so we need to accomodate different weights
                # percent of the total weight in structure
                #value1=sum(wnode,i for i in ll)/len(ll) #multiply score[i] for edge(node, a) by the weight for that edge
                value=scores[i]/len(ll)
                for (a,l) in ll:
                    #value=scores[i]*wnode,a/value1
                    #a is the neighbor of node
                    if l>0:
                        self.unfoldUpdate(node,a,l,value)
                    else:
                        self.update((node,a),value)
        return self.edgescore
    
    


# In[26]:


expl=explainer(l.G,explanation,components)


# In[27]:


# This concatenates the embedding by group and timestamp so that we are classifying a "graph" rather than a node
#key is (timestamp, group), object is nested dictionary where key is source, object is nodeID
#Stores information about where a node there is data in the sparse matrix (where the nodes are for each graph)
# Remember a graph is represented as all the possible nodes it could contain, with weight values where there are
# Actually corresponding nodes
#Sanity check complete on both the creation of the graph and the node order list
dictionary = {}
labels=set()
node_order = []
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
    nodes_graph=[]
    for label in labels:
        if(label in d1):
            ff.extend(emb[d1[label], :].tolist())
            nodes_graph.append(d1[label])
        else:
            ff.extend(np.zeros(emb.shape[1]).tolist())
            nodes_graph.append(-1)
    X.append(ff)
    node_order.append(nodes_graph)
X=np.array(X)


# In[28]:


# This removes the first two columns from X (which contain the terror group and timestamp) so that the training data 
# is only the embedding values
Y=[]
X2=[]
for i in range(X.shape[0]):
    Y.append(X[i][0])
    rows=[]
    for j in range(2, X.shape[1]):
          rows.append(X[i][j])
    X2.append(rows)  
X2=np.array(X2)   


# In[29]:


#This was needed to for some reason convert the array to floats. Doing it as a batch didn't work 
X3 = []
for i in range(X2.shape[0]):
    to_add = []
    for j in range(X2.shape[1]):
        to_add.append(float(X2[i][j]))
    X3.append(to_add)


# In[30]:


# Split train and test data
train2 = X3[: 894]
train2 = train2.copy()
train2=np.array(train2)
train_labels = Y[: 894]
train_labels = train_labels.copy()
test2 = X3[894 :]
test2 = test2.copy()
test2=np.array(test2)
test_labels = Y[894 :]
test_labels = test_labels.copy()


# In[31]:


# Select best data before classifier
KBest= SelectKBest(mutual_info_classif, k=130)
feature_scores= KBest.fit(train2, train_labels)
train3= KBest.transform(train2)
test3= KBest.transform(test2)


# In[32]:


#This shows which data were chosen from the original by the KBest
indices=KBest.get_support(indices=True)


# In[33]:


clf = ExtraTreesClassifier(n_estimators=400)
clf.fit(train3, train_labels)
pred =clf.predict(test3)
print(classification_report(test_labels, pred))


# In[34]:


#List of whether a value was correctly classified
eva=test_labels==pred


# In[36]:


import lime
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(train3, discretize_continuous=True)
#lime is an explainer for classification, SHAP is another
#taking classifier as input, return score for each input


# In[37]:


# Runs lime explainer for each instance in the test data
# this creates a list of scores as a table for each graph representation,
#but so that theyre all in order of the features
result=[]
scores2 = []
for i in range(test3.shape[0]):
    exp=explainer.explain_instance(test3[i],clf.predict_proba,num_features=130, top_labels=1)
    score=np.zeros(130)
    aa=exp.as_map()
    for f in aa:
        for h in aa[f]:
            score[h[0]]=h[1]
        scores2.append(score)


# In[90]:


import pickle
firstscores = open('firstscores', 'wb')
pickle.dump(scores2, firstscores)
firstscores.close()


# In[124]:


import pickle
scores = pickle.load( open( "edgescores", "rb" ) )


# In[38]:


scores2


# In[39]:


#This fills in a score for all the possible graphs, putting in zeros for the features not chosen by the KBest
#This essentially re-expands the KBest, preserving the order of the features for all
#Sanity checked
all_scores=[]
for i in range(len(scores2)):
    graph_score= np.zeros(629)
    for j in range(len(indices)):
            graph_score[indices[j]]= scores2[i][j]
    all_scores.append(graph_score) 


# In[41]:


#This splits the data back into individual nodes, instead of groupings by timestamp/group(graphs)
score_split=[]
for i in range(len(all_scores)):
    score_split.append(np.split(all_scores[i], 37))


# In[44]:


# This just gives a list of which nodes were present in the original large sparse matrix for a test instance
test_nodes=[]
for i in range(894, len(node_order)):
    test_nodes.append(node_order[i])


# In[46]:


# This runs the explanation only for nodes that were correctly classified to place scores on edges
explanationx=[]

for i in range(len(score_split)):
    for j in range(len(score_split[0])):
        node = test_nodes[i][j]
        scorex =  score_split[i][j]
        node_exp= expl.explain(node, scorex)
        explanationx.append(node_exp)    


# In[47]:


# add up the score that each edge got for each node for that edge
resultx = {}
for d in explanationx:
    for k in d.keys():
        resultx[k] = resultx.get(k, 0) + d[k]


# In[48]:


keys=list(resultx.keys())


# In[49]:


# Makes a dictionary that groups the list of edgescores by terror group and timestamp.
final={}
for (src, trg) in resultx:
    src_l = l.y[src]
    src_t = l.t[src]
    src_n = l.revco[src]
    trg_n = l.revco[trg]
    if (src_l, src_t) not in final:
        final[(src_l, src_t)]={}
    if (trg_n, src_n) not in final[(src_l, src_t)]:
        final[(l.y[src], l.t[src])][(l.revco[src], l.revco[trg])]= resultx[(src, trg)]


# In[184]:


# Export final list
import pickle
edgescores = open('edgescores', 'wb')
pickle.dump(final, edgescores)
edgescores.close()

