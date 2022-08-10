import argparse
import pandas as pd
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import vstack,hstack
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import loader
import loader_w
import loader_nlw
import loader_nl
import random

def parse_args():
    parser = argparse.ArgumentParser(description='SparseStruct')
    parser.add_argument("--input", type=str, default="Data/synth_0.0.txt",
            help="Input graph path")
    parser.add_argument("--output", type=str, default='emb/synth0emb.txt',
            help="Output embedding path")
    parser.add_argument("--node_feat", type=str, default=None,
            help="Number of iterations")
    parser.add_argument("--weighted", default=False, action = "store_true",
            help="Indicate True or False if the input graph has weighted edges")
    parser.add_argument("--rep", type=int, default=20,
            help="Dimensionality of final representation. Default = 20.")
    parser.add_argument("--stop", default=False, action = "store_true",
            help="Automatic algorithm stop at convergence. Choose True or False.")
    parser.add_argument("--depth", type=int, default=100,
            help="Number of iterations to run if not using convergence. Default = 10.")

    return parser.parse_args()

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

def fast_st(G,iter=100):
    res=[]
    nv=len(G) 
    size=0
    emb=csc_matrix((nv, 1), dtype=np.int8)
    for i in range(iter):
        print(i)
        print(emb.shape)
        ma=mappa()
        ll,_=ma.load(emb)
        emb=lil_matrix((nv, ma.count))
        for i in range(nv):
            for y in G[i]:
                j=ll[y]
                emb[i,j]+=1
        if emb.shape[1]!=1 and emb.shape[1]==size:
            break
        else:
            size=emb.shape[1]
            res.append(emb)
        #current=hstack(res)
    print("embedding shape: ", emb.shape)
    return hstack(res)
    #return emb

def fast_nost(G,iter=10):
    res=[]
    nv=len(G) 
    size=0
    emb=csc_matrix((nv, 1), dtype=np.int8)
    for i in range(iter):
        print(i)
        print(emb.shape)
        ma=mappa()
        ll,_=ma.load(emb)
        emb=lil_matrix((nv, ma.count))
        for i in range(nv):
            for y in G[i]:
                j=ll[y]
                emb[i,j]+=1
        res.append(emb)
    print("embedding shape: ", emb.shape)
    return hstack(res)
    #return emb

def fast_w_st(G,iter=100):
    res=[]
    nv=len(G) 
    size=0
    emb=csc_matrix((nv, 1), dtype=np.int8)
    for i in range(iter):
        print(i)
        print(emb.shape)
        ma=mappa()
        ll,_=ma.load(emb)
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
    print("embedding shape: ", emb.shape)
    return hstack(res)
    #return emb

def fast_w_nost(G,iter=10):
    res=[]
    nv=len(G) 
    size=0
    emb=csc_matrix((nv, 1), dtype=np.int8)
    for i in range(iter):
        print(i)
        print(emb.shape)
        ma=mappa()
        ll,_=ma.load(emb)
        emb=lil_matrix((nv, ma.count))
        print('emb: ', emb[0,:])
        for i in range(nv):
            for y in G[i]:
                j=ll[y]
                emb[i,j]+=G[i][y]
        res.append(emb)
        #current=hstack(res)
    print("embedding shape: ", emb.shape)
    return hstack(res)
    #return emb

def fast_nl_st(G,l,iter=100):
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
                emb[i,j]+=1
        if emb.shape[1]!=1 and emb.shape[1]==size:
            break
        else:
            size=emb.shape[1]
            res.append(emb)
        #current=hstack(res)
    print("embedding shape: ", emb.shape)
    return hstack(res)
    #return emb

def fast_nl_nost(G,l,iter=10):
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
                emb[i,j]+=1
        res.append(emb)
        #current=hstack(res)
    print("embedding shape: ", emb.shape)
    return hstack(res)
    #return emb

def fast_nlw_st(G,l,iter=100):
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
    print("embedding shape: ", emb.shape)
    return hstack(res)
    #return emb

def fast_nlw_nost(G,l,iter=10):
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
        res.append(emb)
        #current=hstack(res)
    print("embedding shape: ", emb.shape)
    return hstack(res)
    #return emb


 # loader is no edge weights, node labels, or edge labels 
 # loader_w is edge weights, no node labels or edge labels
 # loader_nlw is node labels and edge weights, no node labels 
 # loader_nl is node labels no weights
def main(args):
   depth = args.depth
   rsize = args.rep
   data = pd.read_csv(args.input)
   if args.node_feat is None:
        if args.weighted:
                l = loader_w.loader_w()
                l.read(data)
                if args.stop:
                        emb = fast_w_st(l.G)
                else:
                        emb = fast_w_nost(l.G, iter = depth)
        else:
                l = loader.loader()
                l.read(data)
                if args.stop:
                        emb = fast_st(l.G)
                else:
                        emb = fast_nost(l.G, iter = depth)
   else:
        labels = pd.read_csv(args.node_feat)
        if args.weighted:
                l = loader_nlw.loader_nlw()
                l.read(data, labels)
                if args.stop:
                        emb = fast_nlw_st(l.G, l.nl)
                else:
                        emb = fast_nlw_nost(l.G, l.nl, iter = depth)
        else:
                l = loader_nl.loader_nl()
                l.read(data, labels)
                if args.stop:
                        emb = fast_nl_st(l.G, l.nl)
                else:
                        emb = fast_nost(l.G, l.nl, iter = depth)
   

   svd = TruncatedSVD(n_components=min(rsize,emb.shape[1]-1),n_iter=20,random_state=1)
   emb1=svd.fit_transform(emb)
   l.storeEmb(args.output, emb1)


if __name__ == "__main__":
    args = parse_args()
    main(args)


