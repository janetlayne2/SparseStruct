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