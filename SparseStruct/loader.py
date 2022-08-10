import numpy as np
import pandas as pd

class loader:
    
    def __init__(self):
        self.countID=0
        self.G={}
        self.co={}
        self.revco={}
    
    def nodeID(self,x):
        if x not in self.co:
            self.co[x]=self.countID
            self.countID=self.countID+1
            self.revco[self.co[x]]=x
        return self.co[x]
    
    def read(self,data):
        x = data.values
        for a in range(x.shape[0]):
            i=self.nodeID(x[a,0])
            j=self.nodeID(x[a,1])
            self.addEdge((i,j))
        self.fixG()
        
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

    def addEdge(self,s):
        (l1,l2)=s
        if l1 not in self.G:
            self.G[l1]=set()
        if l2 not in self.G:
            self.G[l2]=set()
        self.G[l1].add(l2)
        self.G[l2].add(l1)