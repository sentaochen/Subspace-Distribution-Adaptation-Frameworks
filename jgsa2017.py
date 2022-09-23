from __future__ import division
import numpy as np
import scipy.linalg as la
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression


knn = neighbors.KNeighborsClassifier(n_neighbors=1)
logreg = LogisticRegression()

class JGSA:
    
    """
    Joint Geometrical and Statistical Alignment for Visual Domain Adaptation. cvpr2017
    Adapted from Jing Zhang MATLAB code, only the linear mapping part,i.e. JGSA(primal)
    
    Parameters
    ----------
    k: subspace dimension.
    T: iteration times, like Joint Distribution Adaptation.
    lamda: the parameter for subspace divergence ||A-B||
    mu: the parameter for target variance
    beta: the parameter for P and Q (source discriminaiton)  
    estimator: estimator for labeling the target data in the iteration. 
    
    """

    def __init__(self,k=30, T=10, mu=1, lamda=1, beta=0.01, estimator=knn):
        args_values = locals()
        args_values.pop("self")
        for arg,value in args_values.items():
            setattr(self,arg,value)
    
    def _constructMMD(self,ns,nt,Ys,Yt0,C):        
        e = np.vstack((1 / ns * np.ones((ns,1)),-1 / nt * np.ones((nt,1))))
        es = 1 / ns * np.ones((ns,1))
        et = -1 / nt * np.ones((nt,1))

        M = e.dot(e.T) * C
        Ms = es.dot(es.T) * C
        Mt = et.dot(et.T) * C
        Mst = es.dot(et.T) * C
        Mts = et.dot(es.T) * C

        for c in np.unique(Ys):
            es = np.zeros((ns,1))
            et = np.zeros((nt,1))
            es[Ys==c] = 1. / np.sum(Ys==c) if np.sum(Ys==c)!=0 else 0.
            et[Yt0==c] = -1. / np.sum(Yt0==c) if np.sum(Yt0==c)!=0 else 0.
            Ms = Ms + es.dot(es.T)
            Mt = Mt + et.dot(et.T)
            Mst = Mst + es.dot(et.T)
            Mts = Mts + et.dot(es.T)
                      
        Ms = Ms / la.norm(M,'fro')
        Mt = Mt / la.norm(M,'fro')
        Mst = Mst / la.norm(M,'fro')
        Mts = Mts / la.norm(M,'fro')
        
        return Ms, Mt, Mst, Mts
    
    def fit_transform(self, Xs, Xt, Ys, Yt0, Yt):
        
        """
        Fit and transform the source and target domain data using JGSA.
        Parameters
        ----------
        Xs : array, shape = [ns_samples, n_features]            
        Xt : array, shape = [nt_samples, n_features]
                    
        Returns
        -------
        transformed source and target domain samples Xs ,Xt, and transformation matries A, Att.
        
        """    
        Xs, Xt = Xs.T, Xt.T        
        
        m = Xs.shape[0]
        ns,nt = Xs.shape[1],Xt.shape[1]
        
        uniclass = np.unique(Ys)
        C = len(uniclass)        
           
        # compute LDA
        dim = Xs.shape[0]
        meanTotal = Xs.mean(1)

        Sw = np.zeros((dim, dim))
        Sb = np.zeros((dim, dim))
        for i in range(C):
            Xi = Xs[:,Ys==uniclass[i]]
            meanClass = Xi.mean(1)
            Hi = np.eye(Xi.shape[1])-1.0 / Xi.shape[1] * np.ones((Xi.shape[1],Xi.shape[1]))
            Sw = Sw + Xi.dot(Hi).dot(Xi.T) # calculate within-class scatter
            ColVec = (meanClass-meanTotal)[:,None]
            Sb = Sb + Xi.shape[1] * ColVec.dot(ColVec.T) # calculate between-class scatter

        P = np.zeros((2*m,2*m))
        P[:m,:m] = Sb
        Q = Sw

        for t in range(self.T):
        
            # Construct MMD matrix
            Ms, Mt, Mst, Mts = self._constructMMD(ns,nt,Ys,Yt0,C)

            Ts = Xs.dot(Ms).dot(Xs.T)
            Tt = Xt.dot(Mt).dot(Xt.T)
            Tst = Xs.dot(Mst).dot(Xt.T)
            Tts = Xt.dot(Mts).dot(Xs.T)
        
            # Construct centering matrix
            Ht = np.eye(nt) - 1.0 / nt * np.ones((nt,nt))
                
            X = np.vstack((np.hstack((np.zeros((m,ns)), np.zeros((m,nt)))),np.hstack((np.zeros((m,ns)), Xt))))
            H = np.vstack((np.hstack((np.zeros((ns,ns)), np.zeros((ns,nt)))),np.hstack((np.zeros((nt,ns)), Ht))))

            Smax = self.mu * X.dot(H).dot(X.T) + self.beta * P
            a11 = Ts + self.lamda * np.eye(m) + self.beta * Q
            a12 = Tst - self.lamda * np.eye(m)
            a21 = Tts - self.lamda * np.eye(m)
            a22 = Tt + (self.lamda + self.mu) * np.eye(m)
            Smin = np.vstack((np.hstack((a11, a12)),np.hstack((a21, a22))))        
        
            eigvals,W = la.eigh(Smax,Smin + 1e-9 * np.eye(2 * m))
            #idx = np.argsort(np.abs(eigvals))[::-1]
            #W = W[:,idx[:self.k]]
            W = W[:,::-1][:,:self.k]
            A,Att = W[:m,:],W[m:,:]

            Zs,Zt = A.T.dot(Xs),Att.T.dot(Xt)
        
            if self.T > 1:
                Yt0 = self.estimator.fit(Zs.T,Ys).predict(Zt.T)
                acc = np.mean(Yt0==Yt)
                #print "acc of iter %d: %0.4f\n"%(t,acc)
        
        Xs = Zs.T
        Xt = Zt.T
        
        return  Xs, Xt, A, Att
