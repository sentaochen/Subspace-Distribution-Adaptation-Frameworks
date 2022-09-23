from __future__ import division
import numpy as np
import scipy.linalg as la
import sklearn.metrics.pairwise as mp

class TCA:
    """Domain adaptation via transfer component analysis. IEEE TNN 2011.
    
    Parameters
    ----------
    kernel: str, default: 'linear'
        The kernel function used to construct the kernel matrix K, can be one of 'linear',
        'rbf' and 'poly'.
    sigma: float, default: 1.0 
        Parameter for the RBF kernel. Useful when kernel is set to be 'rbf'.
    c: float, default: 1.0
        Coefficient in the polynomial kernel. Useful when kernel is set to be 'poly'.
    degree: int, default: 3
        Degree for the polynomial kernel. Useful when kernel is set to be 'poly'.
    dimension: int, default: 20
        Dimension of the subspace.
    mu: float, default: 1.0
        Tradeoff parameter for the regularization term. 
    """

    def __init__(self,kernel='linear',sigma=1.0,c=1.0,degree=3,dimension=20,mu=1.0):
        args_values = locals()
        args_values.pop("self")
        for arg,value in args_values.items():
            setattr(self,arg,value)
    
    def fit_transform(self, Xs, Xt, Xt2=None):
        """Fit and transform the source and target domain data using transfer component analysis.
        Parameters
        ----------
        Xs : array, shape = [ns_samples, n_features]
            ns_samples is the number of source domain samples
            and n_features is the number of features.
        Xt : array, shape = [nt_samples, n_features]
            nt_samples is the number of target domain samples
            and n_features is the number of features.
        Xt2 : array, shape = [nt2_samples, n_features]
            nt2_samples is the number of additional target domain samples
            and n_features is the number of features.
        
        Returns
        -------
        Transformed source and target domain samples Xs_tranformed and Xt_tranformed.
        The transformed additional target domain samples Xt2_tranformed will also returned 
        if Xt2 is not None.
        """        
        X = np.vstack((Xs,Xt))
        n,ns,nt = X.shape[0],Xs.shape[0],Xt.shape[0]
        
        # Construct kernel matrix
        if 'linear' == self.kernel:
            K = mp.linear_kernel(X,X)
            if Xt2 is not None:
                K2 = mp.linear_kernel(Xt2,X)
        elif 'rbf' == self.kernel:
            K = mp.rbf_kernel(X,X,0.5 * self.sigma**-2)
            if Xt2 is not None:
                K2 = mp.rbf_kernel(Xt2,X,0.5 * self.sigma**-2)
        elif 'poly' == self.kernel:
            K = mp.polynomial_kernel(X,X,self.degree,1,self.c)
            if Xt2 is not None:
                K2 = mp.polynomial_kernel(Xt2,X,self.degree,1,self.c)
                
        # Construct centering matrix
        H = np.eye(n) - 1 / n * np.ones((n,n))

        # Construct MMD matrix
        e = np.hstack((1 / ns * np.ones(ns),-1 / nt * np.ones(nt)))[:,None]
        L = e.dot(e.T)       
        
        #eigenvalue decomposition
        M =  la.inv(K.dot(L).dot(K) + self.mu * np.eye(n)).dot(K).dot(H).dot(K)
        eig_vals,eig_vecs = la.eigh(M)
        #select the eigenvectors corresponding to the m=self.dimension largest eigenvalues.
        W = eig_vecs[:,::-1][:,:self.dimension] 
                                
        Z = K.dot(W)
        Xs_tranformed,Xt_tranformed = Z[:ns,:],Z[ns:,:]
        
        if Xt2 is not None:
            Xt2_transformed = K2.dot(W)
            return Xs_tranformed,Xt_tranformed,Xt2_transformed
        else:
            return Xs_tranformed,Xt_tranformed

class SSTCA:
    """Domain adaptation via transfer component ayalysis. IEEE TNN 2011.
    
    Parameters
    ----------
    kernel: str, default: 'linear'
        The kernel function used to construct the kernel matrix K, can be one of 'linear',
        'rbf' and 'poly'.The kernel matrix for the label is fixed at a linear kernel as 
        recommanded in the original paper.
    sigma: float, default: 1.0 
        Parameter for the RBF kernel. Useful when kernel is set to be 'rbf'.This parameter servers
        two purposes: one is for the data kernel matrix if the kernel is chosen as 'rbf',
        the other is for the graph Laplacian matrix.
    c: float, default: 1.0
        Coefficient in the polynomial kernel. Useful when kernel is set to be 'poly'.
    degree: int, default: 3
        Degree for the polynomial kernel. Useful when kernel is set to be 'poly'.
    dimension: int, default: 20
        Dimension of the subspace.
    gamma: float,default: 0.5
        Parameter in the label kernel matrix.
    lamda: float, default: 1.0
        Tradeoff parameter for the graph regularizer.
    mu: float, default: 1.0
        Tradeoff parameter for the regularization term. 
        
    """

    def __init__(self,kernel='linear',sigma=1.0,c=1.0,degree=3,dimension=20,gamma=0.5,lamda=1.0,mu=1.0):
        args_values = locals()
        args_values.pop("self")
        for arg,value in args_values.items():
            setattr(self,arg,value)
    
    def fit_transform(self, Xs, ys, Xt, Xt2=None):
        """Fit and transform the source and target domain data using transfer component analysis.
        Parameters
        ----------
        Xs : array, shape = [ns_samples, n_features]
            ns_samples is the number of source domain samples
            and n_features is the number of features.
        ys : array, shape = [ns_samples]
            labels for the source domain samples. ns_samples is the number of source domain samples.
        Xt : array, shape = [nt_samples, n_features]
            nt_samples is the number of target domain samples
            and n_features is the number of features.
        Xt2 : array, shape = [nt2_samples, n_features]
            nt2_samples is the number of additional target domain samples
            and n_features is the number of features.
        
        Returns
        -------
        Transformed source and target domain samples Xs_tranformed and Xt_tranformed.
        The transformed additional target domain samples Xt2_tranformed will also returned 
        if Xt2 is not None.
        """        
        X = np.vstack((Xs,Xt))
        n,ns,nt = X.shape[0],Xs.shape[0],Xt.shape[0]
        
        # Construct kernel matrix
        if 'linear' == self.kernel:
            K = mp.linear_kernel(X,X)            
            if Xt2 is not None:
                K2 = mp.linear_kernel(Xt2,X)
        elif 'rbf' == self.kernel:
            K = mp.rbf_kernel(X,X,self.gamma)
            if Xt2 is not None:
                K2 = mp.rbf_kernel(Xt2,X,self.gamma)
        elif 'poly' == self.kernel:
            K = mp.polynomial_kernel(X,X,self.degree,1,self.c)
            if Xt2 is not None:
                K2 = mp.polynomial_kernel(Xt2,X,self.degree,1,self.c)        
        
        # Construct centering matrix
        H = np.eye(n) - 1 / n * np.ones((n,n))

        # Construct MMD matrix
        e = np.hstack((1 / ns * np.ones(ns),-1 / nt * np.ones(nt)))[:,None]
        L = e.dot(e.T)    
        
        #Construct the label kernel matrix, the kernel is fixed at a linear kernel.
        y = ys[:,None]
        Kl = mp.linear_kernel(y,y)
        Kyy = np.zeros((n,n))
        Kyy[:ns,:ns] = self.gamma * Kl
        Kyy = Kyy + (1 - self.gamma) * np.eye(n)
        
        #construct the graph Laplacian matrix
        #the k-nearest neighbors based graph Laplacian matrix described in the original paper
        #is ambiguous,here the k-nearest neighbors constraint is removed.The affinity between two points
        #x1 and x2 is measured by a rbf funtion rbf(x1,x2).
        M = mp.rbf_kernel(X,X,0.5 * self.sigma**-2)
        idx = np.arange(n)
        M[idx,idx] = 0
        D = np.diag(M.sum(1))
        Lap = D - M 
        
        #eigenvalue decomposition
        Ma =  la.inv(K.dot(L + self.lamda * Lap).dot(K) + self.mu * np.eye(n)).dot(K).dot(H).dot(Kyy).dot(H).dot(K)
        eig_vals,eig_vecs = la.eigh(Ma)
        #select the eigenvectors corresponding to the m=self.dimension largest eigenvalues.
        W = eig_vecs[:,::-1][:,:self.dimension] 

        Z = K.dot(W)
        Xs_tranformed,Xt_tranformed = Z[:ns,:],Z[ns:,:]
        
        if Xt2 is not None:
            Xt2_transformed = K2.dot(W)
            return Xs_tranformed,Xt_tranformed,Xt2_transformed
        else:
            return Xs_tranformed,Xt_tranformed
