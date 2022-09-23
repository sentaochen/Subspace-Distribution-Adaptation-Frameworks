from __future__ import division
import itertools
import numpy as np
import scipy.stats as st
import scipy.optimize as opt
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder

#check the note to see the gradient of the empirical loss function
class Classifier(BaseEstimator, ClassifierMixin):
    """
    Multi-class classification with arbitrary loss function using one vs one strategy.   
    
    Parameters
    ----------
    loss : function, default: None
        The surragate loss function used for classification.
    loss_grad: function, default: None
        The gradient corresponding to the loss function. 
    kernel: str, default: 'linear'
        Can be one of 'linear' or 'rbf'(rbf network).
    n_kernels: int, default:5000
        Number of kernel basis centers for the rbf network, default to be 5000.
    sigma: float,default: 0.8
        Kernel bandwidth of the gaussian kernel,default to be 0.8.
    lamda: float,default: 1.0
        Regularization parameter for the parameters of the classification model.
        The larger, the stronger regularization.
        
    """             
    def __init__(self,loss=None,loss_grad=None,kernel='linear',n_kernels=5000,sigma=0.8,lamda=1.0):
        args_values = locals()
        args_values.pop("self")
        for arg,value in args_values.items():
            setattr(self,arg,value)

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target class labels.
        
        Returns
        -------
        self : returns an instance of self.
        """
        self.classes_ = np.unique(y)        
        kernel_centers={}
        theta={}
       
        #the one vs one for multiclass classification
        for couple in itertools.combinations(self.classes_, 2):
            tr_idx = np.bitwise_or(y==couple[0],y==couple[1])
            X_tr,y_tr = X[tr_idx],y[tr_idx]
            
            #construct the training kernel matrix             
            if 'linear' == self.kernel:
                K_tr = np.hstack((np.ones((X_tr.shape[0],1)),X_tr))
            elif 'rbf' == self.kernel:
                idx = np.random.permutation(X_tr.shape[0])
                kernel_centers[couple] = X_tr[idx[:min(self.n_kernels,X_tr.shape[0])]]
                K_tr = rbf_kernel(X_tr,kernel_centers[couple],0.5 * self.sigma**-2)
            else:
                print 'kernel not supported !'
                return
            
            y_tr[y_tr==couple[0]] = -1
            y_tr[y_tr==couple[1]] = 1
            
            #the initial point for optimization should be small to avoid overflow
            res = opt.minimize(fun=lambda alpha: np.sum( self.loss(y_tr * K_tr.dot(alpha)) ) 
                                       + 0.5 * self.lamda * alpha.dot(alpha),
                                x0=1e-3 * np.random.randn(K_tr.shape[1]),method='L-BFGS-B',
                                jac=lambda alpha: self.lamda * alpha + 
                                        K_tr.T.dot( self.loss_grad(y_tr * K_tr.dot(alpha)) * y_tr ))
            #print res.success            
            theta[couple] = res.x
         
        self.kernel_centers_ = kernel_centers
        self.theta_ = theta
        return self
        
    def predict(self, X):
        """ 
        Parameters
        ----------
        X : array, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        y_hat={}
        for couple in itertools.combinations(self.classes_, 2):
            #construct the test kernel matrix
            if 'linear' == self.kernel:
                K_te = np.hstack((np.ones((X.shape[0],1)),X))
            else:
                K_te = rbf_kernel(X,self.kernel_centers_[couple],0.5 * self.sigma**-2)
                
            real = K_te.dot(self.theta_[couple])
            y_hat[couple] = np.where(real < 0,couple[0],couple[1])
        
        #use majority vote to decide the label
        y_hat = np.array(y_hat.values())
        predictions = st.mode(y_hat,0)[0][0]
        return predictions
        
    def score(self,X,y):
        """
        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
        
        y : array-like, shape = [n_samples]

        Returns
        -------
        test score : scalar
        """
        
        y_hat = self.predict(X)
        acc = np.mean(y==y_hat)
        return acc      

#=====optional loss function and loss function gradient======== 
#logistic loss
def log(x): return np.logaddexp(0,-x)#use np.logaddexp to prevent overflow problem    
def log_grad(x): return -expit(-x)#use expit to prevent overflow problem 

#exponential loss
def exp(x): return np.exp(-np.where(x<-500,-500,x))#use np.where to prevent overflow problem
def exp_grad(x): return -np.exp(-np.where(x<-500,-500,x))

#hinge loss
def hinge(x): return np.maximum(0,1-x)
def hinge_grad(x): return -(x < 1).astype(np.float64)

#square hinge loss
def square_hinge(x): return np.maximum(0,1-x)**2
def square_hinge_grad(x): return  -2 * np.maximum(0,1-x) * (x < 1).astype(np.float64)

#square loss
def square(x): return (1 - x)**2
def square_grad(x): return 2 * (x - 1)

#=========nonlinear polynomial features and label encoder============================
def poly(X,degree=2): return PolynomialFeatures(degree,include_bias=False).fit_transform(X)
def labelEncoder(y): return LabelEncoder().fit_transform(y)
