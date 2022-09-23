from __future__ import division
import itertools
import autograd as ag
import scipy.stats as st
import autograd.numpy as np
import scipy.optimize as opt
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold,ParameterGrid
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder

class GCSC(BaseEstimator, ClassifierMixin):
    """ 
    Generalized Covarite Shift Classifier
    Parameters
    ----------
    loss: function, default: None
        The surragate loss function used for classification,can be one of log(logistic loss), 
        hinge,square_hinge,square or exp(exponential loss).
    pos_func: function, default: None
        A positive function for the density ratio.
    Xt: 2-d numpy array, default: None
        The unlabeled target domain data for estimating the instance weight.
    kernel: str, default: 'linear'
        Can be one of 'linear' or 'rbf'(rbf network).
    n_kernels: int, default:5000
        Number of kernel basis centers for both the classification model and the density ratio model,
        default to be 5000. Use a random subset of the target domain data rather than the source domain data 
        to be the kernel basis centres.
    sigma: float, default: 0.8
        Kernel bandwidth of the gaussian kernel,default to be 0.8.
    lamda_wht: float, default: 1.0
        Regularization parameter for the parameters of the density ratio model.
        The larger, the stronger regularization.
    lamda_clf: float, default: 1.0
        Regularization parameter for the parameters of the classification model.
        The larger, the stronger regularization.    
        
    """ 
    def __init__(self,loss=None,pos_func=None,Xt=None,kernel='linear',n_kernels=5000,
                 sigma=0.8,lamda_wht=1.0,lamda_clf=1.0):
        args_values = locals()
        args_values.pop("self")
        for arg,value in args_values.items():
            setattr(self,arg,value) 
            
    def set_parameters(self, parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, Xs, ys):
        """Fit the model according to the source domain data.
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
        
        self.classes_ = np.unique(ys)
        nt = self.Xt.shape[0]                
        alpha={}
        theta={}

        #the one vs one for multiclass classification
        for couple in itertools.combinations(self.classes_, 2):
            # process two classes of data every time
            tr_idx = np.bitwise_or(ys==couple[0],ys==couple[1])
            X_tr,y_tr = Xs[tr_idx],ys[tr_idx]        
                     
            #construct the kernel matrix for the density ratio model and the classification model          
            if 'linear'==self.kernel:
                K_ST = np.hstack((np.ones((X_tr.shape[0],1)),X_tr))
                K_TT = np.hstack((np.ones((nt,1)),self.Xt))
            elif 'rbf'==self.kernel:
                idx = np.random.permutation(nt)
                self.kernel_centers_ = self.Xt[idx[:min(self.n_kernels,nt)]]
                K_ST = rbf_kernel(X_tr,self.kernel_centers_,0.5 * self.sigma**-2)
                K_TT = rbf_kernel(self.Xt,self.kernel_centers_,0.5 * self.sigma**-2)   
            else:
                print 'kernel not supported !'
                return            
            dim = K_ST.shape[1]
            
            y_tr[y_tr==couple[0]] = -1
            y_tr[y_tr==couple[1]] = 1 
            
            def LossFunc(vec): 
                p1 = np.mean( self.pos_func(np.dot(K_ST,vec[:dim])) * self.loss(y_tr * np.dot(K_ST,vec[dim:])))
                p2 = np.mean(self.pos_func(np.dot(K_ST,vec[:dim]))) - np.mean(np.dot(K_TT,vec[:dim]))
                p4 = 0.5 * self.lamda_wht * np.dot(vec[:dim],vec[:dim]) + 0.5 * self.lamda_clf * np.dot(vec[dim:],vec[dim:])                    
                return p1 + p2 + p4            
            
            #cons = {'type': 'eq', 'fun': lambda vec:np.mean(self.pos_func(np.dot(K_ST,vec[:dim]))) - 1 },constraints=cons
            res = opt.minimize(fun=LossFunc,jac=ag.grad(LossFunc), x0=1e-3*np.random.rand(2*dim) ,method='L-BFGS-B')
            #print res.success
            alpha[couple] = res.x[:dim]
            theta[couple] = res.x[dim:]
            #print self.pos_func(np.dot(K_ST,alpha[couple]))
         
        self.alpha_ = alpha
        self.theta_ = theta        
        return self
        
    def predict(self, Xt):
        """ 
        Parameters
        ----------
        Xt : array, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        y_hat={}
        for couple in itertools.combinations(self.classes_, 2):
            #construct the test kernel matrix            
            if 'linear'==self.kernel:
                K_te = np.hstack((np.ones((Xt.shape[0],1)),Xt))
            else:
                K_te = rbf_kernel(Xt,self.kernel_centers_,0.5 * self.sigma**-2)                
            
            real = K_te.dot(self.theta_[couple])
            y_hat[couple] = np.where(real < 0,couple[0],couple[1])
        
        #use majority vote to decide the label
        y_hat = np.array(y_hat.values())
        predictions = st.mode(y_hat,0)[0][0]
        return predictions
    
    def weighted_error(self,Xs,ys,ws):        
        y_hat = self.predict(Xs)
        w_err = np.mean((ys!=y_hat)*ws)
        return w_err
    
    def score(self,Xt,yt):
        """
        Parameters
        ----------
        Xt : array, shape = [n_samples, n_features]
        
        yt : array-like, shape = [n_samples]
        Returns
        -------
        test score : scalar
        """ 
        y_hat = self.predict(Xt)
        acc = np.mean(yt==y_hat)
        return acc  
    
#=====optional loss function======== 
#logistic loss
def log(x): return np.logaddexp(0,-x)#use np.logaddexp to prevent overflow problem    

#exponential loss
def exp(x): return np.exp(-np.where(x<-350,-350,x))#use np.where to prevent overflow problem

#hinge loss
def hinge(x): return np.maximum(0,1-x)

#square hinge loss
def square_hinge(x): return np.maximum(0,1-x)**2

#square loss
def square(x): return (1 - x)**2

#=====positive function=============

def expon(x):return np.exp(np.where(x>350,350,x))

#=========nonlinear polynomial features and label encoder============================
def poly(X,degree=2): return PolynomialFeatures(degree,include_bias=False).fit_transform(X)
def labelEncoder(y): return LabelEncoder().fit_transform(y)


# =======can only parallel on Linux============
def fit_and_error(estimator,Xs,ys,ws,tr_te_idx,cv,params):
    estimator.set_parameters(params)
    err_per_fold = np.zeros(cv)
    for i in range(cv):
        tr_idx,te_idx=tr_te_idx[i][0],tr_te_idx[i][1]
        estimator.fit(Xs[tr_idx],ys[tr_idx])
        err_per_fold[i] = estimator.weighted_error(Xs[te_idx],ys[te_idx],ws[te_idx])
    return np.mean(err_per_fold)

class ParallelWCV:
    
    def __init__(self,estimator,param_grid,cv=10,n_jobs=1):
        args_values = locals()
        args_values.pop("self")
        for arg,value in args_values.items():
            setattr(self,arg,value)

    def fit(self,Xs,ys,ws):
        param_list = list(ParameterGrid(self.param_grid))
        tr_te_idx = list(KFold(n_splits = self.cv,shuffle = True).split(Xs))
        parallel = Parallel(n_jobs=self.n_jobs)
        cv_error = parallel(delayed(fit_and_error)(self.estimator,Xs,ys,ws,tr_te_idx,self.cv,params) for params in param_list)        
        cv_error = np.array(cv_error)
        self.estimator.set_parameters(param_list[cv_error.argmin()])
        self.estimator.fit(Xs,ys)
        return self
    
    def predict(self,X):
        return self.estimator.predict(X)

    def score(self,Xt,yt): 
        y_hat = self.estimator.predict(Xt)
        acc = np.mean(yt==y_hat)
        return acc    

#======Weighted Cross Validation but cannot be paralleled=============
#we can also use GridSearchCV to do normal cross validation
class WCV:
    
    def __init__(self,estimator,param_grid,cv=5):
        args_values = locals()
        args_values.pop("self")
        for arg,value in args_values.items():
            setattr(self,arg,value)

    def fit(self,Xs,ys,ws):
        tr_te_idx = list(KFold(n_splits = self.cv,shuffle = True).split(Xs))
        param_list = list(ParameterGrid(self.param_grid))
        cv_error = np.zeros(len(param_list))
        for i, params in enumerate(param_list):
            self.estimator.set_parameters(params)
            err_per_fold = np.zeros(self.cv)
            for j in range(self.cv):
                tr_idx,te_idx=tr_te_idx[j][0],tr_te_idx[j][1]                
                self.estimator.fit(Xs[tr_idx],ys[tr_idx])
                err_per_fold[j] = self.estimator.weighted_error(Xs[te_idx],ys[te_idx],ws[te_idx])
            cv_error[i] = np.mean(err_per_fold)
        self.estimator.set_parameters(param_list[cv_error.argmin()])
        self.estimator.fit(Xs,ys)
        return self
    
    def predict(self,X):        
        return self.estimator.predict(X)
    
    def score(self,Xt,yt): 
        y_hat = self.estimator.predict(Xt)
        acc = np.mean(yt==y_hat)
        return acc
