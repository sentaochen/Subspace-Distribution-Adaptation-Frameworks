from __future__ import division
import numpy as np
from sklearn.decomposition import PCA

class SA:
    """
    Unsupervised Visual Domain Adaptation Using Subspace Alignment iccv2013
    Code adapted from http://users.cecs.anu.edu.au/~basura/DA_SA/
    Parameters
    ----------
    dimension: int, default: 20
        Dimension of the subspace. 
    
    """
    def __init__(self,dimension=20):
        self.dimension = dimension
    
    def fit_transform(self,Source_Data,Target_Data):
        """Fit and transform the source and target domain data using subspace alignment.
        Parameters
        ----------
        Source_Data : array, shape = [ns_samples, n_features]
            ns_samples is the number of source domain samples
            and n_features is the number of features.Use NormalizeData function to
            normalize the data before.
         Target_Data : array, shape = [nt_samples, n_features]
            nt_samples is the number of target domain samples
            and n_features is the number of features.Use NormalizeData function to
            normalize the data before.
            
        Returns
        -------
        Transformed source and target domain samples Target_Aligned_Source_Data and Target_Projected_Data.
        """ 
        # Subspace alignment and projections                                          
        pca = PCA(n_components=self.dimension)
        Xs = pca.fit(Source_Data).components_.T
        Xt = pca.fit(Target_Data).components_.T
        Target_Aligned_Source_Data = Source_Data.dot(Xs).dot(Xs.T).dot(Xt)
        Target_Projected_Data = Target_Data.dot(Xt)
        return Target_Aligned_Source_Data,Target_Projected_Data     
