# Subspace Distribution Adaptation Frameworks

This repository provides the pdf file of the paper "Subspace distribution adaptation frameworks for domain adaptation" published in IEEE TNNLS, and the Python2 codes for the proposed method in the paper. The method is in the file ours.py, and is implemented following the sklearn style. That is, the method is wrapped into a class and has the common operations: fit, predict, score. You can run the Demo.ipynb to see the performance of the method on a synthetic data example. For comparison, the Python2 codes of other domain adaptation methods, including Transfer Compnent Aanalysis (TCA), Subspace Alignment (SA), Joint Geometrical and Statistical Alignment (JGSA) are also provided in the files tcasstca2011.py, sa2013.py, jgsa2017.py, respectively. These codes are adapted from their official MATLAB counterparts, or are implemented following the description in the corresponding papers.  

In a nutshell, this paper introduces the generalized covariate shift assumption to the domain adaptation problem, and proposes 2 subspace distribution adaptation frameworks that align the source distribution to the target distribution in a subspace. By properly choosing the seed function or loss function, the frameworks can lead to convex optimization problems.


More publication information of this paper is listed below: 

@article{Chen2020Subspace,  
  author={Chen, Sentao and Han, Le and Liu, Xiaolan and He, Zongyao and Yang, Xiaowei},  
  journal={IEEE Transactions on Neural Networks and Learning Systems},   
  title={Subspace Distribution Adaptation Frameworks for Domain Adaptation},   
  year={2020},  
  volume={31},  
  number={12},  
  pages={5204-5218},  
  doi={10.1109/TNNLS.2020.2964790}  
  }
