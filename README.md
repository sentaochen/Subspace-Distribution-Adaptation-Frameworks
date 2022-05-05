# Subspace Distribution Adaptation Methods

This repository provides a python 2 implementation (JDIP.py) of the domain adaptation approach JDIP. The implementation relies on the pymanopt toolbox available at https://www.pymanopt.org/. The jupyter notebook demo.ipynb shows how to run this method in a semi-supervised domain adaptation setting.

Briefly speaking, the goal of JDIP is to solve the joint distribution mismatch problem in domain adaptation. To this end, it exploits a couple of points on the Stiefel manifold to match the source and target joint distributions under the $L^{2}$-distance. The following figure illustrates this joint distribution matching idea.


For more details of this domain adaptation approach,  please refer to our IEEE TNNLS work: 

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
