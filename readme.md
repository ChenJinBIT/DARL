# DARL
This is the implementation of ''Domain Adversarial Reinforcement Learning for Partial Domain Adaptation'' using Tensorflow. The original paper can be found at
https://ieeexplore.ieee.org/abstract/document/9228896/

### Environment: Python 3.6 and Tensorflow 1.14.0

### An example of SVHN→MNIST dataset with second setting in original paper.

1. prepare datalist in data_list directory
2. pretrain model by running finetune.py and the example pre-trained SVHN→MNIST source model can be downloaded from:https://drive.google.com/drive/folders/15tEqHFtVNw779-nNJE_cLUEz5-6FIvKA?usp=sharing
3. running main.py

### Citation
@article{chen2020domain,\
  title={Domain adversarial reinforcement learning for partial domain adaptation},\
  author={Chen, Jin and Wu, Xinxiao and Duan, Lixin and Gao, Shenghua},\
  journal={IEEE Transactions on Neural Networks and Learning Systems},\
  year={2020},\
  publisher={IEEE}\
}
