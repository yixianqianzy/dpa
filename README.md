# Title: Diverse Preference Augmentation with Multiple Domains for Cold-start Recommendations
This repository is implementation of MetaDPA submitted to ICDE 2022

[Yan Zhang](), [Changyu Li](), [Ivor W. Tsang](), [Hui Xu](), [Lixin Duan](), [Hongzhi Yin](), [Wen Li](), and [Jie Shao](), "Diverse Preference Augmentation with Multiple Domains for Cold-start Recommendations"

## Contents
1. [Introduction](#introduction)
2. [Environment](#Environment)
3. [Running](#Runnning)
4. [Citation](#citation)
5. [Acknowledgements](#acknowledgements)


## Introduction
Cold-start issues have been more and more challenging for providing accurate recommendations with the fast increase of users and items. Most existing approaches attempt to solve the intractable problems via content-aware recommendations based on auxiliary side information and/or cross-domain recommendations with transfer learning. Their performances are often constrained by the highly sparse user-item interactions, unavailable side information,  or very limited shared users between domains. Recently, meta-learners with meta-augmentation by adding noise to labels have been proven to be effective to avoid overfitting and shown good performance on new tasks. Motivated by the idea of meta-augmentation, in this paper, by treating a user's preference over items as a task, we propose a so-called Diverse Preference Augmentation framework with multiple source domains based on meta-learning (referred to as MetaDPA) to i) generate diverse ratings in a new domain of interest (known as target domain) to handle overfitting on the case of sparse interactions, and to ii) learn a preference model in the target domain via a meta-learning scheme to alleviate cold-start issues. Specifically, we first conduct multi-source domain adaptation by using dual conditional variational autoencoders and impose a Multi-domain InfoMax (MDI) constraint on the latent representations to learn domain-shared and domain-specific preference properties. To generate diverse ratings given content data, we add a Mutual Exclusive (ME) constraint on the output of decoders, which aims to generate mutually exclusive diverse ratings to avoid overfitting. Finally, we generate diverse ratings by learned decoders and then introduce them into the meta-training procedure to learn a preference meta-learner, which produces good generalization ability on cold-start recommendation tasks. Experiments on real-world datasets show our proposed MetaDPA method largely outperforms the current state-of-the-art baselines.

## Environment
Python 3.7.9
Details could be seen on `requirements.txt`

## Running
### Preparing the dataset
1. Download the following [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/) dataset. We only use the following dataset.
|               |                          |
|:--:           |:--:                      |
|Books	        |5-core (8,898,041 reviews)|
|Electronics	|5-core (1,689,188 reviews)|
|Movies and TV	|5-core (1,697,533 reviews)|
|CDs and Vinyl	|5-core (1,097,592 reviews)|
|Digital Music	|5-core (64,706 reviews)   |
2. Download the pretrain word2vec [GoogleNews-vectors-negative300.bin.gz.](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) from the official [word2vec website](https://code.google.com/archive/p/word2vec/).

### Running
Follow the `Readme.md` in 
```
1datasetprepare
2word2vec
3generation
4evaluation
```

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
```

# Acknowledgement.
This code refers code from:

[xuChenSJTU/ETL-master](https://github.com/xuChenSJTU/ETL-master)
[Wenhui-Yu/TDAR](https://github.com/Wenhui-Yu/TDAR)
[waterhorse1/MELU_pytorch](https://github.com/waterhorse1/MELU_pytorch)

We thank the authors for sharing their codes!
