# Diverse Preference Augmentation with Multiple Domains for Cold-start Recommendations
This repository is for RCAN introduced in the following paper

[Yan Zhang](), [Changyu Li](), [Ivor W. Tsang](), [Lichen Wang](), [Hui Xu](), [Lixin Duan](), and [Jie Shao](), "Diverse Preference Augmentation with Multiple Domains for Cold-start Recommendations"

## Contents
1. [Introduction](#introduction)
2. [Environment](#Environment)
3. [Running](#Runnning)
4. [Citation](#citation)
5. [Acknowledgements](#acknowledgements)


## Introduction
Cold-start issues have been increasingly challenging for providing accurate recommendations with the increasing users and items. Most existing approaches attempt to solve the intractable problems via hybrid recommendations with auxiliary side information and cross-domain recommendations with transfer learning. Their performances are often constrained by the high sparse user-item interactions, unavailable side information, and very limited shared users between domains. As the meta-learner with meta-augmentation via adding noise to labels without changing inputs is proven to be effective to avoid overfitting and has a good performance on new tasks. Motivated by the idea of meta-augmentation, in this paper, we treat a user’s preference over items as a task and propose a diverse preference augmentation framework with multiple source domains (DPA) to generate diverse ratings in a target domain to avoid overfitting on insufficient interactions. Then, DPA learns a preference model in the target domain via a metalearning scheme to alleviate cold-start issues. Specifically, we first conduct multi-source domain adaptation by dual conditional variational autoencoders and impose a Multi-domain InfoMax (MDI) constraint on the latent representations to learn domain shared and domain-specific preference properties. To generate diverse ratings given content data, we add a Mutual Exclusive (ME) constraint on the output of decoders, which aims to generate mutually exclusive diverse ratings to avoiding overfitting. Finally, we generate diverse ratings by learned decoders and then introduce them into the meta-training procedure to learn a preference meta-learner, which is evaluated to be effective for cold-start recommendations. Experiments on real-world datasets show the performance of the proposed method is significantly superior to the state-of-the-art baselines including content-aware, cross-domain, and mete-learning based recommender systems.

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
