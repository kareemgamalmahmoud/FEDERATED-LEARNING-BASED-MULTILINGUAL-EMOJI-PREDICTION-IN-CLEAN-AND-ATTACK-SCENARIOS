# FEDERATED LEARNING BASED MULTILINGUAL EMOJI PREDICTION IN CLEAN AND ATTACK SCENARIOS

> This repository contains the code and data for our paper on Federated Learning for Multilingual Emoji Prediction.

# Abstract

> Federated learning is a rapidly growing field in the machine learning community. This growth is because federated
learning is decentralized and private by design. Model training in federated learning is distributed over multiple
clients where each client has its own private data. Then, a server aggregates the training done on these multiple
clients without access to their data, which could be emojis widely used in any social media service and instant
messaging platforms to express users’ sentiments. 

> Therefore, this paper proposes federated learning-based
multilingual emoji prediction in both clean and attack scenarios. Emoji prediction data have been crawled from
both Twitter and standard SemEval emoji datasets. This data is used to train and evaluate different transformer
model sizes including a sparsely activated transformer with either the assumption of clean data in all clients or
poisoned data via label flipping attack in some clients. Experimental results on these models show that federated
learning in either clean or attacked scenarios performs similarly to centralized training in multilingual emoji
prediction on seen and unseen languages under different data sources and distributions. This performance is on
top of the privacy and distributed benefits of federated learning.

# Methodology:

> In this section, we describe the methodology used in our study. The methodology is divided into three main parts: unilingual experiments , multilingual experiments and label flipping experiments.

## Unilingual Experiments

> To evaluate the performance of different transformer models in predicting emojis, we conducted unilingual experiments using the SemEval emoji dataset. 

> We trained and evaluated models in both centralized and federated settings, conducting three different experiments:

* The first experiment involved centralized training on the first half of the data.
* The second involved federated learning with the pretrained model from the first experiment trained on the second half of the data in a federated iid setting.
* The third experiment involved continued training of the pretrained model from the first experiment on the second half of the data to compare results between the three experiments and assess the impact of federated learning.

## Multilingual Experiments:

> We repeated the above experiments, but with additional languages such as French, Spanish, and Italian included in the second experiment to create a non-iid setting.  
## Label Flipping Attack Experiments:

> In our federated experiments, we aimed to simulate adversarial clients in the network attempting to attack the system by flipping data labels. To test the robustness of our models against such attacks, we applied a label flipping attack to half of the labels in the second half of the data.

> Specifically, we labeled the first 10 out of 20 labels in each batch with the opposite label to measure the effectiveness of our chosen aggregation functions, FedAvg and Krum, in mitigating the impact of the label flipping attack.

## Results : 
> We have presented the results in these two directories.
> - [Federated_training](/Federated_training)
> - [Centralized_training](/Centralized_training)


## The Team : 

> ![image](https://user-images.githubusercontent.com/51359449/228678951-2392e73e-0436-454b-8651-6327f159763c.png)


## Demo: 
> [Link](https://huggingface.co/spaces/Karim-Gamal/Federated-Learning-Based-Multilingual-Emoji-Prediction-Demo-2)
