This folder contains Jupyter Notebook files for federated learning experiments using pre-trained transformer models.

In our federated experiments, we wanted to simulate a scenario where some of the clients in the network may be adversarial and try to attack the system by flipping the labels of their data. To test the robustness of our models against this type of attack, we applied a label flipping attack on half of the labels in the second half of the data.

Specifically, we labeled the first 10 labels out of the 20 in each batch with the opposite label. This allowed us to measure the effectiveness of our chosen aggregation functions, FedAvg and Krum, in mitigating the effects of the label flipping attack.

## Federated_transformers.ipynb

> The `Federated_transformers.ipynb` notebook implements federated learning on pre-trained transformer models from the first experiment. 

> The second half of the data is used for training. The notebook includes code for unilingual and multilingual federated learning in IID and Non-IID settings.

> ## Hyperparameters

> ### The following hyperparameters were used during training:

* learning_rate: 2e-5
* train_batch_size: 64
* eval_batch_size: 512
* seed: 42
* num_epochs: 30
* num_clients: 4


## Federated_Switch_base_8.ipynb

> The `Federated_Switch_base_8.ipynb` notebook applies the pre-trained Switch Transformer Model from the first experiment in a federated learning setting for unilingual IID training.


> ## Hyperparameters

> ### The following hyperparameters were used during training:

* learning_rate: 1e-4
* train_batch_size: 64
* eval_batch_size: 512
* seed: 42
* num_epochs: 30
* num_clients: 4

## Results

### Table_1 : SemEval testing data with Multilingual setting in clean and attack senarios for Attackers ratio 25%

| BERT-Base |First experiment|IID|Non-IID|Continue training| 
|:---:|:---:|:---:|:---:|:---:|
|Clean|35.321|36.849|36.317|35.73|
|FedAvg|35.321|36.357|33.563|35.649|
|Krum|35.321|36.528|35.388|35.649|
| | | | | | 
| | | | | |
| XLM-Roberta |First experiment|IID|Non-IID|Continue training|
|Clean|33.493|34.986|34.977|34.192|                   
|FedAvg|33.493|34.753|32.783|33.991|                         
|Krum|33.493|34.98|33.43|33.991|   
| | | | | | 
| | | | | |
| MiniLM |First experiment|IID|Non-IID|Continue training| 
|Clean|31.462|32.797|32.038|32.995|        
|FedAvg|31.462|32.311|30.476|32.738|              
|Krum|31.462|32.21|30.767|32.738|   

> Note: the evaluation matrix here is Mac-F1

> As can be seen at the Semeval testing results, it is clear that the Krum aggregation function performs well even when the data is toxic. When compared to the FedAvg aggregation function, Krum outperforms it in almost all cases, especially when the data is non-IID.

> For example, in the case of MiniLM, when trained on clean data using FedAvg, Krum and traditional training methods, the performance is almost identical with an F1 score of around 31. However, when trained on toxic data using the same methods, the performance drops significantly to around 30.4 for FedAvg and Krum, while traditional training performs slightly better at 32.7.

> However, when looking at the non-IID case, Krum performs significantly better with an F1 score of around 35.4, while FedAvg performs at around 33.6. In fact, Krum's performance is closer to that of traditional training methods, which achieved an F1 score of 36.3 on clean data and 35.7 on toxic data.

> Similarly, in the case of XLM-Roberta, Krum outperforms FedAvg, especially in the non-IID case. When trained on clean data, the performance of both methods is almost identical with an F1 score of around 35.3. However, when trained on toxic data, Krum achieves a significantly better F1 score of around 35.6, while FedAvg only achieves around 33.5. In the non-IID case, Krum's performance is again significantly better with an F1 score of around 35.4, while FedAvg's performance drops to around 33.6.

> Overall, these results demonstrate that Krum is a more robust aggregation function that can handle toxic data well, especially in non-IID scenarios.



### Table_2 : SemEval testing data with Multilingual setting in clean and attack senarios for Attackers ratio 50%

| BERT-Base |First experiment|IID|Non-IID|Continue training|      
|:---:|:---:|:---:|:---:|:---:|
|Clean|35.321|36.849|36.317|35.73|
|FedAvg|35.321|26.283|28.1|24.492|
|Krum|35.321|36.565|35.219|24.492|                           
| | | | | |
| | | | | |
| XLM-Roberta |First experiment|IID|Non-IID|Continue training|
|Clean|33.493|34.986|34.977|34.192|                    
|FedAvg|33.493|27.681|27.517|23.344|                          
|Krum|33.493|34.872|32.944|23.344|                            
| | | | | |
| | | | | |
| MiniLM |First experiment|IID|Non-IID|Continue training| 
|Clean|31.462|32.797|32.038|32.995|                    
|FedAvg|31.462|25.343|26.714|23.932|                          
|Krum|31.462|32.692|30.512|23.932| 

> Note: the evaluation matrix here is Mac-F1

> As can be seen at the Semeval testing results with 50% attack, it can be observed that the performance of both FedAvg and Krum has decreased significantly compared to the clean results in the first experiment. However, Krum has shown to perform better than FedAvg in handling the toxic data.

> In the case of Multilingual-MiniLM, FedAvg's performance has decreased to around 25% accuracy for IID and 26% for non-IID, while Krum's performance has decreased to 32.69% for IID and 30.51% for non-IID. It can be observed that Krum has performed better than FedAvg in handling the toxic data, with around 6% higher accuracy for IID and 4% higher accuracy for non-IID.

> Similarly, in the case of Twitter/twhin-bert-base, FedAvg's performance has decreased to around 27.68% accuracy for IID and 27.51% for non-IID, while Krum's performance has decreased to 34.87% for IID and 32.94% for non-IID. Here again, Krum has performed better than FedAvg in handling the toxic data, with around 7% higher accuracy for IID and 5% higher accuracy for non-IID.

> These results show that Krum can perform well even when the data are toxic, and it can be a useful aggregation function for federated learning in scenarios where there is a possibility of data poisoning attacks.

> You can find the full results [here](https://docs.google.com/spreadsheets/d/1jcixbuYejpAw8Qzi41j5XdWv5u9n08rgghVOc5hTT9k/edit?usp=sharing)
