# Training files for Emojis Classification

This folder contains the following training files:

* `Train_file_for_Emojis_with_Switch_base_8.ipynb` & `Train_file_Emojis_transformers.ipynb`: These files were used for the first & third experiments ( continue training ), where we trained on the first half & the second half of the data for both unilingual and multilingual models.

## Training Hyperparameters

> ### Switch Transformer Model

> The following hyperparameters were used during training of the Switch Transformer Model:

* learning_rate: 1e-4
* train_batch_size: 64
* eval_batch_size: 512
* seed: 42
* num_epochs: 30

> ### Other Transformer Models

> The following hyperparameters were used during training of other transformer models:

* learning_rate: 2e-5
* train_batch_size: 64
* eval_batch_size: 512
* seed: 42
* num_epochs: 30

> These hyperparameters were used for both unilingual and multilingual models.

> Note that the saved tuned transformer model was used for continued training in the second experiment.

## Results

> We trained and tested our models using the data collected and preprocessed from Twitter for five languages: English SemEval, Spanish, German, French, and Italian. The following tables show the evaluation metrics for our best-performing models on each language.

### Table_1 : SemEval testing data with Unilingual setting

> | | Model                               | First experiment | IID    | Continue training | |
> |:---:|:---:|:---:|:---:|:---:|:---:|
> | | "Tubingen-Oslo" First SemEval Team  | -                | -      | 35.99             | |
> | | switch-base-8                       | [33.239](https://huggingface.co/Karim-Gamal/switch-base-8-finetuned-SemEval-2018-emojis-cen-1)           | [37.355](https://huggingface.co/Karim-Gamal/switch-base-8-finetuned-SemEval-2018-emojis-IID-Fed) | [36.66](https://huggingface.co/Karim-Gamal/switch-base-8-finetuned-SemEval-2018-emojis-cen-2)             | |
> | | Twitter/twhin-bert-base             | [36.96](https://huggingface.co/Karim-Gamal/BERT-base-finetuned-SemEval-2018-emojis-cen-1)            | [37.475](https://huggingface.co/Karim-Gamal/BERT-base-finetuned-SemEval-2018-emojis-IID-Fed) | [38.133](https://huggingface.co/Karim-Gamal/BERT-base-finetuned-SemEval-2018-emojis-cen-2)            | |
> | | cardiffnlp/twitter-xlm-roberta-base | [35.971](https://huggingface.co/Karim-Gamal/XLM-Roberta-finetuned-SemEval-2018-emojis-cen-1)           | [36.727](https://huggingface.co/Karim-Gamal/XLM-Roberta-finetuned-SemEval-2018-emojis-cen-2) | [37.647](https://huggingface.co/Karim-Gamal/XLM-Roberta-finetuned-SemEval-2018-emojis-IID-Fed)            | |
> | | Multilingual-MiniLM                 | [33.368](https://huggingface.co/Karim-Gamal/MMiniLM-L12-finetuned-SemEval-2018-emojis-cen-1)           | [33.907](https://huggingface.co/Karim-Gamal/MMiniLM-L12-finetuned-SemEval-2018-emojis-IID-Fed) | [35.937](https://huggingface.co/Karim-Gamal/MMiniLM-L12-finetuned-SemEval-2018-emojis-cen-2)            | |


> Note: the evaluation matrix here is Mac-F1

> As can be seen the best-performing models for this task are the Twitter/twhin-bert-base and cardiffnlp/twitter-xlm-roberta-base models. These models consistently outperform the other models in all three experiments (first experiment, IID, and continue training). The switch-base-8 model also performs relatively well, particularly in the IID experiment, but is outperformed by the other two models overall.

> The Multilingual-MiniLM model performs the worst among the transformer models, although its performance improves slightly in the continue training experiment.

> Overall, these results suggest that transformer-based models are better suited for this task, and that fine-tuned models such as Twitter/twhin-bert-base and cardiffnlp/twitter-xlm-roberta-base may be particularly effective. However, it's worth noting that these results are specific to the dataset and experimental setup used, and may not generalize to other tasks or datasets.

### Table_2.1 : SemEval testing data with Multilingual setting


> |Model|First experiment|IID|Non-IID|Continue training|
> |:---:|:---:|:---:|:---:|:---:|
> | "Tubingen-Oslo" First SemEval Team  | - | - | - | 35.99 |
> |Twitter/twhin-bert-base|[35.321](https://huggingface.co/Karim-Gamal/BERT-base-finetuned-emojis-cen-1)|[36.849](https://huggingface.co/Karim-Gamal/BERT-base-finetuned-emojis-IID-Fed)|[36.317](https://huggingface.co/Karim-Gamal/BERT-base-finetuned-emojis-non-IID-Fed)|[35.73](https://huggingface.co/Karim-Gamal/BERT-base-finetuned-emojis-cen-2)|
> |cardiffnlp/twitter-xlm-roberta-base |[33.493](https://huggingface.co/Karim-Gamal/XLM-Roberta-finetuned-emojis-cen-1)|[34.986](https://huggingface.co/Karim-Gamal/XLM-Roberta-finetuned-emojis-IID-Fed)|[34.977](https://huggingface.co/Karim-Gamal/XLM-Roberta-finetuned-emojis-non-IID-Fed)|[34.192](https://huggingface.co/Karim-Gamal/XLM-Roberta-finetuned-emojis-cen-2)|
> |Multilingual-MiniLM |[31.462](https://huggingface.co/Karim-Gamal/MMiniLM-L12-finetuned-emojis-cen-1)|[32.797](https://huggingface.co/Karim-Gamal/MMiniLM-L12-finetuned-emojis-IID-Fed)|[32.038](https://huggingface.co/Karim-Gamal/MMiniLM-L12-finetuned-emojis-non-IID-Fed)|[32.995](https://huggingface.co/Karim-Gamal/MMiniLM-L12-finetuned-emojis-cen-2)|


> Note: the evaluation matrix here is Mac-F1

> Based on the results table, it seems that the Twitter/twhin-bert-base model performed the best in all four experiments, followed by cardiffnlp/twitter-xlm-roberta-base and Multilingual-MiniLM. The difference between the models is not significant, but the Twitter/twhin-bert-base model achieved the highest accuracy in all experiments.

> Also, it is worth noting that the performance of the models varied when training on IID versus Non-IID datasets. In general, models trained on IID datasets achieved higher accuracy than models trained on Non-IID datasets. This suggests that having balanced data is important for achieving good performance in multiclass classification tasks.

> Finally, continuing the training of the models on the second half of the data seems to have improved the performance of all models, where it performed worse than in the first experiment.

### Table_2.2 : Spanish testing data with Multilingual setting

> |Model|First experiment|IID|Non-IID|Continue training|
> |:---:|:---:|:---:|:---:|:---:|
> |Twitter/twhin-bert-base|27.71|28.506|28.817|27.306|
> |cardiffnlp/twitter-xlm-roberta-base |24.638|26.153|26.21|26.406|
> |Multilingual-MiniLM |22.192|22.898|23.792|23.418|


### Table_2.3 : French testing data with Multilingual setting

> |Model|First experiment|IID|Non-IID|Continue training|
> |:---:|:---:|:---:|:---:|:---:|
> |Twitter/twhin-bert-base|29.699|31.317|31.146|30.915|
> |cardiffnlp/twitter-xlm-roberta-base |26.915|28.799|28.522|28.921|
> |Multilingual-MiniLM |24.776|26.385|26.481|26.558|



### Table_2.4 : Italian testing data with Multilingual setting

> |Model|First experiment|IID|Non-IID|Continue training|
> |:---:|:---:|:---:|:---:|:---:|
> |Twitter/twhin-bert-base|30.294|32.234|32.356|32.455|
> |cardiffnlp/twitter-xlm-roberta-base |28.081|29.769|29.338|29.793|
> |Multilingual-MiniLM |25.783|27.872|28.048|27.995|



### Table_2.5 : German Zero Shot testing data with Multilingual setting

> |Model|First experiment|IID|Non-IID|Continue training|
> |:---:|:---:|:---:|:---:|:---:|
> |Twitter/twhin-bert-base|21.966|23.155|23.145|21.505|
> |cardiffnlp/twitter-xlm-roberta-base |20.042|20.972|21.079|19.261|
> |Multilingual-MiniLM |15.489|16.578|16.464|17.142|


> Note: the evaluation matrix here is Mac-F1

> As can be seen the zero-shot German language testing, it seems that the performance of all models is lower compared to the previous experiments. This is expected as the models were not trained on German data specifically.

> The Twitter/twhin-bert-base model achieved the highest scores among the models, but still relatively low. The cardiffnlp/twitter-xlm-roberta-base and Multilingual-MiniLM models had similar performance, with the latter slightly outperforming the former. 

> Overall, these results suggest that fine-tuning on German data may be necessary to achieve better performance on German language tasks.

> You can find the full results [here](https://docs.google.com/spreadsheets/d/1jcixbuYejpAw8Qzi41j5XdWv5u9n08rgghVOc5hTT9k/edit?usp=sharing)
