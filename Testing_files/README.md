# Emoji Prediction Testing

This folder contains Jupyter Notebook files for testing the performance of different models on the task of detecting emojis in text. The files are described below:

> * `Test_for_Emojis_with_Switch_base_8.ipynb`: This file loads the pre-trained Switch model that was tuned using the base-8 architecture and tests it on the SemEval testing dataset. The notebook includes code for loading the model and the dataset, as well as evaluating the model's performance using metrics such as accuracy and F1-score.

> *  `Test_for_Emojis_with_Transformers.ipynb`: This file is used to test any transformer model created for the SemEval emojis task, whether it is a unilingual or multilingual model. The notebook includes code for loading the pre-trained transformer model and the testing dataset, as well as evaluating the model's performance using various metrics.


## Getting Started

To run the app, follow these steps:

> 1- Clone the repository and navigate to the root directory.

> 2- Install the required Python libraries using the following command:
```
pip install -r requirements.txt
```

> 3- Obtain your Huggingface model name by following the steps below.

> 4- Run the testing file using the following command:

```
python test_file.py "your_model_name"
```
Such as : 
```
python test_file.py "Karim-Gamal/XLM-Roberta-finetuned-emojis-1-client-toxic-cen-2"
```

## Requirements

> To run the code in these notebooks, you will need to have the following software and packages installed:

* `Python 3`: The programming language used for this project.

* `Jupyter Notebook or JupyterLab`: The environment used for developing and running the project code.

* `PyTorch` Python package: Required for loading and testing the Switch model. You can install it using the command `pip install torch`.

* `Transformers` Python package: Required for loading and testing transformer models. You can install it using the command `pip install transformers`.

* `scikit-learn` Python package: Required for evaluating model performance using metrics such as accuracy and F1-score. You can install it using the command `pip install scikit-learn`.

* Enjoy... :)

> Note: These notebooks assume that you have pre-trained models and testing datasets available. If you do not have these, you will need to train your own models and/or obtain the necessary datasets before running the notebooks.
