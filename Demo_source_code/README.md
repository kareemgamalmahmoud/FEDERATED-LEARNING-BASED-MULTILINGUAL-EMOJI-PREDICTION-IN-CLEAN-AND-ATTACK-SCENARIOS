# Transformer Model Demo

> This folder contains a Gradio application for demonstrating the performance of our transformer model on the task of sentiment analysis. 

> The app.py file contains the code for the web application. The web application is available online [here](https://huggingface.co/spaces/Karim-Gamal/Demo-for-Federated-Learning-Based-Multilingual-Emoji-Prediction), and you can also run it locally on your machine using the instructions below.

> ![image](https://user-images.githubusercontent.com/51359449/224560170-23bfc6ab-7f6a-438f-8427-7f9a96c9f0c4.png)

## Design

> The web application has a simple design, as shown in the screenshot above. 

> The user can input some text in the input box and see the sentiment analysis results.

## Prerequisites

> Huggingface account

## Requirements

> To run the web application, you will need to have the following software and packages installed:

* `Python 3`: The programming language used for this project.
* `Gradio`: A Python framework. You can install it using the command `pip install gradio`.


## Getting Started

To run the app, follow these steps:

> 1- Clone the repository and navigate to the root directory.

> 2- Install the required Python libraries using the following command:
```
pip install -r requirements.txt
```

> 3- Obtain your Huggingface reading token by following the steps below.

> 4- Run the app using the following command:
```
python app.py "your_reading_token"
```

## How to obtain your Huggingface reading token

> 1- Sign in to your Huggingface account.

> 2- Click on your profile picture at the top-right corner of the page and select "Settings".

> 3- Scroll down to the "API Token" section and click on "Create New Token".

> 4- Give your token a name and select the "read" scope.

> 5- Click on "Create".

> 6- Copy the token and use it to run the app.

## Acknowledgments

This app was created using the Huggingface API.
