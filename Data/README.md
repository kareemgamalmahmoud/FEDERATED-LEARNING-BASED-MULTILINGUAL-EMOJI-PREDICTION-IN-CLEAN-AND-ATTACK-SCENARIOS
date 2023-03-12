# Twitter Emojis

This folder contains the code and data for this paper. We have focused on 20 commonly used emojis.

## Data Collection

> The `Crawl_Twitter_Data_using_Twitter_API.ipynb` file contains the code for collecting tweet data using the Twitter API. We used this API to crawl 5000 tweets for each of our 20 emojis for each language.

> However, due to the limited nature of the Twitter API, we also used the `snscrape` Python package to collect more data for the same set of emojis. The code for this step is available in the `Data_scraping.ipynb` file.

## Data Labeling

> Since this is a multiclass classification problem, we needed to convert the emojis into numerical labels. The `Data_labeling.ipynb` file contains the code for this step. We converted the emojis from this format: [ ❤ ,😍 ,😂... ] to this format: [0, 1, 2...].

## Data Cleaning

> The `clean_one_emoji_one_sentace.ipynb` file contains the code for cleaning the collected data. We removed all the emojis that we don't consider in our task and removed the following from the tweets:

> 1- Tweets that have in_reply_to_status_id !=null i.e. comments on someone else's tweets.

> 2- Non-ASCII characters from text.

> 3- Hyperlinks from text.

> 4- Stopwords from text.


## Requirements

* Python 3

* Jupyter Notebook or JupyterLab

* `tweepy` Python package for Twitter API data collection (install using `pip install tweepy`)

* `snscrape` Python package for additional Twitter data scraping (install using `pip install snscrape`)

* `pandas` Python package for data manipulation (install using `pip install pandas`)

* `numpy` Python package for numerical computation (install using `pip install numpy`)

* `scikit-learn` Python package for machine learning (install using `pip install scikit-learn`)

* Enjoy... :)

> Note : You may also need to obtain API keys and tokens from Twitter to use the Twitter API. You can learn more about this process on the Twitter Developer website.
