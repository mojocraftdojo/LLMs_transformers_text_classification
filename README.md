# **Fine tune Transformers-based Large Language Models for Text Classification NLP applications**


##  Sentiment Analysis using large language models (LLMs)



## Objective

This analysis aims at predicting various sentiments in financial news headlines using pre-trained transformer-based models, which are hosted on Huggingface.
The LLM model selected for this analysis is DistilBERT model that can be used for text classification including Sentiment Analysis. DistilBert is a relative simpler model than BERT or RoBERTa.

DistilBERT stands for "Bidirectional Encoder Representations from Transformers.


## Modeling using DistilBERT:

This analysis focus on modeling using large language models (LLMs). However, previously, I did a separate analysis to apply various machine learning predictive models and Deep learning models on the same dataset to predict sentiments. If you are interested in checking out the side-by-side model performance, got to: https://github.com/mojocraftdojo/NLP_news_sentiment_analysis


After this analysis, I will also update that Jupyter notebook to include the DistilBERT model performance to compare with the below models:

>### **Classic Supervised learning models( Scikit-learn)**
>> #### Model 0: Naive Bayes (baseline)
>> #### Model 1: Random Forest
>> #### Model 2: XGBoost
>> #### Model 3: Support Vector Machine
>> #### Model 4: Logistic Regression

> ### **Deep Learning with NLP text preprocessing (TensorFlow/Keras)**
>> #### Model 5: RNNs (LSTM)
>> #### Model 6: TensorFlow Hub Pretrained Feature Extractor (Transfer Learning use USE)

> ### **large language models (Hugging face)**
>> #### Transformer-based DistilBERT model

