# **Apply transformer-based LLM models for text classfication applications**


##  Sentiment Analysis using large language models



## Objective

This analysis aims at predicting various sentiments in financial news headlines using pre-trained transformer-based models, which are hosted on Huggingface.
The LLM model selected for this analysis is DistilBERT model that can be used for text classification including Sentiment Analysis. DistilBert is a relative simpler model than BERT or RoBERTa.

DistilBERT stands for "Bidirectional Encoder Representations from Transformers)


## Modeling using LLMs

This analysis focus on large language models (LLMs). 

However, previously, I've done a seperate analysis to apply various classic machine learning predictive models and Deep learning models to the same dataset to predict sentiment. That analysis can be found here: https://github.com/mojocraftdojo/NLP_news_sentiment_analysis


Those models can be used as a baseline to compare with performance of LLMs.

>### **Classic Supervised learning models( Scikit-learn)**
>> #### Model 0: Naive Bayes (baseline)
>> #### Model 1: Random Forest
>> #### Model 2: XGBoost
>> #### Model 3: Support Vector Machine
>> #### Model 4: Logistic Regression

>### **Deep Learning with NLP text preprocessing (TensorFlow/Keras)**
>>#### Model 5: RNNs (LSTM)
>>#### Model 6: TensorFlow Hub Pretrained Feature Extractor (Transfer Learning use USE)

