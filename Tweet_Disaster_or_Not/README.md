# Real or Not

This project is related to the kaggle challenge Real Or Not (You can look at the challenge here : https://www.kaggle.com/c/nlp-getting-started/overview)

## Summary
In this project, we have to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. We have access to a dataset of 10,000 tweets that were hand classified.

## Folder Description

### Disaster_or_not.ipynb
In this notebook, I describe how I built the model. 
First of all, I used a Bag-of-words technique (TF-IDF) + NaiveBayes (Multinomial) Classifier
Secondly, I used word embedding (GloVe) + Neural Network Model (BiLSTM)
This notebook can run on cpu (local).

### Disaster_or_not_colab.ipynb
In this one, I used BERT thanks to the transformers library (from HuggingFace).
This notebook can run on gpu if you have one, or can run on google colab.

## Results

### sample_submissions.csv
Using TF-IDF + NaiveBayes, The kaggle score is 0.79160

### sample_submissions_glove.csv
Using Glove + BiLSTM, the kaggle score is 0.79681

### sample_submissions_bert_v2.csv
Using Bert, the kaggle score is 0.83328

