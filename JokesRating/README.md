# Jokes rating (Regression)

This project is related to the Joke Rating Challenge Practice "Joke Rating Prediction" (You can look at the challenge here : https://datahack.analyticsvidhya.com/contest/jester-practice-problem/)

## Summary
This practice problem challenges us to predict the ratings for jokes given by the users provided the ratings provided by the same users 
for another set of jokes. This dataset is taken from the famous jester online Joke Recommender system dataset.

## Folder Description

### train_HumourDistilBert.ipynb
In this notebook, I based my approach on an article (https://arxiv.org/pdf/2004.12765v1.pdf) in order to finetune DistilBert on humour and no humour text.
I unfreezed 20% of the distilbert architecture and finetuned with a specific task : Predict if a text is a joke or not,  on a big dataset of 200k texts
(100k are jokes, 100k aren't). To speed up the training, I used the colab TPU. His accuracy is 0.9828 (I did a mistake and forget to add a sigmoid at the end
of the model architecture, so my output was just number between -10 and 5, but I manage to find a good threshold and convert it to probability without
retraining the model). I didn't try to improve the accuracy since I'm only interested on make distilbert "understand the essence of a joke". (I called it HumourDistilBert)

### Jokes_Rating_MatrixF_Keras.ipynb
In this notebook, I tackle the problem as a Collaborative Filtering Task.
My first approach is only to take into account the id of the jokes (I didn't take into account the text) and implement a kind of implicit Matrix
Factorization with Keras. 
My best score was 4.270991 (ranked 37/250). I tried to train the model more longer (adding epoch) but I think it overfitted since his performance on the test set
decreased. I tried also to play with the embedding size, I thought that maybe by capturing more feature related to each user & each joke it will help us but It didn't
My second approach (I haven't implemented yet (I don't have much time now since I have many exams and homeworks to do)) is to take into account the text.
For each joke, I will compute thanks to HumourDistilBert (and I will also try to use Bert Base to compare) for each joke x a similarity vector where the ith element of this vector
is the similarity between the joke x's embedding and the ith joke's embedding. To be continued
The third approach is to use the implementation of vanilla matrix factorization with Keras and develop a Bayesian Personalized Ranking via This Matrix Factorization.


### Jokes_Rating_MatrixF.ipynb
In this notebook, I tried many implementation for collaborative filtering thanks to the incredible library scikit-surprise.
I couldn't test KNN With Means & KNN With ZScore since the running time for testing the model can't be handled by my computer (and by colab)
I tried the Singular Value decomposition (SVD), the SVD++ that takes into account implicit ratings and the CoClustering alogirthm (Implementations are available on 
scikit-surprise). The SVD gaves us the lowest rmse (mean rmse on cross validation : 4.3055), so I decided to finetune the hyperparameters (mean rmse on cross validation : 4.1793).
We were able to get a score of 4.0432 (ranked 7/250)
I will try to use another approach based on the homework assignement of this course https://www.coursera.org/learn/machine-learning/programming/fyhXS/anomaly-detection-and-recommender-systems.

### 

## Results

### sub_keras(1).csv
Using the implicit matrix factorization using Keras : 4.270991

### sub_svd.csv
Using the finetuned SVD : 4.0432

