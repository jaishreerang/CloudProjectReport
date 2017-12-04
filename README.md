# Project Report: Story Ending Classification
## Introduction
'Story Ending Classification' is a commonsense learning framework for evaluating story understanding. This Framework requires a system to classify the correct ending to a four-sentence story. Story here means casually (logically) linked set of events.


Example story [1]

Bill thought he was a great basketball player. He challenged Sam to a friendly game. Sam agreed.Sam started to practice really hard. Eventually Sam beat Bill by 40 points.

Bill challenges Sam ----Enables----> Sam agrees to play ----Before---->Sam practices ----before---->Sam beats Bill

## Dataset
The data set used for the project is ROCStories, Cloze Test Competition dataset [1]. This dataset contains around 50,000 (train set) common sense stories, which have high-quality and a 4-sentence structure. Each record in the training data contains storyid, title, sentence1, sentence2, sentence3, sentence4, sentence5. There are also validation and test set which contains same fields as that of the training set, additinally includes the sentence6 which is wrong ending for each story.

## Data Collection

## Data Preparation
The training dataset, did not contain the false entries, in this case the sentence 6 which is the wrong ending for the story. The sample data is shown in table below.

| StoryID        | Title           | Sentence1       | Sentence2       | Sentence3       | Sentence4       | Sentence5  |
|----------------|-----------------|-----------------|-----------------|-----------------|-----------------|------------|
| 8bbe6d11-1e2e-413c-bf81-eaea05f4f1bd|David Drops the Weight|David noticed he had put on a lot of weight recently.|He examined his habits to try and figure out the reason.|He realized he'd been eating too much fast food lately.|He stopped going to burger places and started a vegetarian diet.|After a few weeks, he started to feel much better.|            

We prepared the data set to create the sentence6 for each story with the random approach. In this approach each story ending (sentence 6) is randomly selected entry from a different story in the training set. There are other approaches for generating story ending like nearest ending and RNN. According to [2] we chose to use random approach based on the evaluation results. The resulting train data after appending the sentence 6 is as in below table.

| StoryID        | Title           | Sentence1       | Sentence2       | Sentence3       | Sentence4       | Sentence5  | Sentence6 |
|----------------|-----------------|-----------------|-----------------|-----------------|-----------------|------------|-----------|
| 8bbe6d11-1e2e-413c-bf81-eaea05f4f1bd|David Drops the Weight|David noticed he had put on a lot of weight recently.|He examined his habits to try and figure out the reason.|He realized he'd been eating too much fast food lately.|He stopped going to burger places and started a vegetarian diet.|After a few weeks, he started to feel much better.|Hers was picked.|

## Methods
### Word2vec
### Sentiment Last
### Logistic Regression
In this story ending classification framework we used the classification method called Logistic Regression. Basically Logistic Regression is the classification approach that helps analyse the association between categorical dependant variable and set of independent variables. Here the dependant variable is categorical or quantitative.

In our dataset we have two class labels as 0 0r 1 indicating whether the last sentence is right story ending (1) or wrong story ending (0). We have list of independent variables, which are the sentences 1 to 4. The independent variables are represented in different formats based on the methods Word2Vec and Sentiment Last. The Gradient Ascent approach is implemented as part of thsi project. The derivation for Gradient Ascent is given below.

Gradient Ascent: 
![alt text](https://github.com/jaishreerang/CloudProjectReport/image1.png "GradientAscent Equation")
Logistic Regression is implemented in pyspark.
### Working and Failed Methods
## Demo

## Evaluation
Experiment1: Logistic Regression on Word2Vec Data

Input Data

Train Data          : pca_50_train_vector.csv

Test Data           : pca_50_test_test_vector

Area Under ROC Curve:

Input Data

Train Data          : pca_50_train_vector.csv

Test Data           : pca_50_test_val_vector

Area Under ROC Curve:

Input Data

Train Data          : pca_50_test_val.csv

Test Data           : pca_50_test_test_vector

Area Under ROC Curve:

Experiment2: Logistic Regression on Sentiment Last Data

Input Data

Train Data          : sentiment_train.csv

Test Data           : sentiment_test_test.csv

Area Under ROC Curve:

Input Data

Train Data          : sentiment_train.csv

Test Data           : sentiment_test_val.csv

Area Under ROC Curve:

Input Data

Train Data          : sentiment_test_val.csv

Test Data           : sentiment_test_test.csv

Area Under ROC Curve:

## Execution Instruction
Logistic Regression:

1)To execute in standalone program

  spark-submit <filename.py> <input train data> <iterations> <input test data>
  
2) Resulting evaluation of Area under ROC curve will be displayed in console.

## Source Code
## References
[1] Mostafazadeh, N., Chambers, N., He, X., Parikh, D., Batra, D., Vanderwende, L., ... & Allen, J. (2016). A corpus and evaluation framework for deeper understanding of commonsense stories. arXiv preprint arXiv:1604.01696.
[2] Roemmele, M., Kobayashi, S., Inoue, N., & Gordon, A. M. (2017). An rnn-based binary classifier for the story cloze test. LSDSem 2017, 74.
