# Project Report: Story Ending Classification
## Team Members
Jaishree Ranganathan

Maryam Tavakoli 

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
Word2vec is one of the word embedding algorithms that are designed by Google using shallow neural networks. Like other word embeddings, this model maps the word from a sparse discrete space to a dense continuous one. 
The first method for implementation was using gensim word2vec library along with Google News 300-Feature pretrained model. Using this pretrained model, we obtained a feature map for each of train, test-val, and test-test. Then running a PCA, on top of that reduced the feature size to 50 features for each story. 
The second method was using MLlib Word2vec library, which let us train a word2vec model using our training dataset. Having that model, we were able to run word2vec on spark, as well. 

### Sentiment Last
The assumption behind this method was that the last sentence of story has similar sentiment and emotion as the story-ending and the idea came from the original paper [1]. The dataset that we used for sentiment analysis was NRC Emotion Lexicon [3], which is a crowd-source dataset. Each of the last sentence and the candidates for the end of the story are sent to a function which computes the average of positiveness and negative of the sentence words, as well as following emotions: 'anger','joy','fear','trust','anticipation','surprise','disgust'
Then for each of two candidates, the feature vector is computed as the difference between sentiment of the last sentence and that candidate. 

### Logistic Regression
In this story ending classification framework we used the classification method called Logistic Regression. Basically Logistic Regression is the classification approach that helps analyse the association between categorical dependant variable and set of independent variables. Here the dependant variable is categorical.

In our dataset we have two class labels as 0 0r 1 indicating whether the last sentence is right story ending (1) or wrong story ending (0). We have list of independent variables, which are the sentences 1 to 4. The independent variables are represented in different formats based on the methods Word2Vec and Sentiment Last. The Gradient Ascent approach is implemented as part of thsi project. The derivation for Gradient Ascent [6] equation given below.

Gradient Ascent: 
![alt text](https://github.com/jaishreerang/CloudProjectReport/image1.png "GradientAscent Equation")
Logistic Regression is implemented in pyspark.

### Working and Failed Methods

## Work Division - Team Members
Jaishree Ranganathan - Logistic Regression, ROC, Accuracy Calculation, MLlib implementation (For comparison)

Maryam Tavakoli - Word2Vec, Sentiment Last
## Demo

| Sentence1       | Sentence2       | Sentence3       | Sentence4       | Sentence5  | Sentence6 |Correct Ending|
|-----------------|-----------------|-----------------|-----------------|------------|-----------|--------------|
|David noticed he had put on a lot of weight recently.|He examined his habits to try and figure out the reason.|He realized he'd been eating too much fast food lately.|He stopped going to burger places and started a vegetarian diet.|After a few weeks, he started to feel much better.|Hers was picked.|sentence 5|

| Sentence1       | Sentence2       | Sentence3       | Sentence4       | Sentence5  | Label |
|-----------------|-----------------|-----------------|-----------------|------------|-------|
|David noticed he had put on a lot of weight recently.|He examined his habits to try and figure out the reason.|He realized he'd been eating too much fast food lately.|He stopped going to burger places and started a vegetarian diet.|After a few weeks, he started to feel much better.|1(Correct Ending)|

| Sentence1       | Sentence2       | Sentence3       | Sentence4       | Sentence6  | Label |
|-----------------|-----------------|-----------------|-----------------|------------|-------|
|David noticed he had put on a lot of weight recently.|He examined his habits to try and figure out the reason.|He realized he'd been eating too much fast food lately.|He stopped going to burger places and started a vegetarian diet.|Hers was picked.|0(Correct Ending)|

## Evaluation
#### Experiment1: Logistic Regression on Word2Vec Data (Using our Model and using MLlib)

##### Input Data

Train Data          : pca_50_train_vector.csv

Test Data           : pca_50_test_val_vector

##### Area Under ROC Curve: 0.49785703650253704

##### ('Area Under ROC MLlib:', 0.49839277737690274)

##### Input Data

Train Data          : pca_50_train_vector.csv

Test Data           : pca_50_test_test_vector

##### Area Under ROC Curve: 0.48529955872248087

##### ('Area Under ROC MLlib:', 0.4850328552654134)

##### Input Data

Train Data          : pca_50_test_val.csv

Test Data           : pca_50_test_test_vector

##### Area Under ROC Curve: 0.59100088537844109

##### ('Area Under ROC MLlib:', 0.5979624669983753)

#### Experiment2: Logistic Regression on Sentiment Last Data (Using our Model and using MLlib)

##### Input Data

Train Data          : sentiment_train.csv

Test Data           : sentiment_test_test.csv

##### Area Under ROC Curve: 0.46088903285722715

##### ('Area Under ROC MLlib:', 0.47264348949858265)

##### Input Data

Train Data          : sentiment_train.csv

Test Data           : sentiment_test_val.csv

##### Area Under ROC Curve: 0.45532441237388588

##### ('Area Under ROC MLlib:', 0.47041526743849271)

##### Input Data

Train Data          : sentiment_test_val.csv

Test Data           : sentiment_test_test.csv

##### Area Under ROC Curve: 0.53778644059836089

##### ('Area Under ROC MLlib:', 0.53945034691300764)

## Execution Instruction
Logistic Regression:

1)To execute in standalone program (LRGA_ROC.py and LRGA_calcAccuracy.py)

  spark-submit filename.py input_train_data number_of_iterations input_test_data
  
  Note: Number of iterations is 100 for this dataset

2) Resulting evaluation of Area under ROC curve will be displayed in console.

3) To execute in standalone program (LGMLlib.py)

spark-submit filename.py input_train_data input_test_data

## Source Code
Source code is submitted as a zip file in the course project submission.
1. LRGA_ROC.py - Used BinaryClassification Metric ROC
2. LRGA_calcAccuracy.py - Calculated accuracy based on actual and predicted labels
3. LGMLlib.py - Used MLlib to compare the results

## Installation
#### Dependencies/External Tools
python 2.6.6

Numpy 1.4.1

Apache Spark

python 3.6.0

Apache Spark

Gensim

NLTK stopwords

## Final Product and Aspects Achieved

#### Final Product

Read the story data from ROCStories corpus and predict the outcome of them. 

#### Definitely accomplished

Logistic regression on cloud using Spark
ROC Evaluation for Logistic Regression
Accuracy calculated using predicted and actual labels

Translating the sentences into vectors using word2vec pretrained model by averaging the word-vectors of each sentence on cloud. We need to copy the pre-trained model on all the nodes for that.

Sentiment-Last method, and compare with the word2vec method.

#### Observations

1. The results obtained with the dataset prepared using word2Vec and Sentiment Models are in the range of 50% to 60%. We see that these models do not replicate the deep semantic relationship between the sentences.

2. We observe from the evaluation results that the training with validation data and test with test data yields higher accuracy than the train data and test data. This might be because in the train data we generated the random sentence for wrong ending. But rather the validation data was provided with the semantically related wrong ending sentence. So if the train data wrong ending was generated with appropriate sentence as it was in validation data, we believe better results would have been achieved.

3. From the evaluation it is also observed that the Principal Component Analysis data with Word2Vec model yielded little higher accuracy than the sentiment last model dataset. This is because Word2Vec accounts for higher semantic similarity than the sentiment last model.

4. In logistic Regression implementation used the divide by number of instances after calculating the summation, which helped avoid the overflow error.
 

## References
[1] Mostafazadeh, N., Chambers, N., He, X., Parikh, D., Batra, D., Vanderwende, L., ... & Allen, J. (2016). A corpus and evaluation framework for deeper understanding of commonsense stories. arXiv preprint arXiv:1604.01696.

[2] Roemmele, M., Kobayashi, S., Inoue, N., & Gordon, A. M. (2017). An rnn-based binary classifier for the story cloze test. LSDSem 2017, 74.

[3] Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word–emotion association lexicon. Computational Intelligence, 29(3), 436-465.

[4] Google Word2Vec Project, https://code.google.com/archive/p/word2vec/

[5] https://stackoverflow.com/questions/41046843/how-to-load-a-word2vec-model-and-call-its-function-into-the-mapper/41190031#41190031

[6] http://cs229.stanford.edu/notes/cs229-notes1.pdf
