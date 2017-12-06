# Project Report: Story Ending Classification
## Team Members
Jaishree Ranganathan

Maryam Tavakoli 

## Introduction
'Story Ending Classification' is a commonsense learning framework for evaluating story understanding. This Framework requires a system to classify the correct ending to a four-sentence story. Story here means casually (logically) linked set of events.


Example story [1]

Bill thought he was a great basketball player. He challenged Sam to a friendly game. Sam agreed.Sam started to practice really hard. Eventually Sam beat Bill by 40 points.

Bill challenges Sam ----Enables----> Sam agrees to play ----Before---->Sam practices ----before---->Sam beats Bill

## Motivation
Human interactions have different forms for transfering information, and one of the oldest one is story-form. People use story structure to talk about their experience, ideas, observation or even believes. The information that flows through a story is more than those stored in isolated sentences. There are accumulative information that comes out of the whole story. This huge amount of information hidden in narrative and story formed data are motivation of NLP next generation narrative-base technology [7].


Understanding stories in NLP has different aspect of complexity. Context, time frame, commmonsense knowledge, and characters are some of the important ones. In this project we used a framework that while carrying these complexities, it removes the burden of the text length by providing short stories. Also, to make the task just language understanding (and not language generation) it defines the task as sentence selection. By doing this project, we are taking the introductory step toward story-understanding. Of course, due to lack of time, we chose the simplest methods for analyzing the data, however, it can lead us to the better methodologies in the following steps of the research.

## Dataset
The data set used for the project is ROCStories, Cloze Test Competition dataset [1]. This dataset contains around 50,000 (train set) common sense stories, which have high-quality and a 4-sentence structure. Each record in the training data contains storyid, title, sentence1, sentence2, sentence3, sentence4, sentence5. There are also validation and test set which contains same fields as that of the training set, additionally includes the sentence6, which is wrong ending for each story.

## Data Preparation
The training dataset, did not contain the false entries, in this case the sentence 6 which is the wrong ending for the story. The sample data is shown in table below.

| StoryID        | Title           | Sentence1       | Sentence2       | Sentence3       | Sentence4       | Sentence5  |
|----------------|-----------------|-----------------|-----------------|-----------------|-----------------|------------|
| 8bbe6d11-1e2e-413c-bf81-eaea05f4f1bd|David Drops the Weight|David noticed he had put on a lot of weight recently.|He examined his habits to try and figure out the reason.|He realized he'd been eating too much fast food lately.|He stopped going to burger places and started a vegetarian diet.|After a few weeks, he started to feel much better.|            

We prepared the data set to create the sentence6 for each story with the random approach. In this approach each story ending (sentence 6) is randomly selected from the ending part of a different story in the training set. After that, the sentence 5 & 6 of each story are shuffled to have randomly T/F location in each place, and labeled all the records based on the location of the correct ending. There are other approaches for generating story ending like nearest ending and RNN. According to [2] we chose to use random approach based on the evaluation results. The resulting train data after appending the sentence 6 is as in below table.

| StoryID        | Title           | Sentence1       | Sentence2       | Sentence3       | Sentence4       | Sentence5  | Sentence6 |
|----------------|-----------------|-----------------|-----------------|-----------------|-----------------|------------|-----------|
| 8bbe6d11-1e2e-413c-bf81-eaea05f4f1bd|David Drops the Weight|David noticed he had put on a lot of weight recently.|He examined his habits to try and figure out the reason.|He realized he'd been eating too much fast food lately.|He stopped going to burger places and started a vegetarian diet.|After a few weeks, he started to feel much better.|Hers was picked.|

This generated the training data with 100,000 instances as each story will have a right ending and wrong ending entry in the train model.

## Methods
### Word2vec
Word2vec is one of the word embedding algorithms that are designed by Google using shallow neural networks. Like other word embedding methods, this model maps the word from a sparse discrete space to a dense continuous one. 
The first method for implementation was using gensim word2vec library along with Google News 300-Feature pretrained model. Using this pretrained model, we obtained a feature map for each of train, test-val, and test-test. Then running a PCA, on top of that reduced the feature size to 50 features for each story. 
The second method was using MLlib Word2vec library, which let us train a word2vec model using our training dataset. Having that model, we were able to train a word2vec model on spark, but we couldn't use the model in mapper based on an incompatibility of pyspark in carrying JavaRDD to mapper [5]. 

### Sentiment Last
The assumption behind this method was that the last sentence of story has similar sentiment and emotion as the story-ending and the idea came from the original paper [1]. The dataset that we used for sentiment analysis was NRC Emotion Lexicon [3], which is a crowd-source dataset. Each of the last sentence and the candidates for the end of the story are sent to a function which computes the average of positive and negative sentiment of the sentence words, as well as following emotions:

'anger','joy','fear','trust','anticipation','surprise','disgust'


Then for each of the two candidates, the feature vector is computed as the difference between sentiment of the last sentence and that candidate. This way we can find which candidate has the closest meaning to the story last sentence.

### Logistic Regression
In this story ending classification framework we used the classification method called Logistic Regression. Basically Logistic Regression is the classification approach that helps analyse the association between categorical dependant variable and set of independent variables. Here the dependant variable is categorical.

In our dataset we have two class labels as 0 0r 1 indicating whether the last sentence is right story ending (1) or wrong story ending (0). We have list of independent variables, which are the sentences 1 to 4. The independent variables are represented in different formats based on the methods Word2Vec and Sentiment Last. The Gradient Ascent approach is implemented as part of this project. The derivation for Gradient Ascent is in [6].

Gradient Ascent Logistic Regression is implemented in pyspark.

## Work Division - Team Members
Jaishree Ranganathan - Logistic Regression, ROC, Accuracy Calculation, MLlib implementation (For comparison)

Maryam Tavakoli - Data Preparation, Negative Sampling, Word2Vec, Sentiment Last method, PCA-transformation, Logistic Regression & AUC using sklearn (For data evaluation)

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
In order to do the evaluation, we trained the model using three sets of features: Word2Vec, 50 Features extracted from word2vec using PCA, and finally sentiment last. Also, based on the fact that we generated the training data negative samples randomly, we tried to train by validation data in some of the cases, and interestingly, some of them had better results. We evaluated the method by primarily AUC on local machine and then ROC metric on spark (we brought only the second one here).


As most of the results may suggest, while these methods are suitable as an starting point of the prediction, we need more complicated feature extraction and learning method to be able to tackle the complexity of story understanding, specifically in this framework. The reason behind that is that semantic of story is far beyond even understanding the sentences themselves. Even though these stories were not long, still analyzing the context was so important to get an acceptable result in predicting the right ending. Yet, for some of more semantic based methods like word2vec, you can see a higher result. 

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

##### Input Data

Train Data          : word2vec600_test_val_vector.csv

Test Data           : word2vec600_test_test_vector.csv

##### Area Under ROC Curve: 0.60196979134226103
##### ('Area Under ROC MLlib:', 0.6034430330757854)

## Installation
#### Dependencies/External Tools
Three different parts of the project needs three different dependencies:

part 1: python 3.6.0, Gensim, numpy, NLTK stopwords, sci-kit learn

part 2 and 3: python 2.6.6, Numpy 1.4.1, Apache Spark

## Final Product and Aspects Achieved

#### Final Product

Read the story data from ROCStories corpus and predict the outcome of them. 

#### Definitely accomplished

Logistic regression on cloud using Spark
ROC Evaluation for Logistic Regression
Accuracy calculated using predicted and actual labels

Translating the sentences into vectors using word2vec pretrained model by averaging the word-vectors of each sentence on local.

Sentiment-Last method is implemented on local and cloud, and compare with the word2vec method.

We also had to do Data Preparation, Negative Sampling, PCA-transformation, and some local data evaluation using sklearn library
#### Likely and Ideally Accomplished

We implemented PCA for feature selection.
Based on the results with the current methods that we implemented we identified future works implementation and methodology.

#### Observations

1. The results obtained with the dataset prepared using word2Vec PCA and Sentiment Models are in the range of 50% to 60%. We see that these models do not replicate the deep semantic relationship between the sentences.

2. We observe from the evaluation results that the training with validation data and test with test data yields higher accuracy than the train data and test data. This might be because in the train data we generated the random sentence for wrong ending. But rather the validation data was provided with the semantically related wrong ending sentence. So if the train data wrong ending was generated with appropriate sentence as it was in validation data, we believe better results would have been achieved.

3. From the evaluation it is also observed that the Principal Component Analysis data with Word2Vec model yielded little higher accuracy than the sentiment last model dataset. This is because Word2Vec accounts for higher semantic similarity than the sentiment last model.

4. In logistic Regression implementation used the divide by number of instances after calculating the summation, which helped avoid the overflow error.

## Future works
#### Implementation
In implementation part, we can improve the word2vec by training a model on a relavant dataset. Also, due to the problem we faced on spark, we need to redo that on hadoop streaming (that we couldn't do based on the timing). We also can try other sentiment analysis libraries like Stanford and compare the results with our implementation. Another improvement is moving all the code-parts to the spark/hadoop-streaming and implement some of the library methods by ourselves (like PCA) to make the performance higher. However, with these modifications we do not expect huge improvement in overall results. 

#### Methodology
As we mentioned in the introduction, this work was just a beginning  step towards handling story processing. Following this step, and based on the result, we are sure that we need more context-dependent methods. We need methods that can carry the essense of all previous sentences and compare them with the current one. For that we think both lexicon-based and deep-learning based methods can come helpful for this task. We need to check both relatedness of the subject, event and verb of the candidate with the whole story, and the expctation of commonsense from the events in the story. The former depends more on lexicon analysis, part-of-speech, entity and event-extraction, while for the latter deep-learning methods, like rnn [2], trained on large corpus might be more helpful.

## References
[1] Mostafazadeh, N., Chambers, N., He, X., Parikh, D., Batra, D., Vanderwende, L., ... & Allen, J. (2016). A corpus and evaluation framework for deeper understanding of commonsense stories. arXiv preprint arXiv:1604.01696.

[2] Roemmele, M., Kobayashi, S., Inoue, N., & Gordon, A. M. (2017). An rnn-based binary classifier for the story cloze test. LSDSem 2017, 74.

[3] Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word–emotion association lexicon. Computational Intelligence, 29(3), 436-465.

[4] Google Word2Vec Project, https://code.google.com/archive/p/word2vec/

[5] https://stackoverflow.com/questions/41046843/how-to-load-a-word2vec-model-and-call-its-function-into-the-mapper/41190031#41190031

[6] http://cs229.stanford.edu/notes/cs229-notes1.pdf

[7] Cambria, E., & White, B. (2014). Jumping NLP curves: A review of natural language processing research. IEEE Computational intelligence magazine, 9(2), 48-57.
