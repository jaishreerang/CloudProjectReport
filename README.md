# Project Report: Story Ending Prediction
## Introduction
'Story Ending Prediction' is a commonsense learning framework for evaluating story understanding. This Framework requires a system to classify the correct ending to a four-sentence story. Story here means casually (logically) linked set of events.


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
### Working and Failed Methods
## Demo
## Evaluation
## Execution Instruction
## Source Code
## References
[1] Mostafazadeh, N., Chambers, N., He, X., Parikh, D., Batra, D., Vanderwende, L., ... & Allen, J. (2016). A corpus and evaluation framework for deeper understanding of commonsense stories. arXiv preprint arXiv:1604.01696.
[2] Roemmele, M., Kobayashi, S., Inoue, N., & Gordon, A. M. (2017). An rnn-based binary classifier for the story cloze test. LSDSem 2017, 74.
