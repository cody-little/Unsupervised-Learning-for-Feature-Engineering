# Unsupervised-Learning-for-Feature-Engineering
A machine learning project comparing the inclusion of an unsupervised learning label as a feature for the same task in a comparative analysis of results and methods.

## Summary

This machine learning project intends to discuss the usage of an unsupervised learning prediction as a feature in a supervised learning task. To do this I take a data set with various wine charachteristics and a target label of a quality rating and apply a k-means clustering algorithim on the the feature space minus the quality rating. I then use this unsupervised prediction cluster label as a feature in a supervised learning task with six outcome labels. Below is a short bulleted breakdown of what the project entails.

#### Task 1: Predict a six way classification label for wine quality based off available features
#### Task 2: Create an experiment where I compare unsupervised feature label inclusion in task 1

* Get a cluster label on data from a k-means algorithim
* Discuss methods of evaluating the number of clusters
* Perform a supervised learning task in an experimental fashion showing the effect of an unsupervised feature label
  - Perform the same task, on the same test set, with the same features except for one difference
  - The difference is one application of the task includes the unsupervised cluster label as a feature
  - I perform the supervised learning task with a grid search and 5-k fold cross validated decison tree
  - Discuss results and efficacy on this multi-label classification task
  
  
## Sections

* Exploratory Analysis
* Unsupervised Learning and Evaluation of Clusters
* Supervised Learning Task Experiment
* Results

## Exploratory Analysis

The first step of this exploratory data analysis (EDA) is to just look at the data and see what I am working with. I am not a chemist or a sommelier so I know nothing about wine but all of the columns in this data set are numeric which means I can understand their distributions. 

![](https://github.com/cody-little/Unsupervised-Learning-for-Feature-Engineering/blob/master/img/unsupervised%20learning%20label.PNG)

This table gives me a good indication that later on I will have to normalize my numeric features to perform the k-means algorithim. Looking at the min/max and the standard deviation I can clearly see that some variables used as features will be pretty influential. Another thing I see immmediatly is that this isn't a huge data set. With just under 1,600 observations I wil not be able to use a more data hungry algorithim later on.

The next step I take is to look at the correlations among the features and the target labels.

