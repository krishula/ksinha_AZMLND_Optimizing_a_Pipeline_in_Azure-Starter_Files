# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

This dataset contains bank marketing campaigns data based on the phone calls to potential clients. The target was to convince the potential clients to make a term deposit at the bank. In this project, we seek to predict whether the potential client would accept to make a term deposit at the bank or not.

The best performing model was Scikit-learn pipeline, which uses logistic regression with hyperdrive to find the best hyperparameters.
1. Accuracy for Scikit-learn: 0.9169
2. Accuracy for AutoML pipeline (VotingEnsemble): 0.9159

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

Dataset : https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv . 
Data Cleaning and Prep: Fucntion clean_data is used to clean the data of missing values; a data dict is added to convert the categorical fields to numeric; then we one hot encode the data. After that we split the data into 85:15 ratio using the train_test_split function. 
Algorithm: Logistic Regression. 
Hyperparameters with their Search Spaces:

1. C: The inverse of the reqularization strength. '--C' : choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
2. max_iter: Maximum number of iterations to converge.  '--max_iter': choice(50,100,300)

### Types of hyperparameters: (According to Microsoft docs)

A. Discrete hyperparameters:
The following advanced discrete hyperparameters can be specified using a distribution:
  - quniform(low, high, q) - Returns a value like round(uniform(low, high) / q) * q
  - qloguniform(low, high, q) - Returns a value like round(exp(uniform(low, high)) / q) * q
  - qnormal(mu, sigma, q) - Returns a value like round(normal(mu, sigma) / q) * q
  - qlognormal(mu, sigma, q) - Returns a value like round(exp(normal(mu, sigma)) / q) * q
  
B. Continuous hyperparameters:
The following continuous hyperparameters are specified as a distribution over a continuous range of values:
  - uniform(low, high) - Returns a value uniformly distributed between low and high
  - loguniform(low, high) - Returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is uniformly distributed
  - normal(mu, sigma) - Returns a real value that's normally distributed with mean mu and standard deviation sigma
  - lognormal(mu, sigma) - Returns a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the return value is normally distributed

Configuration that can defines a HyperDrive run:

  - hyperparameter_sampling: It referes to the hyperparameter sampling space.
  - primary_metric_name: It's the name of the primary metric reported by the experiment runs.
  - primary_metric_goal: Either takes maximize or minimize. It helps determine if the primary metric has to be minimized/maximized in the experiment runs' evaluation.
  - max_total_runs: It's the maximum number of runs you want it to run. There may be fewer runs when the sample space is smaller than this value.
  - max_concurrent_runs: It's maximum number of runs that can run concurrently.
  - max_duration_minutes: It's the maximum duration of the run. The run is cancelled once this time exceeds.
  - policy: It's the early termination policy to use. Default, no early termination policy will be used.
  - estimator: It's the estimator that will be called with sampled hyper parameters from the chosen dataset.
  - run_config: It's the object for setting up configuration for script/notebook runs.
  - pipeline: It's the pipeline object for setting up configuration for pipeline runs. The pipeline object will be called with the sample hyperparameters to submit pipeline runs. 

Note: We need to specify only one of the following parameters: estimator, run_config, or pipeline.
  
**What are the benefits of the parameter sampler you chose?**

RandomParameterSampling is used here instead of GridParameterSampling since, the hyperparameters are randomly selected from the search space vs GridParameterSampling where all the possible values from the search space are used, and it supports early termination of low-performance runs.

**What are the benefits of the early stopping policy you chose?**

BanditPolicy is used here which is an "aggressive" early stopping policy. It cuts more runs than a conservative policy like the MedianStoppingPolicy, hence saving the computational time significantly.
Configuration Parameters:-

1. slack_factor/slack_amount : (factor)The slack allowed with respect to the best performing training run.(amount) Specifies the allowable slack as an absolute amount, instead of a ratio. Set to 0.1.

2. evaluation_interval : (optional) The frequency for applying the policy. Set to 2.

3. delay_evaluation : (optional) Delays the first policy evaluation for a specified number of intervals.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

In AutoML many pipelines are produced simultaneously that run different algorithms and parameters in an automated way. Parameters used to setup AutoML train:

experiment_timeout_minutes = 30

task ='classification' : We'll use "classification" since we seek to predict whether or not the potential client would accept to make a term deposit with the bank.

compute_target : The compute target with specific vm_size and max_nodes.

training_data : The data on which the algorithm will be trained.

label_column_name : The name of the column that contains the labels of the train data.

n_cross_validations=3 : It is how many cross validations to perform when user validation data is not specified.

primary_metric = 'accuracy' : The metric that Automated Machine Learning will optimize for model selection. We have set the 'accuracy'.

enable_early_stopping = True : Whether to enable early termination if the score is not improving in the short term.

Let's talk about the learned parameters of Auto ML in brief:

  - enable_dnn: It is a flag to enable neural networks for forecasting and natural language processing. 
  - enable_feature_sweeping: It enables or disables feature sweeping. 
  - feature_sweeping_config: It is the config used for feature sweeping.
  - is_onnx_compatible: It works in onnx compatible mode.
  - force_text_dnn: It is a flag to force add neural networks for natural language processing in feature sweeping.
  
Although these parameters turned out to be set at "None" for our project, they are still very important in Auto ML and are worth knowing about.

## Pipeline Comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?

Although there isn't much a difference in Accuracy between the two models, AutoML helps us more by showing the importance of each feature for prediction and also shows some useful metric outputs like:
  - weighted_accuracy
  - f1_score_weighted
  - precision_score_macro
 as show below:
 
 ![alt text](https://github.com/krishula/ksinha_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Picture2-Project1.png)
 


## Future Work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

  - Use a different less aggressive stopping policy
  - Use Gridsampling



