# Dropout Method: Neural Networks Overfitting Prevention

In this project, we will investigate the effects of varying dropout rates on the MNIST dataset, specifically focusing on how different dropout rates impact error rates.

## Purpose of the Project

The primary goal is to explore the effects of varying the tunable hyperparameter 'p' (the probability of retaining a unit in the network) and the number of hidden layers, 'n'. By fixing the product of `p` and `n`, we can observe that the error magnitude for smaller values of `p` is reduced (Fig. 9a) compared to when the number of hidden layers is constant (Fig. 9b).

## Problem Setting

With limited training data, many complex relationships between inputs and outputs can result from sampling noise. These relationships may exist in the training set but not in the real test data, even if drawn from the same distribution. This complication leads to overfitting. Dropout is one method to help prevent overfitting. The input for this study is a dataset of handwritten digits, and the output, after applying dropout, are different values describing the outcome of using the dropout method. Overall, less error is achieved after applying dropout.

## Data Sources

A real-world application of this concept could be in search engines like Google. For instance, someone searching for a movie title might only be looking for images because they are visual learners. Dropping out the textual parts or brief explanations could help focus on the image features. The data for this project is sourced from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/). Each image is a 28x28 pixel representation of a digit, and the labels correspond to the digit values.

## Algorithm

The goal in reproducing the figures is to test and train the data, and calculate the classification error for each probability `p` (probability of retaining a unit in the network). The objective is to show that as `p` increases, the error decreases, validating the implementation. The process involves tuning this hyperparameter to achieve the desired outcome.

The implementation follows these steps:

1. **Architecture**: Use a neural network architecture of 784-2048-2048-2048-10, where 784 is the input layer, 2048 are the hidden layers, and 10 is the output layer.
2. **Data Looping**: Loop through all the training and testing data.
3. **Fixed Product**: Keep `n` fixed and adjust `pn` to be constant.
4. **Data Collection**: Gather and write the data into a CSV file. This file will contain all the necessary data to produce the figures.
5. **Analysis**: Learn how the dropout rate benefits the overall error in a neural network.

By completing this project, we better understood the impact of dropout rates on neural network performance, specifically regarding error reduction. We also answered critical questions related to Memory Equivalent Capacity (MEC), Generalization, and the nature of the dataset as answered in this ![https://github.com/Apratim08/Dropout-Neural-Networks-Analysis/blob/main/CS294_Final_Project.pdf](pdf report)
