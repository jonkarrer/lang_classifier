# NLP Classifier Introduction

U.S Patent classifier challenge

[Textbook](https://colab.research.google.com/github/fastai/fastbook/blob/master/10_nlp.ipynb)
[Notebook](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners)

## Overview

1. Collect Data
   First, ensure you have all your data collected. This might be from CSV files, databases, or real-time data streams. For house price prediction, this data might include features like area, number of bedrooms, number of bathrooms, age of the house, etc.

2. Load Data into a DataFrame
   Use Pandas to load and organize your data into a DataFrame. Pandas is highly efficient for data manipulation and makes it easy to handle missing values, merge data, and encode categorical variables.

3. Preprocess Data
   Ensure your data is clean and ready for modeling:

4. Convert to Suitable Data Format for Modeling
   If you're using a machine learning library like scikit-learn, you might directly use the array returned by ColumnTransformer. For deep learning models in TensorFlow or PyTorch, you might need to convert this data into tensors.

5. Split Data into Training and Testing Sets

6. Train Model or Fine Tune
   Now, your data is ready to be fed into a machine learning model.

By following these steps, you'll have a robust process for converting raw house features into a structured data object suitable for feeding into machine learning models, ensuring consistency and accuracy in your predictions.

### General Steps to NLP creation

1. Tokenization:: Convert the text into a list of words (or characters, or substrings, depending on the granularity of your model)

2. Numericalization:: Make a list of all of the unique words that appear (the vocab), and convert each word into a number, by looking up its index in the vocab

3. Language model data loader creation:: Creating a dependent variable that is offset from the independent variable by one token. Shuffle the training data in such a way that the dependent and independent variables maintain their structure as required.

4. Language model creation:: We need a special kind of model that does something we haven't seen before: handles input lists which could be arbitrarily big or small. There are a number of ways to do this; in this chapter we will be using a recurrent neural network (RNN).
