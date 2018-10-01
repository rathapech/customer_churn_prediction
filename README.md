# customer_churn_prediction
Python source code for predicting the customer churn based on Scikit Learn. 

The data is available at https://www.kaggle.com/blastchar/telco-customer-churn. It originally contains 21 columns in which the last one is the label. 
Before it is used to train the model, we need some preprocess techniques such as converting the categorical values to 
numerical by using dummy variables technique, scaling to standard, and dimensionality reduction by using principal component analysis (PCA). 

After the data is preprocessed, it contains up to 30 columns. 
