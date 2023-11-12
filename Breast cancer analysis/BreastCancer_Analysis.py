#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


Cancer_data = pd.read_csv("C:/Users/swapn/Downloads/data (1).csv")
Cancer_data.head(5)


# In[3]:


plt.figure(figsize=(8, 4))
plt.hist(Cancer_data['radius_mean'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Radius Mean')
plt.xlabel('Radius Mean')
plt.ylabel('Frequency')
plt.show()


# In[4]:


count = Cancer_data['diagnosis'].value_counts()
plt.figure(figsize=(4, 4))
plt.pie(count, labels=['Benign', 'Malignant'], autopct='%1.1f%%')
plt.title('Distribution of Benign and Malignant Cases')
plt.show()


# In[5]:


features = Cancer_data.drop(['id', 'diagnosis'], axis=1)
plt.figure(figsize=(8, 4))
sns.boxplot(x='variable', y='value', data=pd.melt(features))
plt.title('Feature Distributions by Diagnosis')
plt.xlabel('Feature')
plt.ylabel('Value')
plt.xticks(rotation=90)
plt.show()


# In[6]:


correlation_matrix = Cancer_data.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Features')
plt.show()


# In[14]:


from sklearn.preprocessing import LabelEncoder

# Handle missing values
# The 'Unnamed: 32' column contains NaN values, so we drop it
# Also, the 'id' column is not needed for the model, so we drop it
ndata = Cancer_data.drop(['Unnamed: 32', 'id'], axis=1)

# Encode the 'diagnosis' column
le = LabelEncoder()
ndata['diagnosis'] = le.fit_transform(ndata['diagnosis'])

# Split the data into features (X) and target (y)
X = ndata.drop('diagnosis', axis=1)
y = ndata['diagnosis']

# Display the first few rows of the processed data
X.head(), y.head()


# In[15]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the sizes of the training and testing sets
len(X_train), len(X_test)


# In[55]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the sizes of the training and testing sets
len(X_train), len(X_test)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Train a Support Vector Machine (SVM) model
svm = SVC(random_state=42)
svm.fit(X_train, y_train)

# Train a Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the testing data
svm_preds = svm.predict(X_test)
rf_preds = rf.predict(X_test)

# Calculate the precision of the models
svm_precision = precision_score(y_test, svm_preds)
rf_precision = precision_score(y_test, rf_preds)

svm_precision, rf_precision


# In[59]:


# Creating a new dataset for testing
new_data = pd.DataFrame({
    'radius_mean': [13.54],
    'texture_mean': [14.36],
    'perimeter_mean': [87.46],
    'area_mean': [566.3],
    'smoothness_mean': [0.09779],
    'compactness_mean': [0.08129],
    'concavity_mean': [0.06664],
    'concave points_mean': [0.04781],
    'symmetry_mean': [0.1885],
    'fractal_dimension_mean': [0.05766],
    'radius_se': [0.2699],
    'texture_se': [0.7886],
    'perimeter_se': [2.058],
    'area_se': [23.56],
    'smoothness_se': [0.008462],
    'compactness_se': [0.0146],
    'concavity_se': [0.02387],
    'concave points_se': [0.01315],
    'symmetry_se': [0.0198],
    'fractal_dimension_se': [0.0023],
    'radius_worst': [15.11],
    'texture_worst': [19.26],
    'perimeter_worst': [99.7],
    'area_worst': [711.2],
    'smoothness_worst': [0.144],
    'compactness_worst': [0.1773],
    'concavity_worst': [0.239],
    'concave points_worst': [0.1288],
    'symmetry_worst': [0.2977],
    'fractal_dimension_worst': [0.07259]
})

# Display the new dataset
print("New Dataset for Testing:")
print(new_data)


# In[60]:


# Make predictions on the new data
new_predictions = svm.predict(new_data)

# Print the predictions
print("Predictions for the new data:")
print(new_predictions)


# In[62]:


# Make predictions on the new data
new_predictions = svm.predict(new_data)

# Print the predictions along with labels
for prediction in new_predictions:
    if prediction == 1:
        print("Prediction: Malignant")
    elif prediction == 0:
        print("Prediction: Benign")
    else:
        print("Invalid prediction value")

