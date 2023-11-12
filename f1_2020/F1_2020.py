#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# Load the data
f1_data = pd.read_csv('formula1_2020season_raceResults.csv')

# Display the first few rows of the dataframe
f1_data.head()


# In[7]:


def convert_to_seconds(time_str):
    if 'Retired' in time_str or ':' not in time_str:
        return np.nan
    else:
        time_parts = time_str.split(':')
        if len(time_parts) == 3:
            # Format is 'hh:mm:ss.sss'
            hours, minutes, seconds = time_parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            # Format is 'mm:ss'
            minutes, seconds = time_parts
            return int(minutes) * 60 + int(seconds)

# Convert 'Total Time/Gap/Retirement' to seconds for easier analysis
f1_data['Total Time (s)'] = f1_data['Total Time/Gap/Retirement'].apply(convert_to_seconds)

# Calculate summary statistics
summary_stats = f1_data[['Laps', 'Points', 'Total Time (s)']].describe()
summary_stats


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.title('Points Scored Per Race by Drivers')
sns.boxplot(x='Driver', y='Points', data=f1_data)
plt.xticks(rotation=90)
plt.show()


# In[10]:


plt.figure(figsize=(10, 6))
plt.title('Points Scored Per Race by Teams')
sns.boxplot(x='Team', y='Points', data=f1_data)
plt.xticks(rotation=90)
plt.show()


# In[11]:


plt.figure(figsize=(10, 6))
plt.title('Distribution of Lap Times')
sns.histplot(f1_data['Total Time (s)'].dropna(), kde=True)
plt.xlabel('Total Time (s)')
plt.show()


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Define the independent variables
X = f1_data[['Starting Grid', 'Laps']]

# Define the dependent variable
y = f1_data['Points']

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate the mean absolute error of the predictions
mae = metrics.mean_absolute_error(y_test, predictions)
mae


# In[15]:


driver_avg_points = f1_data.groupby('Driver')['Points'].mean().reset_index()
driver_avg_points.columns = ['Driver', 'Average Points']
driver_avg_points.head()


# In[16]:


driver_podiums = f1_data[f1_data['Position'].isin(['1', '2', '3'])].groupby('Driver').size().reset_index()
driver_podiums.columns = ['Driver', 'Podiums']
driver_podiums.head()


# In[17]:


unique_values = f1_data['Total Time/Gap/Retirement'].unique()
unique_values


# In[19]:


def convert_to_seconds(time_str):
    try:
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        return np.nan

f1_data['Total Time (s)'] = f1_data['Total Time/Gap/Retirement'].apply(convert_to_seconds)
f1_data['Average Lap Time (s)'] = f1_data['Total Time (s)'] / f1_data['Laps']

driver_avg_lap_time = f1_data.groupby('Driver')['Average Lap Time (s)'].mean().reset_index()
driver_avg_lap_time.columns = ['Driver', 'Average Lap Time']
driver_avg_lap_time.head()


# In[20]:


driver_avg_lap_time = driver_avg_lap_time.dropna()
driver_avg_lap_time.head()


# In[21]:


driver_data = pd.merge(driver_avg_points, driver_podiums, on='Driver', how='outer')
driver_data = pd.merge(driver_data, driver_avg_lap_time, on='Driver', how='outer')
driver_data = driver_data.fillna(0)
driver_data.head()


# In[22]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
driver_data_scaled = scaler.fit_transform(driver_data[['Average Points', 'Podiums', 'Average Lap Time']])

# Determine the optimal number of clusters using the Elbow method
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(driver_data_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow method for optimal k')
plt.show()


# In[23]:


kmeans = KMeans(n_clusters=3, random_state=0).fit(driver_data_scaled)
driver_data['Cluster'] = kmeans.labels_
driver_data.head()


# In[24]:


plt.figure(figsize=(10, 7))
plt.scatter(driver_data['Average Points'], driver_data['Average Lap Time'], c=driver_data['Cluster'])
plt.xlabel('Average Points')
plt.ylabel('Average Lap Time')
plt.title('Driver Clusters')
plt.show()


# In[36]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Load the data
f1_data = pd.read_csv('formula1_2020season_raceResults.csv')

# Preprocess the data
f1_data_preprocessed = f1_data.copy()
f1_data_preprocessed['Position'] = pd.to_numeric(f1_data_preprocessed['Position'], errors='coerce')
conditions = [
    (f1_data_preprocessed['Position'] <= 3),
    (f1_data_preprocessed['Position'] <= 10) & (f1_data_preprocessed['Position'] > 3),
    (f1_data_preprocessed['Position'] > 10),
    (f1_data_preprocessed['Position'].isna())
]
choices = ['top_3', 'points_scoring', 'non_points_scoring', 'DNF']
f1_data_preprocessed['race_finish_position_category'] = np.select(conditions, choices, default='DNF')
le = LabelEncoder()
f1_data_preprocessed['Driver'] = le.fit_transform(f1_data_preprocessed['Driver'])
f1_data_preprocessed['Team'] = le.fit_transform(f1_data_preprocessed['Team'])
f1_data_preprocessed['race_finish_position_category'] = le.fit_transform(f1_data_preprocessed['race_finish_position_category'])

# Define the features and the target
X = f1_data_preprocessed[['Driver', 'Team', 'Starting Grid']]
y = f1_data_preprocessed['race_finish_position_category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print('Training set:', X_train.shape, y_train.shape)
print('Testing set:', X_test.shape, y_test.shape)


# In[37]:


# Create a Logistic Regression model
lr = LogisticRegression(random_state=42)

# Train the model
lr.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = lr.predict(X_test)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)


# In[38]:


# Create a Decision Tree model
dt = DecisionTreeClassifier(random_state=42)

# Train the model
dt.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = dt.predict(X_test)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)


# In[39]:


# Create a Random Forest model
rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)


# In[40]:


# Calculate the total points scored by each driver
points_scored = f1_data.groupby('Driver')['Points'].sum().sort_values(ascending=False)

# Get the top 10 drivers with the most points scored
top_10_drivers_points = points_scored.head(10)

# Create a bar plot for the top 10 drivers with the most points scored
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_drivers_points.index, y=top_10_drivers_points.values, palette='viridis')
plt.title('Top 10 Drivers with the Most Points Scored')
plt.xlabel('Driver')
plt.ylabel('Total Points Scored')
plt.xticks(rotation=45)
plt.show()


# In[41]:


# Calculate the total points scored by each team
teams_points = f1_data.groupby('Team')['Points'].sum().sort_values(ascending=False)

# Create a bar plot for the teams with the points scored
plt.figure(figsize=(10, 6))
sns.barplot(x=teams_points.index, y=teams_points.values, palette='viridis')
plt.title('Teams Based on Points Scored')
plt.xlabel('Team')
plt.ylabel('Total Points Scored')
plt.xticks(rotation=45)
plt.show()


# In[44]:


# Update the function to handle 'DNS'
def time_to_seconds(time_str):
    if 'Retirement' in time_str or '+' in time_str or 'DNF' in time_str or 'DNS' in time_str:
        return np.nan
    else:
        time_parts = time_str.split(':')
        return int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + float(time_parts[2])

# Apply the updated function to the 'Total Time/Gap/Retirement' column
f1_data['Total Time (s)'] = f1_data['Total Time/Gap/Retirement'].apply(time_to_seconds)

# Display the first few rows of the dataframe
f1_data.head()


# In[46]:


# Remove rows with missing values
f1_team_performance = f1_team_performance.dropna()

# Display the first few rows of the cleaned dataframe
f1_team_performance.head()


# In[47]:


# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the plot
sns.set(style='whitegrid')

# Create a bar plot
plt.figure(figsize=(15, 10))
plot = sns.barplot(x='Total Time (s)', y='Team', hue='Track', data=f1_team_performance, ci=None)

# Set the labels and title
plot.set(xlabel='Average Time (s)', ylabel='Team', title='Average Time Taken by Each Team in Each Race')

# Show the plot
plt.show()


# In[48]:


# Convert 'Starting Grid' and 'Position' columns to numeric
f1_data['Starting Grid'] = pd.to_numeric(f1_data['Starting Grid'], errors='coerce')
f1_data['Position'] = pd.to_numeric(f1_data['Position'], errors='coerce')

# Calculate the correlation between 'Starting Grid' and 'Position'
correlation = f1_data[['Starting Grid', 'Position']].corr()

# Display the correlation
print(correlation)


# In[49]:


# Group by 'Team' and 'Position' and count the number of occurrences
f1_team_position = f1_data.groupby(['Team', 'Position']).size().reset_index(name='Count')

# Display the first few rows of the grouped dataframe
f1_team_position.head()


# In[50]:


# Create a bar plot
plt.figure(figsize=(15, 10))
plot = sns.catplot(x='Position', y='Count', hue='Team', data=f1_team_position, kind='bar', height=6, aspect=2)

# Set the labels and title
plot.set_xlabels('Finish Position')
plot.set_ylabels('Count')
plot.fig.suptitle('Distribution of Finish Positions for Each Team')

# Show the plot
plt.show()


# In[51]:


# Group by 'Team' and 'Driver' and calculate the average position
f1_driver_performance = f1_data.groupby(['Team', 'Driver'])['Position'].mean().reset_index()

# Display the first few rows of the grouped dataframe
f1_driver_performance.head()


# In[52]:


# Create a bar plot
plt.figure(figsize=(15, 10))
plot = sns.barplot(x='Position', y='Driver', hue='Team', data=f1_driver_performance, ci=None)

# Set the labels and title
plot.set(xlabel='Average Finish Position', ylabel='Driver', title='Performance of Drivers Within the Same Team')

# Show the plot
plt.show()


# In[ ]:




