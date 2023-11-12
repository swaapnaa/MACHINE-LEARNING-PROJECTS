#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd

# Load the data
fifa_data = pd.read_csv('fifa21_male2.csv')

# Display the first few rows of the dataframe
fifa_data.head()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

# Display the summary statistics of the dataframe
fifa_data.describe(include='all')

# Plot the distribution of the 'OVA' column which represents the overall performance of the players
plt.figure(figsize=(10,6))
plt.title('Distribution of Overall Performance of Players')
sns.histplot(fifa_data['OVA'], kde=True)
plt.show()


# In[19]:


# Select only the numeric columns
numeric_columns = fifa_data.select_dtypes(include=['int64', 'float64']).columns

# Group the data by 'Club' and calculate the mean performance metrics for the numeric columns
club_data = fifa_data.groupby('Club')[numeric_columns].mean()

# Display the first few rows of the grouped data
club_data.head()


# In[20]:


# Sort the clubs by the average overall performance rating of their players
club_performance = club_data['OVA'].sort_values(ascending=False)

# Display the top 10 clubs with the highest average overall performance rating
club_performance.head(10)


# In[21]:


# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the visualization
sns.set(style='whitegrid')

# Create a bar plot of the average overall performance rating of the top 10 clubs
plt.figure(figsize=(10, 6))
club_performance_plot = sns.barplot(x=club_performance.head(10).index, y=club_performance.head(10))
club_performance_plot.set_xticklabels(club_performance_plot.get_xticklabels(), rotation=45, horizontalalignment='right')

# Set the title and labels of the plot
plt.title('Top 10 Clubs by Average Overall Performance Rating')
plt.xlabel('Club')
plt.ylabel('Average Overall Performance Rating')

# Save the plot as a .png file
plt.savefig('club_performance_plot.png')

# Display the plot
plt.show()


# In[22]:


# Calculate the number of players in each club
club_player_count = fifa_data['Club'].value_counts()

# Calculate the average age of players in each club
club_average_age = fifa_data.groupby('Club')['Age'].mean()

# Calculate the average overall performance rating of players in each club
club_average_rating = fifa_data.groupby('Club')['OVA'].mean()

# Create a new dataframe to store the calculated metrics
club_analysis = pd.DataFrame({'Number of Players': club_player_count, 'Average Age': club_average_age, 'Average Rating': club_average_rating})

# Display the first few rows of the dataframe
club_analysis.head()


# In[23]:


# Create a bar plot of the number of players in each club
plt.figure(figsize=(10, 6))
player_count_plot = sns.barplot(x=club_analysis['Number of Players'].head(10).index, y=club_analysis['Number of Players'].head(10))
player_count_plot.set_xticklabels(player_count_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Top 10 Clubs by Number of Players')
plt.xlabel('Club')
plt.ylabel('Number of Players')
plt.savefig('player_count_plot.png')
plt.show()

# Create a bar plot of the average age of players in each club
plt.figure(figsize=(10, 6))
age_plot = sns.barplot(x=club_analysis['Average Age'].head(10).index, y=club_analysis['Average Age'].head(10))
age_plot.set_xticklabels(age_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Top 10 Clubs by Average Age of Players')
plt.xlabel('Club')
plt.ylabel('Average Age')
plt.savefig('age_plot.png')
plt.show()

# Create a bar plot of the average overall performance rating of players in each club
plt.figure(figsize=(10, 6))
rating_plot = sns.barplot(x=club_analysis['Average Rating'].head(10).index, y=club_analysis['Average Rating'].head(10))
rating_plot.set_xticklabels(rating_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Top 10 Clubs by Average Overall Performance Rating')
plt.xlabel('Club')
plt.ylabel('Average Rating')
plt.savefig('rating_plot.png')
plt.show()


# In[24]:


# Group the data by player's position
position_data = fifa_data.groupby('Position')

# Calculate the number of players in each position
position_player_count = fifa_data['Position'].value_counts()

# Calculate the average age of players in each position
position_average_age = position_data['Age'].mean()

# Calculate the average overall performance rating of players in each position
position_average_rating = position_data['OVA'].mean()

# Create a new dataframe to store the calculated metrics
position_analysis = pd.DataFrame({'Number of Players': position_player_count, 'Average Age': position_average_age, 'Average Rating': position_average_rating})

# Display the first few rows of the dataframe
position_analysis.head()


# In[61]:


# Create a pie chart of the number of players in each position
plt.figure(figsize=(15, 40))
plt.pie(position_analysis['Number of Players'].head(10), labels=position_analysis['Number of Players'].head(10).index, autopct='%1.1f%%')
plt.title('Top 10 Positions by Number of Players')
plt.savefig('player_count_pie.png')
plt.show()

# Create a scatter plot of the average age vs. average rating of players in each position
plt.figure(figsize=(10, 6))
plt.scatter(position_analysis['Average Age'], position_analysis['Average Rating'])
plt.title('Average Age vs. Average Rating of Players by Position')
plt.xlabel('Average Age')
plt.ylabel('Average Rating')
plt.savefig('age_vs_rating_scatter.png')
plt.show()


# In[28]:


# Select numerical columns
numerical_cols = fifa_data.select_dtypes(include=['int64', 'float64']).columns

# Compute the correlation matrix
corr_matrix = fifa_data[numerical_cols].corr()

# Plot the heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()


# In[29]:


# Check for missing values
missing_values = fifa_data.isnull().sum()
missing_values = missing_values[missing_values > 0]

missing_values


# In[30]:


# Fill missing values with appropriate imputation strategy

# For numerical columns, we'll use median imputation
for col in ['Volleys', 'Curve', 'Agility', 'Balance', 'Jumping', 'Interceptions', 'Positioning', 'Vision', 'Composure', 'Sliding Tackle']:
    fifa_data[col].fillna(fifa_data[col].median(), inplace=True)

# For categorical columns, we'll use mode imputation
for col in ['Club', 'Position', 'Club Logo', 'Joined', 'A/W', 'D/W']:
    fifa_data[col].fillna(fifa_data[col].mode()[0], inplace=True)

# 'Loan Date End' is a special case as it has a large number of missing values because not all players are on loan
# We'll fill the missing values with 'Not on loan'
fifa_data['Loan Date End'].fillna('Not on loan', inplace=True)

# Check if there are any missing values left
fifa_data.isnull().sum().sum()


# In[34]:


pip install scikit-learn


# In[36]:


# Select features and target
X = fifa_data.select_dtypes(include=['int64', 'float64']).drop(['ID', 'OVA'], axis=1)
y = fifa_data['OVA']

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and test sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[37]:


# Train a linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a linear regression object
lr = LinearRegression()

# Train the model using the training sets
lr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = lr.predict(X_test)

# The mean squared error
mse = mean_squared_error(y_test, y_pred)
mse


# In[38]:


# Train a random forest regressor
from sklearn.ensemble import RandomForestRegressor

# Create a random forest regressor object
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the training sets
rf.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_rf = rf.predict(X_test)

# The mean squared error
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_rf


# In[39]:


import pandas as pd

# Load the data
fifa_data = pd.read_csv('fifa21_male2.csv')

# Display the head of the dataframe
fifa_data.head()


# In[40]:


# Count of players by nationality
player_counts = fifa_data['Nationality'].value_counts()

# Display the top 10 nationalities
player_counts.head(10)


# In[41]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the plot
sns.set(style='whitegrid')

# Create a bar plot of the top 10 nationalities
plt.figure(figsize=(10, 6))
sns.barplot(x=player_counts.head(10), y=player_counts.head(10).index)

# Add labels and title
plt.xlabel('Number of Players')
plt.ylabel('Nationality')
plt.title('Top 10 Nationalities in FIFA 21')

# Show the plot
plt.show()


# In[42]:


# Select skill-related features for clustering
features = ['Crossing', 'Finishing', 'Heading Accuracy', 'Short Passing', 'Volleys', 'Dribbling', 'Curve', 'FK Accuracy', 'Long Passing', 'Ball Control', 'Acceleration', 'Sprint Speed', 'Agility', 'Reactions', 'Balance', 'Shot Power', 'Jumping', 'Stamina', 'Strength', 'Long Shots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'Standing Tackle', 'Sliding Tackle', 'GK Diving', 'GK Handling', 'GK Kicking', 'GK Positioning', 'GK Reflexes']

# Create a new dataframe with only the selected features
fifa_data_skills = fifa_data[features]

# Display the head of the new dataframe
fifa_data_skills.head()


# In[43]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

# Fit on the data and transform it
fifa_data_skills_imputed = imputer.fit_transform(fifa_data_skills)

# Create a scaler object
scaler = StandardScaler()

# Fit on the data and transform it
fifa_data_skills_scaled = scaler.fit_transform(fifa_data_skills_imputed)

# Print the shape of the preprocessed data
fifa_data_skills_scaled.shape


# In[44]:


from sklearn.cluster import KMeans

# List to hold the values of the clustering scores for different number of clusters
scores = []

# Range of number of clusters
range_n_clusters = range(1, 15)

# Run K-Means with a range of number of clusters
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(fifa_data_skills_scaled)
    scores.append(kmeans.inertia_)

# Plot the scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Clustering Score')
plt.title('The Elbow Method')
plt.grid(True)
plt.show()


# In[45]:


# Perform K-Means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=0)

# Fit the model
kmeans.fit(fifa_data_skills_scaled)

# Get the cluster assignments for each data point
cluster_assignments = kmeans.labels_

# Add the cluster assignments to the original dataframe
fifa_data['Cluster'] = cluster_assignments

# Display the head of the dataframe with the new 'Cluster' column
fifa_data.head()


# In[48]:


import numpy as np

# Select only the numeric columns from the dataframe
numeric_columns = fifa_data.select_dtypes(include=[np.number]).columns

# Group the data by the 'Cluster' column and compute the mean of the numeric columns
cluster_analysis = fifa_data.groupby('Cluster')[numeric_columns].mean()

# Display the result
cluster_analysis


# In[ ]:




