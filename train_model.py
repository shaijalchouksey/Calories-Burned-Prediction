import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the CSV files
calories = pd.read_csv('calories.csv')
exercise = pd.read_csv('exercise.csv')

# Merge both datasets on User_ID
data = pd.merge(exercise, calories, on='User_ID')

# Preprocess the data
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Features and target variable
X = data[['Age', 'Gender', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
y = data['Calories']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('calories_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print('Model training complete and saved as calories_model.pkl')
