import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. CREATE DATASET
# Prices are in Million Tomans
data = {
    'Brand': ['Pride 131', 'Peugeot 206', 'Samand LX', 'Pride 131', 'Peugeot 206', 
              'Samand LX', 'Pride 131', 'Peugeot 206', 'Dena Plus', 'Dena Plus'],
    'Year': [1395, 1398, 1397, 1392, 1400, 1394, 1399, 1396, 1401, 1399],
    'Mileage': [120000, 65000, 90000, 210000, 25000, 180000, 45000, 110000, 15000, 60000],
    'Body_Condition': ['No Paint', 'One Piece', 'No Paint', 'Two Pieces', 'No Paint', 
                       'One Piece', 'No Paint', 'Two Pieces', 'No Paint', 'One Piece'],
    'Price': [210, 420, 380, 165, 510, 310, 245, 365, 750, 610]
}

df = pd.DataFrame(data)

# 2. EXPLORATORY DATA ANALYSIS (EDA)
print('---- First 5 Rows of Dataset ----')
print(df.head(5))

print('\n---- Dataset Information ----')
print(df.info())

# Plot 1: Mileage vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Mileage', y='Price', hue='Brand', data=df, s=100)
plt.title('Mileage vs Price Distribution for Iranian Cars')
plt.xlabel('Mileage (km)')
plt.ylabel('Price (Million Toman)')
plt.grid(True)
plt.show()

# Plot 2: Body Condition vs Price
plt.figure(figsize=(8, 5))
sns.boxplot(x='Body_Condition', y='Price', data=df)
plt.title('Impact of Body Condition on Car Price')
plt.xlabel('Body Condition')
plt.ylabel('Price (Million Toman)')
plt.show()

# 3. FEATURE ENGINEERING & PREPROCESSING
# Calculate the age of the car based on the current Persian year (1405)
df['Age'] = 1405 - df['Year']
df = df.drop('Year', axis=1)

# One-hot encode categorical variables (Brand and Body Condition)
df_encoded = pd.get_dummies(df, columns=['Brand', 'Body_Condition'], drop_first=True)

print('\n---- First 5 Rows After Preprocessing ----')
print(df_encoded.head(5))

# 4. MODEL TRAINING
# Separating features (X) from target (y)
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. MODEL EVALUATION
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\n---- Model Performance ----')
print(f'Mean Absolute Error (MAE): {mae:.2f} Million Toman')
print(f'R-squared Score: {r2:.2f}')

# 6. PREDICTION FOR A NEW SAMPLE CAR
# Predicting the price for a Peugeot 206 (Age: 6 years, Mileage: 80,000 km, No Paint)
sample_car = pd.DataFrame([{
    'Mileage': 80000,
    'Age': 6,
    'Brand_Dena Plus': 0,
    'Brand_Peugeot 206': 1, 
    'Brand_Pride 131': 0,
    'Brand_Samand LX': 0,
    'Body_Condition_One Piece': 0,
    'Body_Condition_Two Pieces': 0 
}])

# Align sample columns with model training features
sample_car = sample_car[X.columns]

predicted_price = model.predict(sample_car)
print(f'\nPredicted Price for the Sample Car: {predicted_price[0]:.2f} Million Toman')