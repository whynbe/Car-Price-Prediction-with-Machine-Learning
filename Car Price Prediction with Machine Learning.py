# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
df = pd.read_csv(r'C:\Users\Vector\Desktop\Boufti_Yassine_Activité\Internships Tasks\Task 3\car data.csv')
print(f"Data shape : {df.shape}")
print(df)

# Drop columns with more than 40% of its cells are missing values for accuracy reasons
missing_percent=df.isnull().mean()
df=df.loc[:,missing_percent<0.4]

# Drop any rows with missing values to keep things clean initially
df = df.dropna()

# 2. Data Preprocessing & Feature Engineering
# Assuming 'Selling_Price' is what we want to predict (Target)
# pd.get_dummies converts categorical text data into numerical (0s (False) and 1s (True))
df_processed = pd.get_dummies(df, drop_first=True)
print(df_processed)
print(df_processed.columns)

# Separate features (X) and target variable (y)
# Replace 'Selling_Price' with your dataset's actual target column name
X = df_processed.drop('Selling_Price', axis=1)
y = df_processed['Selling_Price']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions on the test data
y_pred = model.predict(X_test)

# 5. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# 6. Visualization: Actual vs. Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Perfect prediction line
plt.title('Actual vs Predicted Car Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.tight_layout()
plt.show()