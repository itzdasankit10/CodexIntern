import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Create a new, custom sample dataset
data = {
    'Bedrooms': [3, 4, 2, 5, 3, 4, 2, 3, 5, 6],
    'Bathrooms': [2, 3, 1, 4, 2, 2, 1, 2, 3, 4],
    'Kitchens': [1, 2, 1, 2, 1, 1, 1, 1, 2, 2],
    'Capacity': [6, 8, 4, 10, 6, 8, 4, 6, 10, 12],
    'SquareMeters': [150, 200, 90, 300, 180, 210, 110, 160, 280, 350],
    'Price': [300000, 450000, 180000, 650000, 340000, 480000, 210000, 330000, 600000, 750000]
}
df = pd.DataFrame(data)

# Define the features (X) and the target (y)
feature_names = ['Bedrooms', 'Bathrooms', 'Kitchens', 'Capacity', 'SquareMeters']
X = df[feature_names]
y = df['Price']

# 2. Split data, scale it, and train the new model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

b = StandardScaler()
X_train_scaled = b.fit_transform(X_train)

c = LinearRegression()
c.fit(X_train_scaled, y_train)
print("--- New custom model training is complete ---")


# --- Get live input from the user for the 5 custom features ---
print("\n--- Enter New House Details for Price Prediction ---")

d = []
for feature in feature_names:
    while True:
        try:
            value = float(input(f"Enter value for {feature}: "))
            d.append(value)
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

# Convert, scale, and predict using the new model
e = np.array([d])
f = b.transform(e)
g = c.predict(f)

print(f"\nThe predicted price for the house with your features is: ${g[0]:,.2f}")