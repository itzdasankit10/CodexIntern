import pandas as pd
import seaborn as sdsds
import matplotlib.pyplot as pp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
a = load_iris()
X = pd.DataFrame(a.data, columns=a.feature_names)
y = pd.Series(a.target, name='species')

# 2. Explore the dataset visually
df = pd.concat([X, y], axis=1)
df['species_name'] = df['species'].apply(lambda x: a.target_names[x])

print("Displaying Data Exploration Plot")
sdsds.pairplot(df, hue='species_name', palette='viridis')
pp.suptitle("Iris Dataset Feature Relationships", y=1.02)
pp.show()

# 3. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(" Data Split ")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 4. Train a K-Nearest Neighbors (KNN) classifier
b = KNeighborsClassifier(n_neighbors=3)
b.fit(X_train, y_train)
print("Model Training Complete")

# 5. Evaluate the model
c = b.predict(X_test)

accuracy = accuracy_score(y_test, c)
print(f"Model Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, c, target_names=a.target_names))

print("Displaying Confusion Matrix ")
cm = confusion_matrix(y_test, c)
sdsds.heatmap(cm, annot=True, fmt='d', xticklabels=a.target_names, yticklabels=a.target_names)
pp.xlabel('Predicted')
pp.ylabel('Actual')
pp.title('Confusion Matrix')
pp.show()
