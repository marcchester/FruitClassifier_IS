# Import necessary libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Part 1: Customize the Dataset
# Features: [weight (grams), texture (0=soft, 1=crisp, 2=juicy), color (0=yellow, 1=green, 2=orange)]
# Labels: 0=banana, 1=pear, 2=orange

x = np.array([
[120, 0, 0], # banana
[180, 1, 1], # pear
[140, 2, 2], # orange
[110, 0, 0], # banana
[170, 1, 1], # pear
[100, 0, 0], # banana
[170, 1, 1], # pear
[130, 2, 2], # orange
[1160, 2, 2], # orange
])

y = np.array([0, 1, 2, 0, 1, 0, 1, 2, 2]) # Fruit labels

# Step 2: Create and Train the Model
model = DecisionTreeClassifier()
model.fit(x, y)
print("Training successfully done.")

# Step 3: Run Predictions
test_fruits = np.array([
[90, 0, 0], # likely banana
[150, 1, 1], # likely pear
[160, 2, 2], # likely orange
])
labels_predicted = model.predict(test_fruits)

fruit_names = {0: "Banana", 1: "Pear", 2: "Orange"}
for i, fruits in enumerate(labels_predicted):
    print(f"TheFruit {i+1} is Predicted To Be: {fruit_names[fruits]}")

# Step 4: Check Accuracy Score
# Separate Data into Train and Test Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
splitmodel = DecisionTreeClassifier()
splitmodel.fit(x_train, y_train)
y_predict = splitmodel.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")