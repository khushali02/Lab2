import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('/Users/khushali/Desktop/Lab2/wdbc.data', header=None)

# Define column names
column_names = [
    'ID', 'Diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
    'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# Assign column names to the dataframe
data.columns = column_names

# Drop the ID column as it is not needed
# (I'll not remove this as you requested)
# data = data.drop(['ID'], axis=1)

# Encode the diagnosis column
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Split the data into features and target variable
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=35)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
model_v1 = LogisticRegression()
model_v1.fit(X_train, y_train)

# Train a Random Forest Classifier
model_v2 = RandomForestClassifier(n_estimators=100, random_state=35)
model_v2.fit(X_train, y_train)

# Make predictions on the test set for both models
y_pred_v1 = model_v1.predict(X_test)
y_pred_v2 = model_v2.predict(X_test)

# Evaluate the models
accuracy_v1 = accuracy_score(y_test, y_pred_v1)
accuracy_v2 = accuracy_score(y_test, y_pred_v2)

print(f'Accuracy of Logistic Regression: {accuracy_v1:.2f}')
print(f'Accuracy of Random Forest Classifier: {accuracy_v2:.2f}')
