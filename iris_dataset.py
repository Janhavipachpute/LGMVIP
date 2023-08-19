import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# Load dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, names=column_names)

# Display basic info
print(iris_data.head())
print(iris_data.info())
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
knn_predictions = knn.predict(X_test_scaled)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_scaled, y_train)
dt_predictions = dt_classifier.predict(X_test_scaled)

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_scaled, y_train)
svm_predictions = svm_classifier.predict(X_test_scaled)
def evaluate_model(predictions, model_name):
    print(f"{model_name} Confusion Matrix:\n{confusion_matrix(y_test, predictions)}")
    print(f"\n{model_name} Classification Report:\n{classification_report(y_test, predictions)}")

evaluate_model(knn_predictions, "K-Nearest Neighbors")
evaluate_model(dt_predictions, "Decision Tree")
evaluate_model(svm_predictions, "Support Vector Machine")
