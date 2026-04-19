from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='assets')

# Load dataset
data = pd.read_csv("diabetes.csv")

# Features & target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Create Graph
import os

# Create assets folder if not exists
if not os.path.exists("assets"):
    os.makedirs("assets")

# 1. Diabetes Count Graph
plt.figure()
data['Outcome'].value_counts().plot(kind='bar')
plt.title("Diabetes Count")
plt.savefig("assets/graph1.png")
plt.close()

# 2. Glucose Distribution
plt.figure()
data['Glucose'].plot(kind='hist')
plt.title("Glucose Distribution")
plt.savefig("assets/graph2.png")
plt.close()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        input_data = [
            float(request.form["preg"]),
            float(request.form["glucose"]),
            float(request.form["bp"]),
            float(request.form["skin"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]

        result = model.predict([input_data])[0]

        if result == 1:
            prediction = "Diabetes Detected"
        else:
            prediction = "No Diabetes"

    return render_template("index.html", prediction=prediction, accuracy=round(accuracy*100,2))

app.run(debug=True)