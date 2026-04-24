import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, render_template_string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Models
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



app = Flask(__name__)
os.makedirs("static", exist_ok=True)

saved_model = None
best_model_name = None
saved_rows = None
all_scores = {}

# ==========================================
# HTML
# ==========================================

html = """
<!DOCTYPE html>
<html>
<head>
<title>Fake Giveaway Detection</title>

<style>
body{
font-family:Arial;
background:#0f172a;
padding:30px;
margin:0;
}

.container{
max-width:1300px;
margin:auto;
background:white;
padding:30px;
border-radius:15px;
}

h1{
text-align:center;
}

.card{
background:#f8fafc;
padding:20px;
border-radius:12px;
margin-bottom:20px;
}

input, textarea{
width:100%;
padding:12px;
margin-top:10px;
margin-bottom:15px;
}

button{
padding:12px 20px;
background:#2563eb;
color:white;
border:none;
border-radius:8px;
cursor:pointer;
}

table{
width:100%;
border-collapse:collapse;
margin-top:15px;
}

th,td{
border:1px solid #ddd;
padding:10px;
text-align:center;
}

th{
background:#2563eb;
color:white;
}

.result{
padding:15px;
font-size:22px;
font-weight:bold;
text-align:center;
background:#ecfeff;
border-radius:10px;
}

img{
width:100%;
margin-top:10px;
border-radius:10px;
}
</style>
</head>

<body>
<div class="container">

<h1>Fake Giveaway Detection System</h1>

<div class="card">
<h2>Upload Dataset</h2>

<form method="POST" enctype="multipart/form-data">
<input type="file" name="file" required>
<button type="submit" name="train">Train Models</button>
</form>
</div>

{% if scores %}

<div class="card">
<h2>Model Accuracy Comparison</h2>

<table>
<tr>
<th>Model</th>
<th>Accuracy</th>
</tr>

{% for k,v in scores.items() %}
<tr>
<td>{{k}}</td>
<td>{{v}}%</td>
</tr>
{% endfor %}

</table>

<h3>Best Model: {{best}}</h3>

</div>

{% endif %}

<div class="card">
<h2>Live Prediction</h2>

<form method="POST">
<textarea name="message" rows="4"></textarea>
<button type="submit" name="predict">Check Message</button>
</form>

{% if pred %}
<div class="result">{{pred}}</div>
{% endif %}

</div>

{% if report %}
<div class="card">
<h2>Classification Report</h2>
<pre>{{report}}</pre>
</div>

<div class="card">
<h2>Confusion Matrix</h2>
<img src="/static/cm.png">
</div>
{% endif %}

</div>
</body>
</html>
"""

# ==========================================
# Home Route
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    global saved_model, best_model_name, saved_rows, all_scores

    pred = None
    report = None

    if request.method == "POST":

        # ==================================
        # TRAINING
        # ==================================
        if "train" in request.form:

            file = request.files["file"]
            df = pd.read_csv(file)

            saved_rows = len(df)

            X = df.drop("label", axis=1)
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # ==================================
            # Preprocessor
            # ==================================

            preprocessor = ColumnTransformer(
                transformers=[

                    ("text",
                     TfidfVectorizer(stop_words="english"),
                     "text"),

                    ("cat",
                     OneHotEncoder(handle_unknown="ignore"),
                     ["platform", "link"]),

                    ("num",
                     StandardScaler(),
                     ["account_age_days", "likes", "followers", "verified_account"])
                ]
            )

            # ==================================
            # Models
            # ==================================

            models = {
                "Linear SVM": LinearSVC(),
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Naive Bayes": MultinomialNB()
            }

            all_scores = {}

            best_acc = 0

            for name, clf in models.items():

                pipe = Pipeline([
                    ("prep", preprocessor),
                    ("clf", clf)
                ])

                pipe.fit(X_train, y_train)

                pred_test = pipe.predict(X_test)

                acc = round(accuracy_score(y_test, pred_test) * 100, 2)

                all_scores[name] = acc

                if acc > best_acc:
                    best_acc = acc
                    saved_model = pipe
                    best_model_name = name
                    best_pred = pred_test

            report = classification_report(y_test, best_pred)

            # ==================================
            # Confusion Matrix
            # ==================================

            cm = confusion_matrix(y_test, best_pred)

            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(best_model_name)
            plt.tight_layout()
            plt.savefig("static/cm.png")
            plt.close()

        # ==================================
        # Prediction
        # ==================================
        if "predict" in request.form:

            if saved_model is None:
                pred = "Train Model First"

            else:

                msg = request.form["message"]

                sample = pd.DataFrame([{
                    "text": msg,
                    "platform": "Instagram",
                    "link": "unknown.com",
                    "account_age_days": 5,
                    "likes": 10,
                    "followers": 100,
                    "verified_account": 0
                }])

                output = saved_model.predict(sample)[0]

                if output == 1:
                    pred = "Fake Giveaway Detected"
                else:
                    pred = "Genuine Giveaway"

    return render_template_string(
        html,
        pred=pred,
        report=report,
        scores=all_scores,
        best=best_model_name
    )

# ==========================================
# Run
# ==========================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
