# app.py

import matplotlib
matplotlib.use("Agg")

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, redirect, session, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==================================================
# APP
# ==================================================

app = Flask(__name__)
app.secret_key = "secret123"

os.makedirs("static", exist_ok=True)

models_store = {}
scores_store = {}
reports_store = {}

# ==================================================
# DATABASE
# ==================================================

conn = sqlite3.connect("users.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT UNIQUE,
password TEXT
)
""")

conn.commit()
conn.close()

# ==================================================
# STYLE
# ==================================================

style = """
<style>
body{
font-family:Arial;
background:#0f172a;
padding:30px;
margin:0;
}
.box{
max-width:1100px;
margin:auto;
background:white;
padding:30px;
border-radius:14px;
}
.card{
background:#f8fafc;
padding:20px;
margin-bottom:20px;
border-radius:10px;
}
input,textarea,select{
width:100%;
padding:12px;
margin-top:10px;
margin-bottom:15px;
border-radius:8px;
border:1px solid #ccc;
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
}
a{
text-decoration:none;
}
</style>
"""

# ==================================================
# HOME
# ==================================================

@app.route("/")
def home():

    if "user" in session:
        return redirect("/dashboard")

    return render_template_string(f"""
    <html><head>{style}</head><body>
    <div class='box'>
    <h1>Fake Giveaway Detector</h1>

    <div class='card'>
    <a href='/login'>Login</a>
    </div>

    <div class='card'>
    <a href='/register'>Register</a>
    </div>

    </div>
    </body></html>
    """)

# ==================================================
# REGISTER
# ==================================================

@app.route("/register", methods=["GET","POST"])
def register():

    msg = ""

    if request.method == "POST":

        u = request.form["username"]
        p = generate_password_hash(request.form["password"])

        try:
            conn = sqlite3.connect("users.db")
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users(username,password) VALUES (?,?)",
                (u,p)
            )
            conn.commit()
            conn.close()

            return redirect("/login")

        except:
            msg = "Username exists"

    return render_template_string(f"""
    <html><head>{style}</head><body>
    <div class='box'>
    <h1>Register</h1>

    <div class='card'>
    <form method='POST'>
    <input name='username' required placeholder='Username'>
    <input name='password' type='password' required placeholder='Password'>
    <button>Register</button>
    </form>
    {msg}
    </div>

    </div>
    </body></html>
    """)

# ==================================================
# LOGIN
# ==================================================

@app.route("/login", methods=["GET","POST"])
def login():

    msg = ""

    if request.method == "POST":

        u = request.form["username"]
        p = request.form["password"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT password FROM users WHERE username=?",
            (u,)
        )

        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row[0], p):
            session["user"] = u
            return redirect("/dashboard")
        else:
            msg = "Invalid login"

    return render_template_string(f"""
    <html><head>{style}</head><body>
    <div class='box'>
    <h1>Login</h1>

    <div class='card'>
    <form method='POST'>
    <input name='username' required placeholder='Username'>
    <input name='password' type='password' required placeholder='Password'>
    <button>Login</button>
    </form>
    {msg}
    </div>

    </div>
    </body></html>
    """)

# ==================================================
# LOGOUT
# ==================================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ==================================================
# DASHBOARD
# ==================================================

@app.route("/dashboard", methods=["GET","POST"])
def dashboard():

    if "user" not in session:
        return redirect("/login")

    user = session["user"]
    pred = ""

    if request.method == "POST":

        # TRAIN
        if "train" in request.form:

            file = request.files["file"]
            df = pd.read_csv(file)

            X = df.drop("label", axis=1)
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            prep = ColumnTransformer([
                ("text",
                 TfidfVectorizer(stop_words="english"),
                 "text"),

                ("cat",
                 OneHotEncoder(handle_unknown="ignore"),
                 ["platform","link"]),

                ("num",
                 StandardScaler(),
                 ["account_age_days","likes","followers","verified_account"])
            ])

            algos = {
                "SVM": LinearSVC(),
                "Logistic": LogisticRegression(max_iter=3000),
                "Tree": DecisionTreeClassifier(),
                "Forest": RandomForestClassifier()
            }

            trained = {}
            scores = {}

            best_acc = 0
            best_pred = None

            for name, clf in algos.items():

                pipe = Pipeline([
                    ("prep", prep),
                    ("clf", clf)
                ])

                pipe.fit(X_train, y_train)

                yp = pipe.predict(X_test)

                acc = round(
                    accuracy_score(y_test, yp) * 100,
                    2
                )

                trained[name] = pipe
                scores[name] = acc

                if acc > best_acc:
                    best_acc = acc
                    best_pred = yp

            models_store[user] = trained
            scores_store[user] = scores
            reports_store[user] = classification_report(
                y_test,
                best_pred
            )

            cm = confusion_matrix(y_test, best_pred)

            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.tight_layout()
            plt.savefig(f"static/{user}.png")
            plt.close()

        # PREDICT
        if "predict" in request.form:

            model_name = request.form["model"]
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

            model = models_store[user][model_name]

            out = model.predict(sample)[0]

            pred = (
                "Fake Giveaway Detected"
                if out == 1 else
                "Genuine Giveaway"
            )

    scores = scores_store.get(user, {})
    report = reports_store.get(user, "")

    options = ""

    if user in models_store:
        for m in models_store[user]:
            options += f"<option>{m}</option>"

    rows = ""

    for k,v in scores.items():
        rows += f"<tr><td>{k}</td><td>{v}%</td></tr>"

    return render_template_string(f"""
    <html><head>{style}</head><body>

    <div class='box'>

    <h1>Dashboard</h1>
    <a href='/logout'>Logout</a>

    <div class='card'>
    <h2>Upload CSV</h2>

    <form method='POST' enctype='multipart/form-data'>
    <input type='file' name='file' required>
    <button name='train'>Train</button>
    </form>
    </div>

    <div class='card'>
    <h2>Prediction</h2>

    <form method='POST'>
    <select name='model'>
    {options}
    </select>

    <textarea name='message'></textarea>

    <button name='predict'>Check</button>
    </form>

    <div class='result'>{pred}</div>
    </div>

    <div class='card'>
    <h2>Accuracy</h2>

    <table>
    <tr>
    <th>Model</th>
    <th>Accuracy</th>
    </tr>

    {rows}

    </table>
    </div>

    <div class='card'>
    <h2>Report</h2>
    <pre>{report}</pre>
    </div>

    <div class='card'>
    <h2>Confusion Matrix</h2>
    <img src='/static/{user}.png'>
    </div>

    </div>

    </body></html>
    """)

# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
