# ==========================================================
# app.py
# DETECTOR PRO V2 (UPGRADED)
# Fresh Upload Reset + Better Charts + Multi Model + Render Safe
# ==========================================================

import matplotlib
matplotlib.use("Agg")

import os
import sqlite3
import time
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

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================================
# APP
# ==========================================================

app = Flask(__name__)
app.secret_key = "change_this_secret_key"

os.makedirs("static", exist_ok=True)

models_store = {}
scores_store = {}
reports_store = {}

# ==========================================================
# DATABASE
# ==========================================================

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

# ==========================================================
# STYLE
# ==========================================================

style = """
<style>
*{box-sizing:border-box}
body{
margin:0;
font-family:Arial;
background:#eef2f7;
}
.layout{
display:flex;
min-height:100vh;
}
.sidebar{
width:240px;
background:#0f172a;
color:white;
padding:25px;
}
.sidebar h2{
margin-top:0;
}
.sidebar a{
display:block;
padding:12px 0;
color:#cbd5e1;
text-decoration:none;
font-size:15px;
}
.sidebar a:hover{
color:white;
}
.main{
flex:1;
padding:30px;
}
.grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
gap:18px;
margin-bottom:20px;
}
.card{
background:white;
padding:22px;
border-radius:18px;
box-shadow:0 10px 25px rgba(0,0,0,.06);
margin-bottom:20px;
}
.metric{
font-size:30px;
font-weight:bold;
color:#2563eb;
}
input,textarea,select{
width:100%;
padding:12px;
margin-top:8px;
margin-bottom:14px;
border:1px solid #d1d5db;
border-radius:10px;
}
button{
width:100%;
padding:12px;
border:none;
border-radius:10px;
background:#2563eb;
color:white;
font-weight:bold;
cursor:pointer;
}
button:hover{
background:#1d4ed8;
}
table{
width:100%;
border-collapse:collapse;
}
th,td{
padding:12px;
border-bottom:1px solid #e5e7eb;
text-align:left;
}
th{
background:#f8fafc;
}
.result{
padding:14px;
border-radius:12px;
background:#eff6ff;
color:#1d4ed8;
font-weight:bold;
}
img{
width:100%;
border-radius:12px;
margin-top:10px;
}
.center{
height:100vh;
display:flex;
justify-content:center;
align-items:center;
padding:20px;
}
.auth{
width:430px;
background:white;
padding:30px;
border-radius:18px;
box-shadow:0 10px 25px rgba(0,0,0,.08);
}
pre{
white-space:pre-wrap;
font-size:13px;
}
</style>
"""

# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    if "user" in session:
        return redirect("/dashboard")

    return render_template_string(f"""
    <html><head>{style}</head><body>
    <div class='center'>
      <div class='auth'>
        <h1>Fake Giveaway Detector</h1>
        <p>Advanced ML Fraud Detection Platform</p>

        <a href='/login'><button>Login</button></a><br><br>
        <a href='/register'><button>Create Account</button></a>
      </div>
    </div>
    </body></html>
    """)

# ==========================================================
# REGISTER
# ==========================================================

@app.route("/register", methods=["GET","POST"])
def register():

    msg = ""

    if request.method == "POST":

        try:
            u = request.form["username"]
            p = generate_password_hash(request.form["password"])

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
            msg = "Username already exists"

    return render_template_string(f"""
    <html><head>{style}</head><body>
    <div class='center'>
      <div class='auth'>
        <h2>Create Account</h2>

        <form method='POST'>
        <input name='username' required placeholder='Username'>
        <input type='password' name='password' required placeholder='Password'>
        <button>Create Account</button>
        </form>

        <p style='color:red'>{msg}</p>
      </div>
    </div>
    </body></html>
    """)

# ==========================================================
# LOGIN
# ==========================================================

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
            msg = "Invalid Login"

    return render_template_string(f"""
    <html><head>{style}</head><body>
    <div class='center'>
      <div class='auth'>
        <h2>Login</h2>

        <form method='POST'>
        <input name='username' required placeholder='Username'>
        <input type='password' name='password' required placeholder='Password'>
        <button>Login</button>
        </form>

        <p style='color:red'>{msg}</p>
      </div>
    </div>
    </body></html>
    """)

# ==========================================================
# LOGOUT
# ==========================================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/dashboard", methods=["GET","POST"])
def dashboard():

    if "user" not in session:
        return redirect("/login")

    user = session["user"]
    pred = ""

    # ======================================================
    # TRAIN MODELS
    # ======================================================

    if request.method == "POST" and "train" in request.form:

        try:
            # RESET OLD DATA
            models_store.pop(user, None)
            scores_store.pop(user, None)
            reports_store.pop(user, None)

            for img in [
                f"static/{user}_cm.png",
                f"static/{user}_chart.png"
            ]:
                if os.path.exists(img):
                    os.remove(img)

            df = pd.read_csv(request.files["file"])

            X = df.drop("label", axis=1)
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.25,
                random_state=7,
                stratify=y
            )

            prep = ColumnTransformer([
                ("text",
                 TfidfVectorizer(
                     stop_words="english",
                     max_features=2500
                 ),
                 "text"),

                ("cat",
                 OneHotEncoder(handle_unknown="ignore"),
                 ["platform","link"]),

                ("num",
                 StandardScaler(),
                 [
                    "account_age_days",
                    "likes",
                    "followers",
                    "verified_account"
                 ])
            ])

            algos = {
                "SVM": LinearSVC(class_weight="balanced"),
                "Logistic": LogisticRegression(max_iter=1500),
                "Tree": DecisionTreeClassifier(max_depth=6)
            }

            models = {}
            scores = {}

            best_acc = 0

            for name, clf in algos.items():

                pipe = Pipeline([
                    ("prep", prep),
                    ("clf", clf)
                ])

                pipe.fit(X_train, y_train)

                yp = pipe.predict(X_test)

                acc = round(
                    accuracy_score(y_test, yp)*100,2
                )

                models[name] = pipe
                scores[name] = acc

                if acc > best_acc:
                    best_acc = acc
                    best_pred = yp

            models_store[user] = models
            scores_store[user] = scores
            reports_store[user] = classification_report(
                y_test,
                best_pred
            )

            # ==================================================
            # CONFUSION MATRIX
            # ==================================================

            cm = confusion_matrix(y_test, best_pred)

            plt.figure(figsize=(7,5), dpi=140)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues"
            )
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(f"static/{user}_cm.png")
            plt.close()

            # ==================================================
            # BETTER DATASET CHART
            # ==================================================

            counts = df["label"].value_counts()

            labels = ["Genuine", "Fake"]
            values = [counts.get(0,0), counts.get(1,0)]

            plt.figure(figsize=(8,5), dpi=140)

            bars = plt.barh(labels, values)

            for i,v in enumerate(values):
                plt.text(v+2, i, str(v), va="center")

            plt.title("Dataset Distribution")
            plt.xlabel("Rows")
            plt.tight_layout()
            plt.savefig(f"static/{user}_chart.png")
            plt.close()

        except Exception as e:
            pred = str(e)

    # ======================================================
    # PREDICTION
    # ======================================================

    if request.method == "POST" and "predict" in request.form:

        try:
            model_name = request.form["model"]
            msg = request.form["message"]
            platform = request.form["platform"]

            sample = pd.DataFrame([{
                "text": msg,
                "platform": platform,
                "link": "official.com",
                "account_age_days": 600,
                "likes": 5000,
                "followers": 50000,
                "verified_account": 1
            }])

            bad = [
                "otp","winner","claim",
                "urgent","click","bank"
            ]

            if any(i in msg.lower() for i in bad):
                sample["link"] = "claim-now.xyz"
                sample["account_age_days"] = 10
                sample["likes"] = 5
                sample["followers"] = 50
                sample["verified_account"] = 0

            out = models_store[user][model_name].predict(sample)[0]

            pred = (
                "Fake Giveaway Detected"
                if out == 1
                else "Genuine Giveaway"
            )

        except:
            pred = "Train model first"

    # ======================================================
    # OUTPUT
    # ======================================================

    scores = scores_store.get(user,{})
    report = reports_store.get(user,"")

    options = ""
    rows = ""

    for m in scores:
        options += f"<option>{m}</option>"

    for k,v in scores.items():
        rows += f"<tr><td>{k}</td><td>{v}%</td></tr>"

    stamp = int(time.time())

    return render_template_string(f"""
    <html><head>{style}</head><body>

    <div class='layout'>

      <div class='sidebar'>
        <h2>Detector Pro</h2>
        <a href='/dashboard'>Dashboard</a>
        <a href='/logout'>Logout</a>
      </div>

      <div class='main'>

        <div class='grid'>

          <div class='card'>
            <div class='metric'>{len(scores)}</div>
            Models
          </div>

          <div class='card'>
            <div class='metric'>
            {max(scores.values()) if scores else 0}%
            </div>
            Best Accuracy
          </div>

          <div class='card'>
            <div class='metric'>AI</div>
            Smart Detection
          </div>

        </div>

        <div class='card'>
          <h2>Upload Dataset</h2>

          <form method='POST' enctype='multipart/form-data'>
          <input type='file' name='file' required>
          <button name='train'>Train Models</button>
          </form>
        </div>

        <div class='card'>
          <h2>Quick Prediction</h2>

          <form method='POST'>
          <select name='model'>{options}</select>

          <textarea
          name='message'
          rows='4'
          placeholder='Enter giveaway message'></textarea>

          <select name='platform'>
            <option>Instagram</option>
            <option>Facebook</option>
            <option>Twitter</option>
            <option>YouTube</option>
          </select>

          <button name='predict'>Check</button>
          </form>

          <div class='result'>{pred}</div>
        </div>

        <div class='card'>
          <h2>Accuracy Table</h2>

          <table>
          <tr><th>Model</th><th>Accuracy</th></tr>
          {rows}
          </table>
        </div>

        <div class='card'>
          <h2>Classification Report</h2>
          <pre>{report}</pre>
        </div>

        <div class='card'>
          <h2>Confusion Matrix</h2>
          <img src='/static/{user}_cm.png?x={stamp}'>
        </div>

        <div class='card'>
          <h2>Dataset Distribution</h2>
          <img src='/static/{user}_chart.png?x={stamp}'>
        </div>

      </div>

    </div>

    </body></html>
    """)

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
