# ==========================================================
# app.py
# ACCURATE PRO VERSION
# Better Predictions + Better UI + Model Switching
# ==========================================================

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================================
# APP
# ==========================================================

app = Flask(__name__)
app.secret_key = "secret123"

os.makedirs("static", exist_ok=True)

# user wise storage
models_store = {}
scores_store = {}
reports_store = {}
best_store = {}

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
color:#111827;
}
.layout{
display:flex;
min-height:100vh;
}
.sidebar{
width:240px;
background:#0f172a;
padding:25px;
color:white;
}
.sidebar h2{
margin-top:0;
}
.sidebar a{
display:block;
padding:12px 0;
color:#cbd5e1;
text-decoration:none;
}
.sidebar a:hover{
color:white;
}
.main{
flex:1;
padding:30px;
}
.card{
background:white;
padding:22px;
border-radius:16px;
box-shadow:0 8px 20px rgba(0,0,0,.06);
margin-bottom:20px;
}
.grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
gap:18px;
margin-bottom:20px;
}
.metric{
font-size:28px;
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
background:#2563eb;
color:white;
border:none;
padding:12px 18px;
border-radius:10px;
cursor:pointer;
font-weight:bold;
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
font-weight:bold;
background:#eff6ff;
color:#1d4ed8;
}
img{
width:100%;
border-radius:12px;
}
pre{
white-space:pre-wrap;
font-size:13px;
}
.small{
font-size:14px;
color:#6b7280;
}
</style>
"""

# ==========================================================
# ROUTES
# ==========================================================

@app.route("/")
def home():
    if "user" in session:
        return redirect("/dashboard")

    return render_template_string(f"""
    <html><head>{style}</head><body>
    <div class='main'>
      <div class='card'>
        <h1>Fake Giveaway Detector</h1>
        <p class='small'>Professional ML Scam Detection Platform</p>
        <a href='/login'>Login</a> |
        <a href='/register'>Register</a>
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
    <div class='main'>
    <div class='card'>
    <h2>Register</h2>

    <form method='POST'>
    <input name='username' required placeholder='Username'>
    <input type='password' name='password' required placeholder='Password'>
    <button>Create Account</button>
    </form>

    {msg}
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
    <div class='main'>
    <div class='card'>
    <h2>Login</h2>

    <form method='POST'>
    <input name='username' required placeholder='Username'>
    <input type='password' name='password' required placeholder='Password'>
    <button>Login</button>
    </form>

    {msg}
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

    # ------------------------------------------------------
    # TRAIN MODELS
    # ------------------------------------------------------

    if request.method == "POST" and "train" in request.form:

        try:
            df = pd.read_csv(request.files["file"])

            X = df.drop("label", axis=1)
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            prep = ColumnTransformer([

                ("text",
                 TfidfVectorizer(
                     stop_words="english",
                     ngram_range=(1,2),
                     min_df=1
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

                "SVM":
                LinearSVC(class_weight="balanced"),

                "Logistic":
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced"
                ),

                "Forest":
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42
                ),

                "Gradient":
                GradientBoostingClassifier(),

                "Tree":
                DecisionTreeClassifier(max_depth=8)
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
                    accuracy_score(y_test, yp) * 100,
                    2
                )

                models[name] = pipe
                scores[name] = acc

                if acc > best_acc:
                    best_acc = acc
                    best_pred = yp
                    best_name = name

            models_store[user] = models
            scores_store[user] = scores
            best_store[user] = best_name

            reports_store[user] = classification_report(
                y_test,
                best_pred
            )

            cm = confusion_matrix(y_test, best_pred)

            plt.figure(figsize=(6,4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues"
            )

            plt.tight_layout()
            plt.savefig(f"static/{user}.png")
            plt.close()

        except Exception as e:
            pred = f"Training Error: {str(e)}"

    # ------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------

    if request.method == "POST" and "predict" in request.form:

        try:

            if user not in models_store:
                pred = "Please Train Model First"

            else:

                model_name = request.form["model"]

                msg = request.form["message"]
                platform = request.form["platform"]
                link = request.form["link"]
                age = int(request.form["age"])
                likes = int(request.form["likes"])
                followers = int(request.form["followers"])
                verified = int(request.form["verified"])

                sample = pd.DataFrame([{
                    "text": msg,
                    "platform": platform,
                    "link": link,
                    "account_age_days": age,
                    "likes": likes,
                    "followers": followers,
                    "verified_account": verified
                }])

                out = models_store[user][model_name].predict(sample)[0]

                pred = (
                    "Fake Giveaway Detected"
                    if out == 1
                    else
                    "Genuine Giveaway"
                )

        except Exception as e:
            pred = f"Prediction Error: {str(e)}"

    # ------------------------------------------------------
    # DISPLAY DATA
    # ------------------------------------------------------

    scores = scores_store.get(user, {})
    report = reports_store.get(user, "")
    best = best_store.get(user, "")

    options = ""

    if user in models_store:
        for m in models_store[user]:
            options += f"<option>{m}</option>"

    rows = ""

    for k,v in scores.items():
        rows += f"<tr><td>{k}</td><td>{v}%</td></tr>"

    return render_template_string(f"""
    <html>
    <head>{style}</head>
    <body>

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
          Models Trained
        </div>

        <div class='card'>
          <div class='metric'>{max(scores.values()) if scores else 0}%</div>
          Best Accuracy
        </div>

        <div class='card'>
          <div class='metric'>{best}</div>
          Top Model
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
        <h2>Live Prediction</h2>

        <form method='POST'>

          <select name='model'>
          {options}
          </select>

          <textarea name='message'
          rows='4'
          placeholder='Enter giveaway message'></textarea>

          <input name='platform' placeholder='Instagram' required>
          <input name='link' placeholder='official.com' required>
          <input name='age' placeholder='Account Age Days' required>
          <input name='likes' placeholder='Likes' required>
          <input name='followers' placeholder='Followers' required>

          <select name='verified'>
            <option value='1'>Verified</option>
            <option value='0'>Not Verified</option>
          </select>

          <button name='predict'>Check Message</button>

        </form>

        <div class='result'>{pred}</div>

      </div>

      <div class='card'>
        <h2>Model Accuracy</h2>

        <table>
        <tr>
        <th>Model</th>
        <th>Accuracy</th>
        </tr>

        {rows}

        </table>
      </div>

      <div class='card'>
        <h2>Classification Report</h2>
        <pre>{report}</pre>
      </div>

      <div class='card'>
        <h2>Confusion Matrix</h2>
        <img src='/static/{user}.png'>
      </div>

    </div>
    </div>

    </body>
    </html>
    """)

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
