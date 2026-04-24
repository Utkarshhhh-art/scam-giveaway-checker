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

app = Flask(__name__)
app.secret_key = "secret123"

os.makedirs("static", exist_ok=True)

models_store = {}
scores_store = {}
reports_store = {}

# ---------------- DB ----------------
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

# ---------------- STYLE ----------------
style = """
<style>
*{box-sizing:border-box}
body{
margin:0;
font-family:Inter,Arial,sans-serif;
background:#eef2f7;
color:#1f2937;
}
.layout{
display:flex;
min-height:100vh;
}
.sidebar{
width:250px;
background:#0f172a;
color:white;
padding:25px;
}
.sidebar h2{
margin-top:0;
font-size:22px;
}
.sidebar a{
display:block;
color:#cbd5e1;
padding:12px 0;
text-decoration:none;
}
.sidebar a:hover{color:white}
.main{
flex:1;
padding:30px;
}
.topbar{
background:white;
padding:20px 25px;
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
.card{
background:white;
padding:20px;
border-radius:16px;
box-shadow:0 8px 20px rgba(0,0,0,.06);
margin-bottom:20px;
}
.metric{
font-size:28px;
font-weight:700;
color:#2563eb;
}
input,textarea,select{
width:100%;
padding:12px;
border:1px solid #d1d5db;
border-radius:10px;
margin-top:8px;
margin-bottom:14px;
}
button{
background:#2563eb;
color:white;
border:none;
padding:12px 18px;
border-radius:10px;
cursor:pointer;
font-weight:600;
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
th{background:#f8fafc}
.result{
padding:14px;
border-radius:12px;
font-weight:700;
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
color:#64748b;
font-size:14px;
}
</style>
"""

# ---------------- HELPERS ----------------
def home_page():
    return render_template_string(f"""
    <html><head>{style}</head><body>
    <div class='main'>
      <div class='card'>
        <h1>Fake Giveaway Detector Pro</h1>
        <p class='small'>Enterprise ML dashboard</p>
        <a href='/login'>Login</a> | <a href='/register'>Register</a>
      </div>
    </div>
    </body></html>
    """)

@app.route("/")
def home():
    if "user" in session:
        return redirect("/dashboard")
    return home_page()

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET","POST"])
def register():
    msg=""
    if request.method=="POST":
        try:
            u=request.form["username"]
            p=generate_password_hash(request.form["password"])
            conn=sqlite3.connect("users.db")
            cur=conn.cursor()
            cur.execute("INSERT INTO users(username,password) VALUES(?,?)",(u,p))
            conn.commit()
            conn.close()
            return redirect("/login")
        except:
            msg="Username already exists"

    return render_template_string(f"""
    <html><head>{style}</head><body>
    <div class='main'><div class='card'>
    <h2>Register</h2>
    <form method='POST'>
    <input name='username' placeholder='Username' required>
    <input type='password' name='password' placeholder='Password' required>
    <button>Create Account</button>
    </form>
    {msg}
    </div></div></body></html>
    """)

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET","POST"])
def login():
    msg=""
    if request.method=="POST":
        u=request.form["username"]
        p=request.form["password"]

        conn=sqlite3.connect("users.db")
        cur=conn.cursor()
        cur.execute("SELECT password FROM users WHERE username=?",(u,))
        row=cur.fetchone()
        conn.close()

        if row and check_password_hash(row[0],p):
            session["user"]=u
            return redirect("/dashboard")
        else:
            msg="Invalid credentials"

    return render_template_string(f"""
    <html><head>{style}</head><body>
    <div class='main'><div class='card'>
    <h2>Login</h2>
    <form method='POST'>
    <input name='username' required>
    <input type='password' name='password' required>
    <button>Login</button>
    </form>
    {msg}
    </div></div></body></html>
    """)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard", methods=["GET","POST"])
def dashboard():

    if "user" not in session:
        return redirect("/login")

    user=session["user"]
    pred=""

    if request.method=="POST":

        # TRAIN
        if "train" in request.form:
            try:
                df=pd.read_csv(request.files["file"])

                X=df.drop("label",axis=1)
                y=df["label"]

                X_train,X_test,y_train,y_test=train_test_split(
                    X,y,test_size=0.2,random_state=42
                )

                prep=ColumnTransformer([
                    ("text",TfidfVectorizer(stop_words="english"),"text"),
                    ("cat",OneHotEncoder(handle_unknown="ignore"),["platform","link"]),
                    ("num",StandardScaler(),["account_age_days","likes","followers","verified_account"])
                ])

                algos={
                    "SVM":LinearSVC(),
                    "Logistic":LogisticRegression(max_iter=3000),
                    "Tree":DecisionTreeClassifier(),
                    "Forest":RandomForestClassifier()
                }

                models={}
                scores={}
                best=0
                best_pred=None

                for name,clf in algos.items():
                    pipe=Pipeline([("prep",prep),("clf",clf)])
                    pipe.fit(X_train,y_train)
                    yp=pipe.predict(X_test)
                    acc=round(accuracy_score(y_test,yp)*100,2)

                    models[name]=pipe
                    scores[name]=acc

                    if acc>best:
                        best=acc
                        best_pred=yp

                models_store[user]=models
                scores_store[user]=scores
                reports_store[user]=classification_report(y_test,best_pred)

                cm=confusion_matrix(y_test,best_pred)
                plt.figure(figsize=(6,4))
                sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
                plt.tight_layout()
                plt.savefig(f"static/{user}.png")
                plt.close()

            except Exception as e:
                pred=f"Training Error: {str(e)}"

        # PREDICT
        if "predict" in request.form:
            try:
                if user not in models_store:
                    pred="Please train models first."
                else:
                    model_name=request.form["model"]

                    if model_name not in models_store[user]:
                        pred="Selected model unavailable."
                    else:
                        msg=request.form["message"]

                        sample=pd.DataFrame([{
                            "text":msg,
                            "platform":"Instagram",
                            "link":"unknown.com",
                            "account_age_days":5,
                            "likes":10,
                            "followers":100,
                            "verified_account":0
                        }])

                        out=models_store[user][model_name].predict(sample)[0]
                        pred="Fake Giveaway Detected" if out==1 else "Genuine Giveaway"

            except Exception as e:
                pred=f"Prediction Error: {str(e)}"

    scores=scores_store.get(user,{})
    report=reports_store.get(user,"")
    options=""

    if user in models_store:
        for m in models_store[user]:
            options+=f"<option>{m}</option>"

    rows="".join([f"<tr><td>{k}</td><td>{v}%</td></tr>" for k,v in scores.items()])

    trained = user in models_store

    return render_template_string(f"""
    <html><head>{style}</head><body>
    <div class='layout'>

      <div class='sidebar'>
        <h2>Detector Pro</h2>
        <a href='/dashboard'>Dashboard</a>
        <a href='/logout'>Logout</a>
      </div>

      <div class='main'>

        <div class='topbar'>
          <h2>Welcome, {user}</h2>
          <div class='small'>Machine Learning Fraud Detection Console</div>
        </div>

        <div class='grid'>
          <div class='card'><div class='metric'>{len(scores)}</div>Models</div>
          <div class='card'><div class='metric'>{max(scores.values()) if scores else 0}%</div>Top Accuracy</div>
          <div class='card'><div class='metric'>CSV</div>Upload Dataset</div>
        </div>

        <div class='card'>
          <h3>Train Models</h3>
          <form method='POST' enctype='multipart/form-data'>
            <input type='file' name='file' required>
            <button name='train'>Train Now</button>
          </form>
        </div>

        <div class='card'>
          <h3>Predict Message</h3>
          <form method='POST'>
            {"<select name='model'>"+options+"</select>" if trained else "<p class='small'>Train models first.</p>"}
            <textarea name='message' placeholder='Win free iPhone now click link'></textarea>
            <button name='predict'>Analyze</button>
          </form>
          <div class='result'>{pred}</div>
        </div>

        <div class='card'>
          <h3>Accuracy Table</h3>
          <table>
            <tr><th>Model</th><th>Accuracy</th></tr>
            {rows}
          </table>
        </div>

        <div class='card'>
          <h3>Classification Report</h3>
          <pre>{report}</pre>
        </div>

        <div class='card'>
          <h3>Confusion Matrix</h3>
          <img src='/static/{user}.png'>
        </div>

      </div>
    </div>
    </body></html>
    """)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
