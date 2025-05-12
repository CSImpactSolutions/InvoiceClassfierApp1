
from flask import Flask, request, render_template_string, redirect, send_file
import subprocess
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'C:\\Users\\Senthil Arumugam\\Downloads\\InvoiceClassifierApp_MVP_CleanFinal\\InvoiceClassifierApp\\Invoices'
TRAIN_FOLDER = 'C:\\Users\\Senthil Arumugam\\Downloads\\InvoiceClassifierApp_MVP_CleanFinal\\InvoiceClassifierApp\\TrainData'
SOURCE_FOLDER = 'source'
TARGET_FOLDER = 'C:\\Users\\Senthil Arumugam\\Downloads\\InvoiceClassifierApp_MVP_CleanFinal\\InvoiceClassifierApp\\bin\\output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(SOURCE_FOLDER, exist_ok=True)

HTML = """
<!doctype html>
<html>
<head>
    <title>Invoice Classifier UI</title>
    <style>
        body {
            background: linear-gradient(120deg, #f5f7fa, #c3cfe2);
            font-family: Arial, sans-serif;
            padding: 40px;
        }
        h2 {
            color: #333;
        }
        form {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }
        input[type=submit] {
            background: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        input[type=file] {
            padding: 5px;
            margin-bottom: 10px;
        }
        a {
            text-decoration: none;
            color: #333;
        }
    </style>
</head>
<body>
<h2>Upload Files</h2>
<form method=post enctype=multipart/form-data action="/upload">
  <label>Training Data (.txt):</label><br>
  <input type=file name=training_files multiple><br><br>
  <label>Invoices (.pdf):</label><br>
  <input type=file name=invoice_files multiple><br><br>
  <input type=submit value="Upload">
</form>

<h2>Run Classification</h2>
<form action="/classify" method="post">
  <input type=submit value="Run Classify">
</form>

<h2>Run Similarity Match</h2>
<form action="/match" method="post">
  <input type=submit value="Run Match">
</form>

<h2>Download Outputs</h2>
<ul>
  <li><a href="/download/predictions">Download predictions.csv</a></li>
  <li><a href="/download/similarity">Download similarity_results.csv</a></li>
</ul>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/upload", methods=["POST"])
def upload():
    for f in request.files.getlist("training_files"):
        f.save(os.path.join(TRAIN_FOLDER, f.filename))
    for f in request.files.getlist("invoice_files"):
        f.save(os.path.join(UPLOAD_FOLDER, f.filename))
    return redirect("/")

@app.route("/classify", methods=["POST"])
def classify():
    subprocess.run(
        [
            "dotnet", "run",
            "--project", "InvoiceClassifierApp.csproj"
        ],
        cwd="C:/Users/Senthil Arumugam/Downloads/InvoiceClassifierApp_MVP_CleanFinal/InvoiceClassifierApp",
        shell=True
    )
    return redirect("/")

@app.route("/match", methods=["POST"])
def match():
    subprocess.run(["dotnet", "run", "--project", "InvoiceClassifierApp.csproj", "match", "--source", SOURCE_FOLDER, "--target", TARGET_FOLDER, "--model", "text-embedding-3-small"],cwd="C:\\Users\\Senthil Arumugam\\Downloads\\InvoiceClassifierApp_MVP_CleanFinal\\InvoiceClassifierApp")
    return redirect("/")

@app.route("/download/predictions")
def download_predictions():
    return send_file("C:/Users/Senthil Arumugam/Downloads/InvoiceClassifierApp_MVP_VerifiedFinal/InvoiceClassifierApp/output/predictions.csv",
    as_attachment = True)

@app.route("/download/similarity")
def download_similarity():
    return send_file("output/similarity_results.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
