from flask import Flask, render_template, request
import pickle
from preprocessing import PProcess

app = Flask(__name__)

LR = pickle.load(open("SpamModel.pkl", "rb"))
CV = pickle.load(open("CV.pkl", 'rb'))

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    text = request.form['message']
    text = PProcess(text)  # Preprocess The Data.

    vector = CV.transform([text])
    pvalue = LR.predict(vector)

    prediction = "SPAM"
    if(pvalue[0] == 0): prediction = "NOT SPAM"

    return render_template("index.html", output = prediction, processed = text)

if __name__ == '__main__':
    app.run()