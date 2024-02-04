from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("/workspaces/Model-Deployment-v1/models/cv.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

with open("/workspaces/Model-Deployment-v1/models/clf.pkl", 'rb') as file:
    clf = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('content')
    
    # Transform the input email using the loaded vectorizer
    tokenized_email = vectorizer.transform([email])

    # Predict using the loaded classifier
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1

    return render_template("index.html", prediction=prediction, content=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
