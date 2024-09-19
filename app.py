from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import numpy as np
import joblib
import pickle
from datetime import datetime
from feature_extraction import main

app = Flask(__name__)
app.secret_key = 'Malicious.535'

# Load the trained model (replace 'model.pkl' with your actual model file)
loaded_model=joblib.load('model.pkl')

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def history():
    history = session.get('history',[])
    return render_template('history.html',history=history)

@app.route('/check_url', methods=['POST'])
def check_url():
    url = request.form['url']
    features = main(url)
    features = np.array(features).reshape((1, -1))
    prediction = loaded_model.predict(features)
    
    res = "UNKNOWN"
    if int(prediction[0]) == 0:
        res = "SAFE"
    elif int(prediction[0]) == 1:
        res = "DEFACEMENT"
    elif int(prediction[0]) == 2:
        res = "PHISHING"
    elif int(prediction[0]) == 3:
        res = "MALWARE"

    history = session.get('history',[])
    history.append({'url':url,'status':res,'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    session['history']=history

    return render_template('result.html',url=url,status=res)

if __name__ == '__main__':
    app.run(debug=True)
