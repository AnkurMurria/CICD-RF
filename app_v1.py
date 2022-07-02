from statistics import mode
from flask import Flask, request
import pandas as pd
import numpy as np
import pickle


#point to start flask application
app=Flask(__name__)

pickle_in = open('rf.pkl','rb')
model=pickle.load(pickle_in)


#default method is get
@app.route('/')
def index():
    return "Hello All"

@app.route('/predict')
def prediction_service():
    Cement = request.args.get('Cement')
    Slag = request.args.get('Slag')
    Flyash = request.args.get('Flyash')
    Water = request.args.get('Water')
    SP = request.args.get('SP')
    CoarseAggr = request.args.get('CoarseAggr')
    FineAggr = request.args.get('FineAggr')
    SLUMP = request.args.get('SLUMP')
    FLOW = request.args.get('FLOW')

    #print(Cement, Slag, Flyash, Water, SP, CoarseAggr, FineAggr, SLUMP, FLOW)

    prediction = model.predict([[Cement, Slag, Flyash, Water, SP, CoarseAggr, FineAggr, SLUMP, FLOW]])

    return "The predicted value is=" + str(prediction)

@app.route('/predict_file', methods=["POST"])
def prediction_fromfile():
    df = pd.read_csv(request.files.get("file"))
    print(df.head())
    prediction = model.predict(df)

    return "The predicted values for the csv is=" + str(list(prediction))

#this flask app will start from here
if __name__ =='__main__':
    app.run()