
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Pregnancies=float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])
            #Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

            with open("tandardScalar.sav", 'rb') as f:
                scalar = pickle.load(f)

            with open("modelForPrediction.sav", 'rb') as f:
                model = pickle.load(f)


            #prediction=loaded_model.predict([[gre_score,toefl_score]])
            data_df = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

            scaled_data = scalar.transform(data_df)

            predict = model.predict(scaled_data)

            if predict == 1:
                result = 'Diabetic'
            else:
                result = 'Non-Diabetic'

            print('prediction is', result)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=result)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app