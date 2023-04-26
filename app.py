import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, request, render_template,jsonify
from flask_cors import CORS
import numpy as np


app = Flask("__name__")
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/forecast", methods=['POST'])
def fypPrediction():
    try:
        compare=request.get_json()['compare']
        data=request.get_json()['data']
        if (compare=='month'):
            DF=pd.DataFrame({'Sales': data})
            series= pd.Series(DF['Sales'].values)
            
            model = ExponentialSmoothing(series, trend='add', damped=False, seasonal='add', seasonal_periods=12).fit()
            holt_pred = model.forecast(12)
            outputs=holt_pred.tolist()
        elif (compare=='year'):
            DF=pd.DataFrame({'Sales': data})
            series= pd.Series(DF['Sales'].values)
            model = ExponentialSmoothing(series, trend='add', damped=False, seasonal=None).fit()
            holt_pred = model.forecast(3)
            outputs=holt_pred.tolist()
    except ValueError:
        outputs='Insufficient Data!'
    except IndexError:
        outputs='Insufficient Data!'
    except:
        outputs='Something went Wrong!'
    # fypPrediction('month')
    
    try:
        outputs=jsonify(outputs)
        outputs.headers.add('Access-Control-Allow-Origin', '*')
    except:
        outputs='Something went Wrong!'

    return outputs
    
app.run()



























# # coding: utf-8

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier 
# from sklearn import metrics
# from flask import Flask, request, render_template,jsonify
# from flask_cors import CORS


# app = Flask("__name__")
# cors = CORS(app, resources={r"/*": {"origins": "*"}})
# # q = ""

# # @app.route("/")
# # def loadPage():
# 	# return render_template('home.html', query="")



# @app.route("/result", methods=['POST'])
# def cancerPrediction():
#     print("HEYYYYY1")
#     dataset_url = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/master/breast-cancer-data.csv"
#     df = pd.read_csv(dataset_url)
#     data = request.get_json()
#     print(data)
#     inputQuery1=data['query1']
#     inputQuery2=data['query2']
#     inputQuery3=data['query3']
#     inputQuery4=data['query4']
#     inputQuery5=data['query5']

#     df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

#     train, test = train_test_split(df, test_size = 0.2)

#     features = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']

#     train_X = train[features]
#     train_y=train.diagnosis
    
#     test_X= test[features]
#     test_y =test.diagnosis

#     model=RandomForestClassifier(n_estimators=100, n_jobs=-1)
#     model.fit(train_X,train_y)

#     prediction=model.predict(test_X)
#     metrics.accuracy_score(prediction,test_y)
#     data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
#     #print('data is: ')
#     #print(data)
#     #016.14, 74.00, 0.01968, 0.05914, 0.1619
    
#     # Create the pandas DataFrame 
#     new_df = pd.DataFrame(data, columns = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean'])
#     single = model.predict(new_df)
#     probability = model.predict_proba(new_df)[:,1]
#     print("probability",probability)
#     if single==1:
#         output1 = "The patient is diagnosed with Breast Cancer"
#         output2 = "Confidence: {}".format(probability*100)
#     else:
#         output1 = "The patient is not diagnosed with Breast Cancer"
#         output2 = ""
    
#     # return jsonify(output1,output2)
#     # return render_template('home.html', output1=output1, output2=output2,)
#     # return jsonify(incomes)
#     # print(jsonify(output1))
#     # print(output1)
#     outputs={'output1':output1,'output2':output2}
#     outputs=jsonify(outputs)
#     outputs.headers.add('Access-Control-Allow-Origin', '*')

#     return outputs
    
# @app.route("/getrequest", )
# def getRequest():
#     # Model starts here
#     # print("HELLLLOOO")
#     # Model ends here
#     incomes = [
#     { 'description': 'salary', 'amount': 5000 }
#     ]
#     return jsonify(incomes)
#     # return render_template('home.html', data="HELLLOOOO",)
# app.run()

