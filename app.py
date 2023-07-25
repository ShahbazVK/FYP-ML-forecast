import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, request, render_template,jsonify
from flask_cors import CORS
import numpy as np
from waitress import serve

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
    
@app.route('/item-recommender', methods=['POST'])
def recommend_item():
    data = request.json
    dataset=data.get('dataset')

    #create a data frame of this dataset
    dataset_df=pd.DataFrame(dataset)
    dataset_df.fillna("Not Seen Yet",inplace=True)
    # dataset_df

    # custom function to create unique set of products

    def unique_items():
        unique_items_list = []
        for person in dataset.keys():
            for items in dataset[person]:
                unique_items_list.append(items)
        s=set(unique_items_list)
        unique_items_list=list(s)
        return unique_items_list

    unique_items()

    # custom function to implement cosine similarity between two items

    def item_similarity(item1,item2):
        both_rated = {}
        for person in dataset.keys():
            if item1 in dataset[person] and item2 in dataset[person]:
                both_rated[person] = [dataset[person][item1],dataset[person][item2]]

        number_of_ratings = len(both_rated)
        if number_of_ratings == 0:
            return 0

        item1_ratings = [[dataset[k][item1] for k,v in both_rated.items() if item1 in dataset[k] and item2 in dataset[k]]]
        item2_ratings = [[dataset[k][item2] for k, v in both_rated.items() if item1 in dataset[k] and item2 in dataset[k]]]
        cs = cosine_similarity(item1_ratings,item2_ratings)
        return cs[0][0]

    #custom function to check most similar items

    def most_similar_items(target_item):
        un_lst=unique_items()
        scores = [(item_similarity(target_item,other_item),target_item+" --> "+other_item) for other_item in un_lst if other_item!=target_item]
        scores.sort(reverse=True)
        return scores

    #custom function to filter the seen products and unseen products of the target user

    def target_products_to_users(target_person):
        target_person_products_list = []
        unique_list =unique_items()
        for products in dataset[target_person]:
            target_person_products_list.append(products)

        s=set(unique_list)
        recommended_products=list(s.difference(target_person_products_list))
        a = len(recommended_products)
        if a == 0:
            return 0
        return recommended_products,target_person_products_list

    def recommendation_phase(target_person):
        if target_products_to_users(target_person=target_person) == 0:
            print(target_person, "has bought the products")
            return -1
        unseen_products,seen_products=target_products_to_users(target_person=target_person)
        seen_ratings = [[dataset[target_person][products],products] for products in dataset[target_person]]
        weighted_avg,weighted_sim = 0,0
        rankings =[]
        for i in unseen_products:
            weighted_avg,weighted_sim = 0,0
            for rate,movie in seen_ratings:
                item_sim=item_similarity(i,movie)
                weighted_avg +=(item_sim*rate)
                weighted_sim +=item_sim
            if (weighted_sim!=0):
                weighted_rank=weighted_avg/weighted_sim
            else:
                weighted_rank=0
            rankings.append([weighted_rank,i])

        rankings.sort(reverse=True)
        return rankings

    tp = data.get('customerId')
    if tp in dataset.keys():
        recommended_items=recommendation_phase(tp)
        if recommended_items != -1:
            print("Recommendation!!!")
            recommended_products_names=[]
            for w,m in recommended_items:
                if w!=0:
                    recommended_products_names.append(m)
            recommended_products_names_json=jsonify(recommended_products_names)
            recommended_products_names_json.headers.add('Access-Control-Allow-Origin', '*')
            return recommended_products_names_json
        else:
            return jsonify([])
    else:
        print("Person not found in the dataset")
        return jsonify([])

# if __name__ == '__main__':
#     app.run()

serve(app, host="0.0.0.0", port=8000)
