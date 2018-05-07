from flask import Flask, request
import pandas as pd
import numpy as np
import os, json, requests, pickle

app = Flask(__name__)


directry_path = os.path.join(os.path.pardir, os.path.pardir)
model_file_path = os.path.join(directry_path, 'models', 'lr_model.pkl')
scaler_file_path = os.path.join(directry_path, 'models', 'lr_scaler.pkl')

print(model_file_path)

model = pickle.load(open(model_file_path))
scaler = pickle.load(open(scaler_file_path))

columns = [
    u'Age', u'Fare', u'Family_Size', u'IsMother', u'isMale', u'Embarked_C', u'Embarked_Q', \
    u'Embarked_S', u'Title_Lady', u'Title_Master', u'Title_Miss', u'Title_Mr', u'Title_Mrs', u'Title_Officer', \
    u'Title_Sir', u'Age_State_Adults', u'Age_State_Child', u'Deck_A', u'Deck_B', u'Deck_C', u'Deck_D', u'Deck_E', \
    u'Deck_F', u'Deck_G', u'Deck_Z', u'Fare_Bin_very low', u'Fare_Bin_low', u'Fare_Bin_high', u'Fare_Bin_very high', \
    u'Pclass_1', u'Pclass_2', u'Pclass_3'
]

@app.route('/api', methods=['POST'])

def make_predictions():
    #Read Data from request
    data = request.get_json(force=True)
    #Make data frame.
    df = pd.read_json(data)
    # Extract Passenger IDs
    passenger_ids = df['PassengerID'].ravel()
    # Get Actuals from inpiut frame
    actuals = df['Survived'].ravel()
    # Convert dataframe into Matrix format in float
    X = df[columns].as_matrix().astype('float')
    # Scaling Data Set
    x_scaled = scaler.transform(X)
    # Make predictions
    predictions = model.predict(x_scaled)
    # Make response data frame
    df_response = pd.DataFrame({'PassengerID' : passenger_ids, 'Predicted': predictions, 'Actual' : actuals})
    # return response after converting to json
    return df_response.to_json()

if __name__ == '__main__':
    app.run(port=10001, debug=True)
    