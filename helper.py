import pickle
import pandas as pd
import numpy as np


def load_model(model_name):  # function for load the model and scaler
    model_in = open(model_name, 'rb')
    model = pickle.load(model_in)

    return model


def to_dataframe(pm10, so2, co, o3, no2):
    # convert data input into a dataframe
    # convert to pandas dataframe
    data = {'pm10': [pm10],
            'so2': [so2],
            'co': [co],
            'o3': [o3],
            'no2': [no2]}

    dataframe = pd.DataFrame(data=data)
    return dataframe


def get_summary(data):  # get data input summary
    data_summary = data.copy()
    data_summary = data_summary.rename({'pm10': 'PM10',
                                        'so2': 'SO2',
                                        'co': 'CO',
                                        'o3': 'O3',
                                        'no2': 'NO2'}, axis=1)
    data_summary = data_summary.T
    data_summary.reset_index(inplace=True)
    data_summary = data_summary.rename(
        {'index': 'Features', 0: 'Data Input'}, axis=1)

    return data_summary


def data_preprocessing(data, scaler):  # Preprocess the data
    data_prep = data.copy()

    # log transform for co and no2
    data_prep['co'] = np.log(data_prep['co'])
    data_prep['no2'] = np.log(data_prep['no2'])

    # normalize the data using Robust Scaler
    data_norm = data_prep.copy()
    label = ['pm10', 'so2', 'co', 'o3', 'no2']

    normalized_features = scaler.transform(data_norm)

    # put the normalized data into dataframe
    data_balanced = pd.DataFrame(normalized_features, columns=label)

    return data_balanced


def get_prediction(data_prep, classifier):
    # defining the function which will make the prediction using prepared data
    # make the prediction
    pred_result = classifier.predict(data_prep)

    if pred_result == 0:
        result = 'GOOD'
    elif pred_result == 1:
        result = 'MODERATE'
    else:
        result = 'BAD (UNHEALTHY)'

    return result
