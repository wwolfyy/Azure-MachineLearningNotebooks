# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import pickle
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame(data=[{'WeekStarting': '1990-06-14T00:00:00.000Z', 'Store': 2, 'Brand': 'dominicks', 'logQuantity': 9.264828557, 'Advert': 1, 'Price': 1.59, 'Age60': 0.232864734, 'COLLEGE': 0.248934934, 'INCOME': 10.55320518, 'Hincome150': 0.463887065, 'Large HH': 0.103953406, 'Minorities': 0.114279949, 'WorkingWoman': 0.303585347, 'SSTRDIST': 2.110122129, 'SSTRVOL': 1.142857143, 'CPDIST5': 1.927279669, 'CPWVOL5': 0.376926613, 'y_query': 1.0}])


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = Model.get_model_path(model_name = 'AutoML11a264c9d6')
    model = joblib.load(model_path)


@input_schema('data', PandasParameterType(input_sample, enforce_shape=False))
def run(data):
    try:
        y_query = None
        if 'y_query' in data.columns:
            y_query = data.pop('y_query').values
        result = model.forecast(data, y_query)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})

    forecast_as_list = result[0].tolist()
    index_as_df = result[1].index.to_frame().reset_index(drop=True)
    
    return json.dumps({"forecast": forecast_as_list,   # return the minimum over the wire: 
                       "index": json.loads(index_as_df.to_json(orient='records'))  # no forecast and its featurized values
                      })
