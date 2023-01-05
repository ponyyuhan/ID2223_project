from datetime import datetime
import requests
import os
import joblib
import pandas as pd

import json


def decode_features(df, feature_view):
    """Decodes features in the input DataFrame using corresponding Hopsworks Feature Store transformation functions"""
    df_res = df.copy()

    import inspect


    td_transformation_functions = feature_view._batch_scoring_server._transformation_functions

    res = {}
    for feature_name in td_transformation_functions:
        if feature_name in df_res.columns:
            td_transformation_function = td_transformation_functions[feature_name]
            sig, foobar_locals = inspect.signature(td_transformation_function.transformation_fn), locals()
            param_dict = dict([(param.name, param.default) for param in sig.parameters.values() if param.default != inspect._empty])
            if td_transformation_function.name == "min_max_scaler":
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: x * (param_dict["max_value"] - param_dict["min_value"]) + param_dict["min_value"])

            elif td_transformation_function.name == "standard_scaler":
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: x * param_dict['std_dev'] + param_dict["mean"])
            elif td_transformation_function.name == "label_encoder":
                dictionary = param_dict['value_to_index']
                dictionary_ = {v: k for k, v in dictionary.items()}
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: dictionary_[x])
    return df_res


def get_model1(project, model_name, evaluation_metric, sort_metrics_by):
    """Retrieve desired model or download it from the Hopsworks Model Registry.
    In second case, it will be physically downloaded to this directory"""
    TARGET_FILE = "model_tempmax.pkl"
    list_of_files = [os.path.join(dirpath,filename) for dirpath, _, filenames \
                     in os.walk('.') for filename in filenames if filename == TARGET_FILE]

    if list_of_files:
        model_path = list_of_files[0]
        model = joblib.load(model_path)
    else:
        if not os.path.exists(TARGET_FILE):
            mr = project.get_model_registry()
            # get best model based on custom metrics
            model = mr.get_best_model(model_name,
                                      evaluation_metric,
                                      sort_metrics_by)
            model_dir = model.download()
            model = joblib.load(model_dir + "/model_tempmax.pkl")

    return model
def get_model2(project, model_name, evaluation_metric, sort_metrics_by):
    """Retrieve desired model or download it from the Hopsworks Model Registry.
    In second case, it will be physically downloaded to this directory"""
    TARGET_FILE = "model_tempmin.pkl"
    list_of_files = [os.path.join(dirpath,filename) for dirpath, _, filenames \
                     in os.walk('.') for filename in filenames if filename == TARGET_FILE]

    if list_of_files:
        model_path = list_of_files[0]
        model = joblib.load(model_path)
    else:
        if not os.path.exists(TARGET_FILE):
            mr = project.get_model_registry()
            # get best model based on custom metrics
            model = mr.get_best_model(model_name,
                                      evaluation_metric,
                                      sort_metrics_by)
            model_dir = model.download()
            model = joblib.load(model_dir + "/model_tempmin.pkl")

    return model
def get_model(project, model_name, evaluation_metric, sort_metrics_by):
    """Retrieve desired model or download it from the Hopsworks Model Registry.
    In second case, it will be physically downloaded to this directory"""
    TARGET_FILE = "model_temp.pkl"
    list_of_files = [os.path.join(dirpath,filename) for dirpath, _, filenames \
                     in os.walk('.') for filename in filenames if filename == TARGET_FILE]

    if list_of_files:
        model_path = list_of_files[0]
        model = joblib.load(model_path)
    else:
        if not os.path.exists(TARGET_FILE):
            mr = project.get_model_registry()
            # get best model based on custom metrics
            model = mr.get_best_model(model_name,
                                      evaluation_metric,
                                      sort_metrics_by)
            model_dir = model.download()
            model = joblib.load(model_dir + "/model_temp.pkl")

    return model

  
  
