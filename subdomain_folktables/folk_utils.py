import pandas as pd
import numpy as np
import scipy as sp

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def format_params(pipeline_params):
    without_pipeline = {}
    for x in pipeline_params:
        without_pipeline[x.split('__')[1]] = pipeline_params[x]
    return without_pipeline

def initialize_base_model(model_name, saved_params_after_tuning, SEED):
    if model_name == 'lgbm':
        base_model = LGBMClassifier(random_state=SEED, **format_params(saved_params_after_tuning['lgbm']))
    elif model_name == 'rf':
        base_model = RandomForestClassifier(random_state=SEED, **format_params(saved_params_after_tuning['rf']))
    return base_model

def set_protected_groups_by_input(test, protected_groups):
    groups={}
    for i, group_name in enumerate(protected_groups):
        groups[group_name] = test[test[protected_groups[group_name]["Column_name"]] == protected_groups[group_name]["Value"]]
    return groups
        

def set_protected_groups_config(k):
    groups_config={}
    for i in range(k):
        name = "Domain"+str(i+1)
        temp = {
            "Column_name": "predicted_domain",
            "Value" : i
        }
        groups_config[name] = temp
    return groups_config

def intialize_splits_with_cluster_labels(SEED, k, dataset, test_size=0.2):
    train, test = train_test_split(dataset.dataset, test_size=test_size, random_state=SEED)
    
    encoder = ColumnTransformer(transformers=[
                        ('categorical_features', OneHotEncoder(categories='auto', handle_unknown='ignore'), dataset.categorical_columns),
                        ('numerical_features', StandardScaler(), dataset.numerical_columns)])
    model = Pipeline([
                                    ('features', encoder),
                                    ('learner', KMeans(init="random", n_clusters=k, random_state=SEED))
                        ])

    model.fit(train[dataset.features])
    predicted_groups_train = model.predict(train[dataset.features])
    predicted_groups_test = model.predict(test[dataset.features])
    
    train['predicted_domain'] = predicted_groups_train
    test['predicted_domain'] = predicted_groups_test
    
    return train, test

def predict_with_subdomain_model(models, train_groups, test_samples_groups, features_lst, target_name):
    y_true = []
    y_preds = []
    for test_group_name in test_samples_groups.keys():
        if (len(test_samples_groups[test_group_name]) > 0) & (test_group_name in train_groups) :
            y_true+= list(test_samples_groups[test_group_name][target_name].values)
            y_preds+=list(models[test_group_name].predict(test_samples_groups[test_group_name][features_lst]))
    
    return y_true, y_preds


def predict_with_subdomain_model_with_metrics(models, train_groups, test_samples_groups, features_lst, target_name):
    y_true = []
    y_preds = []
    domain_info = []
    for test_group_name in test_samples_groups.keys():
        if (len(test_samples_groups[test_group_name]) > 0) & (test_group_name in train_groups) :
            y_true+= list(test_samples_groups[test_group_name][target_name].values)
            y_preds+=list(models[test_group_name].predict(test_samples_groups[test_group_name][features_lst]))
            domain_info+=[test_group_name]*len(test_samples_groups[test_group_name])

    results_df = pd.DataFrame({"y_true": y_true, "y_pred": y_preds, "domain": domain_info})
    return results_df