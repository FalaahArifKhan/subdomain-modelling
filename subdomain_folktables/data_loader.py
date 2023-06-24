import pandas as pd
import numpy as np
from sys import getsizeof
#from tempeh.configurations import datasets

from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSTravelTime, ACSPublicCoverage, ACSMobility

class CompasDataset():
    # save compas to file (only first time)
    # Following https://fairlearn.org/v0.4.6/auto_examples/plot_binary_classification_COMPAS.html
    '''
    compas_dataset = datasets["compas"]()
    X_train, X_test = compas_dataset.get_X(format=pd.DataFrame)
    y_train, y_test = compas_dataset.get_y(format=np.ndarray)

    (race_train,race_test,) = compas_dataset.get_sensitive_features("race", format=np.ndarray)

    # 0 for Female and 1 for Male
    (sex_train,sex_test,) = compas_dataset.get_sensitive_features("sex", format=np.ndarray)

    X_train['race'] = race_train
    X_train['sex'] = sex_train
    X_test['race'] = race_test
    X_test['sex'] = sex_test
    X_train['recidivism'] = y_train
    X_test['recidivism'] = y_test

    compas_all = X_train.append(X_test)

    compas_all.to_csv("COMPAS.csv", index=False)
    '''
    def __init__(self):
        df = pd.read_csv("COMPAS.csv")
        self.features = ['juv_fel_count', 'juv_misd_count', 'juv_other_count',
       'priors_count', 'age_cat_25 - 45', 'age_cat_Greater than 45',
       'age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M']
        self.target = 'recidivism'
        self.numerical_columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count','priors_count']
        self.categorical_columns = ['age_cat_25 - 45', 'age_cat_Greater than 45','age_cat_Less than 25', 'c_charge_degree_F', 'c_charge_degree_M']
        
        self.X_data = df[self.features]
        self.y_data = df[self.target]
        self.dataset = df
        
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()
        
class ACSMobilityDataset():
    def __init__(self, state, year, with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features = ACSMobility.features
        self.target = ACSMobility.target
        self.categorical_columns = ['MAR','SEX','DIS','ESP','CIT','MIL','ANC','NATIVITY','RELP','DEAR','DEYE','DREM','RAC1P','GCL','COW','ESR']
        self.numerical_columns = ['AGEP', 'SCHL', 'PINCP', 'WKHP', 'JWMNP']

        if with_nulls==True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        self.X_data = X_data[self.categorical_columns].astype('str')
        for col in self.numerical_columns:
            self.X_data[col] = X_data[col]

        self.y_data = acs_data[self.target].apply(lambda x: int(x == 1))

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSPublicCoverageDataset():
    def __init__(self, state, year, with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features = ACSPublicCoverage.features
        self.target = ACSPublicCoverage.target
        self.categorical_columns = ['MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','ESR','ST','FER','RAC1P']
        self.numerical_columns = ['AGEP', 'SCHL', 'PINCP']

        if with_nulls is True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        self.X_data = X_data[self.categorical_columns].astype('str')
        for col in self.numerical_columns:
            self.X_data[col] = X_data[col]
            
        self.y_data = acs_data[self.target].apply(lambda x: int(x == 1))

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSTravelTimeDataset():
    def __init__(self, state, year, with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features = ACSTravelTime.features
        self.target = ACSTravelTime.target
        self.categorical_columns = ['MAR','SEX','DIS','ESP','MIG','RELP','RAC1P','PUMA','ST','CIT','OCCP','POWPUMA','POVPIP']
        self.numerical_columns = ['AGEP', 'SCHL']

        if with_nulls==True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        self.X_data = X_data[self.categorical_columns].astype('str')
        for col in self.numerical_columns:
            self.X_data[col] = X_data[col]
            
        self.y_data = acs_data[self.target].apply(lambda x: int(x > 20))

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()
    

class ACSIncomeDataset():
    def __init__(self, state, year, with_nulls=False):
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features = ACSIncome.features
        self.target = ACSIncome.target
        self.categorical_columns = ['COW','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P']
        self.numerical_columns = ['AGEP', 'SCHL']

        if with_nulls==True:
            X_data = acs_data[self.features]
        else:
            X_data = acs_data[self.features].apply(lambda x: np.nan_to_num(x, -1))

        self.X_data = X_data[self.categorical_columns].astype('str')
        for col in self.numerical_columns:
            self.X_data[col] = X_data[col]
            
        self.y_data = acs_data[self.target].apply(lambda x: int(x > 50000))

        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()

    def update_X_data(self, X_data):
        """
        To save simulated nulls
        """
        self.X_data = X_data
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list()


class ACSEmploymentDataset():
    def __init__(self, state, year, root_dir="data", with_nulls=False, optimize=True,subsample=None, download=False):
        """
        Loading task data: instead of using the task wrapper, we subsample the acs_data dataframe on the task features
        We do this to retain the nulls as task wrappers handle nulls by imputing as a special category
        Alternatively, we could have altered the configuration from here:
        https://github.com/zykls/folktables/blob/main/folktables/acs.py
        """
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=root_dir
        )
        acs_data = data_source.get_data(states=state, download=True)
        self.features = ACSEmployment.features
        self.target = ACSEmployment.target
        self.categorical_columns = ['MAR', 'MIL', 'ESP', 'MIG', 'DREM', 'NATIVITY', 'DIS', 'DEAR', 'DEYE', 'SEX', 'RAC1P', 'RELP', 'CIT', 'ANC','SCHL']
        self.numerical_columns = ['AGEP']

        if with_nulls == False:
            acs_data = acs_data.apply(lambda x: np.nan_to_num(x, -1))

        if subsample !=None:
            acs_data = acs_data.sample(subsample)

        dataset = acs_data[self.features]
        dataset[self.target] = acs_data[self.target].apply(lambda x: int(x == 1))
        self.dataset = dataset
        

    def update_data(self, new_dataset):
        self.dataset = new_dataset
        return self.dataset


def optimize_data_loading(data, categorical):
    """
    Optimizing the dataset size by downcasting categorical columns
    """
    for column in categorical:
        data[column] = pd.to_numeric(data[column], downcast='integer')
    return data
