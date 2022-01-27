from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from feature_engine.encoding import OneHotEncoder

import warnings
import pandas as pd
import numpy as np
import os
from scipy import stats

class df_scaler(BaseEstimator, TransformerMixin):
    def __init__(self, method = "standard"):
        self.scaler_obj = None
        self.scale_ = None
        self.method = method
        self.columns = []
        if self.method == 'standard':
            self.mean_ = None
        elif method == 'robust':
            self.center_ = None
        

    def fit(self, X, y = None):
        if self.method == 'standard':
            self.scaler_obj = StandardScaler()
            self.scaler_obj.fit(X)
            self.mean_ = pd.Series(self.scaler_obj.mean_, index = X.columns)
        elif self.method == 'robust':
            self.scaler_obj = RobustScaler()
            self.scaler_obj.fit(X)
            self.center_ = pd.Series(self.scaler_obj.center_, index = X.columns)
        self.scaler_ = pd.Series(self.scaler_obj.scale_, index = X.columns)
        return self

    def transform(self, X):
        X_scaled = self.scaler_obj.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index = X.index, columns = X.columns)
        self.columns = X_scaled_df.columns
        return X_scaled_df

    def get_features_name(self):
        return self.columns

class dummify(BaseEstimator, TransformerMixin):
    '''
    Wrapper for get dummies
    '''
    def __init__(self, variables, drop_first=False, match_cols=True):
        self.drop_first = drop_first
        self.columns = []  # useful to well behave with FeatureUnion
        self.match_cols = match_cols
        self.variables = variables

    def fit(self, X, y=None):
        self.columns = []  # for safety, when we refit we want new columns
        return self
    
    def match_columns(self, X):
        miss_train = list(set(X.columns) - set(self.columns))
        miss_test = list(set(self.columns) - set(X.columns))
        
        err = 0
        
        if len(miss_test) > 0:
            for col in miss_test:
                X[col] = 0  # insert a column for the missing dummy
                err += 1
        if len(miss_train) > 0:
            for col in miss_train:
                del X[col]  # delete the column of the extra dummy
                err += 1
                
        if err > 0:
            warnings.warn('The dummies in this set do not match the ones in the train set, we corrected the issue.',
                         UserWarning)
            
        return X
        
    def transform(self, X):
        X_cat = pd.get_dummies(X[self.variables], drop_first=self.drop_first)
        X_non_cat = X.loc[:, ~X.columns.isin(self.variables)]
        X = pd.concat([X_non_cat, X_cat], axis=1, join='inner')
        if (len(self.columns) > 0): 
            if self.match_cols:
                X = self.match_columns(X)
            self.columns = X.columns
        else:
            self.columns = X.columns
        return X
    
    def get_features_name(self):
        return self.columns

class FeatureUnion_df(BaseEstimator, TransformerMixin):

    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.columns = []
        self.feat_un = FeatureUnion(self.transformer_list,
                                    self.n_jobs,
                                    self.transformer_weights,
                                    self.verbose)

    def fit(self, X, y=None):
        self.feat_un.fit(X)
        return self

    def transform(self, X, y=None):
        X_tr = self.feat_un.transform(X)
        columns = []

        for trsnf in self.transformer_list:
            cols = trsnf[1].steps[-1][1].get_features_name()  # getting the features name from the last step of each pipeline
            columns += list(cols)

        X_tr = pd.DataFrame(X_tr, index=X.index, columns=columns)
        self.columns = columns
        return X_tr

    def get_params(self, deep=True):  # necessary to well behave in GridSearch
        return self.feat_un.get_params(deep=deep)

    def get_features_name(self):
        return self.columns


class feat_sel(BaseEstimator, TransformerMixin):
    '''
    This transformer selects either numerical or categorical features.
    In this way we can build separate pipelines for separate data types.
    '''
    def __init__(self, dtype='numeric'):
        self.dtype = dtype

    def fit( self, X, y=None ):
        return self 

    def transform(self, X, y=None):
        if self.dtype == 'numeric':
            num_cols = X.select_dtypes(include=['float64','int64']).columns.tolist()
            return X[num_cols]
        elif self.dtype == 'category':
            cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
            return X[cat_cols]

