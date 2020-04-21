#!/usr/bin/env python

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DummyTransformer(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer
    
    '''
    def __init__(self):
        pass

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame):
        return df

class Relative_Frequency_Bands(BaseEstimator, TransformerMixin):
    ''' Calculate the frequency bands and the bandpower, lower/upper envelope

    - Calculate different frequency bands
    - For each frequency band calculate:
        - bandpower
        - upper envelope of the bandpower
        - lower envelope of the bandpower
    
    - Calculate mean and standrad deviation of band powers and their envlopoes over 5 windows

    If there are 6 Frequency bands this would be a total of 36 Features (6 x 3 x 2)
    '''
    def __init__(self):
        pass

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame):
        return df