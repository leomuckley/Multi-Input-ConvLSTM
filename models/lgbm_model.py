#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:21:27 2020

@author: leo
"""

import pandas as pd
import lightgbm as lgb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error


class LGBM():
    
    
    def __init__(self, df_train, X_test, seed=400):
        self.params = {'learning_rate':0.07,'max_depth':8, 'num_leaves':2*8}
        self.n_estimators = 221
        self.n_iters = 5
        self.df_train = df_train
        self.df_test = X_test
        self.model = None
        self.seed = seed
        
    def metric(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    def check(self, col):
         if col < 0:
             return 0
         elif col > 1:
             return 1
         else:
             return col
     
    def preprocess(self):
        train = self.df_train.copy()
        test = self.df_test.copy()
        train = train.fillna(0)
        test = test.fillna(0)
        train['week_9_precip'] = train[train[[i for i in train.columns if 'max_precip' in i]].columns[-7:]].max().max()
        train['week_8_precip'] = train[train[[i for i in train.columns if 'max_precip' in i]].columns[-14:-7]].max().max()
        train['week_7_precip'] = train[train[[i for i in train.columns if 'max_precip' in i]].columns[-21:-14]].max().max()
        train['week_6_precip'] = train[train[[i for i in train.columns if 'max_precip' in i]].columns[:-21]].max().max()
     
        test['week_9_precip'] = test[test[[i for i in test.columns if 'max_precip' in i]].columns[-7:]].max().max()
        test['week_8_precip'] = test[test[[i for i in test.columns if 'max_precip' in i]].columns[-14:-7]].max().max()
        test['week_7_precip'] = test[test[[i for i in test.columns if 'max_precip' in i]].columns[-21:-14]].max().max()
        test['week_6_precip'] = test[test[[i for i in test.columns if 'max_precip' in i]].columns[:-21]].max().max()
    
        train['week_7_precip_']=train['week_7_precip']+train['week_6_precip']
        test['week_7_precip_']=test['week_7_precip']+test['week_6_precip']
        train['week_8_precip_']=train['week_8_precip']+train['week_7_precip']
        test['week_8_precip_']=test['week_8_precip']+test['week_7_precip']
        train['week_9_precip_']=train['week_9_precip']+train['week_8_precip']
        test['week_9_precip_']=test['week_9_precip']+test['week_8_precip']
        train['max_2_weeks']=train[['week_7_precip_','week_8_precip_','week_9_precip_']].apply(np.max,axis=1)
        test['max_2_weeks']=test[['week_7_precip_','week_8_precip_','week_9_precip_']].apply(np.max,axis=1)    
        train['LC_Type1_mode'] = train['land_cover_mode_t-1'].astype(int)
        test['LC_Type1_mode'] = test['land_cover_mode_t-1'].astype(int)    
        train['X'], test['X'] = train['X_t-1'], test['X_t-1']
        train['Y'], test['Y']= train['Y_t-1'], test['Y_t-1']    
        train['elevation'] = train['elevation_t-1']
        test['elevation'] = test['elevation_t-1']    
        train['soil_major'] = train['soc_mean_t-1']
        test['soil_major'] = test['soc_mean_t-1']
        
        def index(col):
            l=list(col)
            return l.index(max(l))
        
        train['max_index'] = train[['week_6_precip', 'week_7_precip', 'week_8_precip','week_9_precip']].apply(index,axis=1)
        test['max_index'] = test[['week_6_precip', 'week_7_precip', 'week_8_precip','week_9_precip']].apply(index,axis=1)
        train['slope_8_7'] = ((train['week_8_precip']/train['week_7_precip'])>1)*1   
        test['slope_8_7'] = ((test['week_8_precip']/test['week_7_precip'])>1)*1        
        train['slope_9_8'] = ((train['week_9_precip']/train['week_8_precip'])>1)*1
        test['slope_9_8'] = ((test['week_9_precip']/test['week_8_precip'])>1)*1      
        cols = ['LC_Type1_mode', 'X', 'Y', 'elevation','week_7_precip', 'week_8_precip', 'week_9_precip',
                'max_2_weeks', 'slope_8_7', 'slope_9_8', 'soil_major', 'target']
        test_cols = ['LC_Type1_mode', 'X', 'Y', 'elevation','week_7_precip', 'week_8_precip', 'week_9_precip',
                     'max_2_weeks', 'slope_8_7', 'slope_9_8', 'soil_major']
        
        return train[cols], test[test_cols]


 
    
    def fit_predict(self, seed):
        train, test = self.preprocess()
        preds_buf = []
        X = train.drop(columns=['target'])
        y = train['target']
        for _ in range(self.n_iters): 
            x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, 
                                                                  random_state=np.random.randint(0, 100))
            d_train = lgb.Dataset(x_train, label=y_train)
            d_valid = lgb.Dataset(x_valid, label=y_valid)
            watchlist = [d_valid]        
            model = lgb.train(self.params, d_train, self.n_estimators, watchlist, verbose_eval=1)        
            #preds = model.predict(x_valid)
            #err_buf.append(metric(model.predict(x_valid),y_valid))
            preds = model.predict(self.df_test)            
            preds_buf.append(preds)
            self.model = model
        preds1 = pd.Series(np.mean(preds_buf, axis=0))
        preds=preds1.apply(self.check)
        preds-=0.08
        return preds
    
    
    # def predict(self, X_test)
    #     return self.model.predict(X_test)
        
        
        

