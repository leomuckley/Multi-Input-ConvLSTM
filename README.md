# Implementation of a Multi-Input ConvLSTM for the predicting flood extent


### Abstract:
Flooding is among the most destructive natural disasters in the world.  
As a result of this, there is a great need to develop accurate flood prediction models to prevent 
flooding ahead of time. However, due to the many factors that are related to predictingflood extent, 
floods are highly complex to model.  Machine learning techniques have shown promise in this space but
the current state-of-the-art techniques fail to generaliseto other flood events.  
This study shows that Long Short-Term Memory (LSTM) net-works are effective at predicting flood extent 
by surpassing the current state-of-the-artand generalising to other types of flood events.


**Example**

``` python

def run_lstm_exp(model_list, folds=3, seed=SEED):
    
    tf.random.set_seed(seed)

    exp_dict = defaultdict(list)
    for model in model_list:
        if model == lmi:
            for fold in range(1, folds+1):
                print(f"*Training on fold {fold}")
                model.fit(X_train_temp, X_train_const, y_train, X_val_temp, X_val_const, y_val)
                preds = model.predict(X_test_temp, X_test_const).reshape(len(X_test,))
                preds[preds < 0.0] = 0.0
                preds[preds > 1.0] = 1.0   
                y = y_test.flatten()
                res = rmse(preds, y)
                exp_dict[model].append(np.round(res, 6))
        else:
            for fold in range(1, folds+1):
                print(f"*Training on fold {fold}")
                model.fit(X_train, y_train, X_val, y_val)
                preds = model.predict(X_test).reshape(len(X_test,))
                y = y_test.flatten()
                preds[preds < 0.0] = 0.0
                preds[preds > 1.0] = 1.0                 
                res = rmse(preds, y)
                exp_dict[model].append(np.round(res, 6))
    return exp_dict

```

