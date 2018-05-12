# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 22:29:20 2018

@author: tmatsumoto
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime
import os

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping

def _load_data(X,Y, n_prev=100):
    docX, docY = [], []
    for i in range(X.shape[0]-n_prev):
        docX.append(X[i:i+n_prev])
        docY.append(Y[i+n_prev])
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev=50):
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    
    #loda test data
    EPOCH_NUM = 100
    
    #学習用データの作成
    #学習用データの作成
    csv_path = './currency_data/USDT_BTC_15minutes_4month_20180512_1.csv'
    coin_data = pd.read_csv(csv_path, encoding='UTF-8')
    
    coin_data = coin_data.drop(coin_data.index[len(coin_data)-1])
    #不要なカラムを削除
    coin_data = coin_data.drop(["DATE","USDT_BTC_close_after_15min_diff",
                                "USDT_BTC_close_after_15min_ratio",
                                "USDT_BTC_close_after_15min_log",
                                "USDT_BTC_close_after_15min_flag"], axis=1)

        
#    coin_data_y = coin_data['USDT_BTC_close_after_5min']
#    coin_data_x = coin_data.drop(["USDT_BTC_close_after_5min"], axis=1)
    coin_data = coin_data.fillna(method='ffill')
    coin_data_all = coin_data.as_matrix()
    
    #データを正規化
    scaler = MinMaxScaler()
    yscaler = MinMaxScaler()
    length_of_sequences = 50
    batch_train = int(len(coin_data_all)*0.95)
#    coin_data_batch = make_minibatch(coin_data_train, batch_size)
    
    #データをinputとoutputに分解
    coin_data_train = coin_data_all[:batch_train]
    
    x_train = coin_data_train[:,:-1].astype(np.float32)
    x_train = scaler.fit_transform(x_train)
    
    y_train = coin_data_train[:,-1].astype(np.float32)
    y_train = yscaler.fit_transform(y_train)
    
    train_X ,train_Y = _load_data(x_train, y_train, length_of_sequences)
    
    coin_data_val = coin_data_all[batch_train:]
    
    x_val = coin_data_val[:,:-1].astype(np.float32)
    x_val = scaler.transform(x_val)
    
    y_val = coin_data_val[:,-1].astype(np.float32)
    y_val = yscaler.transform(y_val)
    
#    val_X = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
#    val_Y = y_val
    
    
    val_X ,val_Y = _load_data(x_val, y_val, length_of_sequences)
    #define model
    
    in_neurons = 220
    hidden_neurons = 1000
    out_neurons = 1
    
    model = Sequential()
    
#    LSTM accepts input of shape (n_samples, n_timestamps, ...). 
#    Specifying return_sequences=True makes LSTM layer to return the full history 
#    including outputs at all times
#     (i.e. the shape of output is (n_samples, n_timestamps, n_outdims)), 
#    or the return value contains only the output at the last timestamp
#     (i.e. the shape will be (n_samples, n_outdims)), 
#    which is invalid as the input of the next LSTM layer.

    model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_neurons), return_sequences=False))
#    model.add(LSTM(hidden_neurons, return_sequences=False))
    model.add(Dense(out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="adam",)    
    

    #start learning
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
    history = model.fit(train_X, train_Y, batch_size=600, epochs=100, validation_split=0.1)
    
    f_model ='./model/lstm_model.h5'
    model.save(f_model)