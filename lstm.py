import os, sys, math, time
import json

import tensorflow as tf

import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from tensorflow.keras.utils import to_categorical

from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from technical_analysis.poly_interpolation import PolyInter
from technical_analysis.dpo import Dpo
from technical_analysis.coppock import Coppock

import matplotlib.pyplot as plt






def mean_n_past(data, n): # include current
    l = data.shape[0]
    result = []
    for i in range(l):
        p1 = 0 if i<n else (i-n+1)
        p2 = i + 1
        m = np.mean(data[p1:p2])
        result.append(m)
    result = np.array(result)
    result = np.reshape(result, [l,1])
    return result

def mean_hull(data, n):
    hull = mean_n_past( mean_n_past(data,n//2)*2 - mean_n_past(data,n) , int(math.sqrt(n)) )
    return hull



def extract_data(data):
    # obtain labels
    labels = Genlabels(data, window=25, polyorder=3).labels

    # obtain features
    macd = Macd(data, 6, 12, 3).values
    stoch_rsi = StochRsi(data, period=14).hist_values
    dpo = Dpo(data, period=4).values
    cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
    inter_slope = PolyInter(data, progress_bar=True).values

    macd        = macd[30:-1]
    stoch_rsi   = stoch_rsi[30:-1]
    inter_slope = inter_slope[30:-1]
    dpo         = dpo[30:-1]
    cop         = cop[30:-1]
    labels      = labels[31:]

    # truncate bad values and shift label
    X = np.array([macd, stoch_rsi, inter_slope, dpo, cop])
    X = np.transpose(X)

    data = data[30:-1]

    return X, labels, data

def adjust_data(X, y, split=0.8):
    # count the number of each label
    count_1 = np.count_nonzero(y)
    count_0 = y.shape[0] - count_1
    cut = min(count_0, count_1)

    # save some data for testing
    train_idx = int(cut * split)
    
    # shuffle data
    np.random.seed(42)
    shuffle_index = np.random.permutation(X.shape[0])
    X, y = X[shuffle_index], y[shuffle_index]

    # find indexes of each label
    idx_1 = np.argwhere(y == 1).flatten()
    idx_0 = np.argwhere(y == 0).flatten()

    # grab specified cut of each label put them together 
    X_train = np.concatenate((X[idx_1[:train_idx]]   , X[idx_0[:train_idx]])   , axis=0)
    X_test  = np.concatenate((X[idx_1[train_idx:cut]], X[idx_0[train_idx:cut]]), axis=0)
    y_train = np.concatenate((y[idx_1[:train_idx]]   , y[idx_0[:train_idx]])   , axis=0)
    y_test  = np.concatenate((y[idx_1[train_idx:cut]], y[idx_0[train_idx:cut]]), axis=0)

    # shuffle again to mix labels
    np.random.seed(7)
    shuffle_train = np.random.permutation(X_train.shape[0])
    shuffle_test  = np.random.permutation(X_test.shape[0])

    X_train, y_train = X_train[shuffle_train], y_train[shuffle_train]
    X_test , y_test  = X_test[shuffle_test]  , y_test[shuffle_test]

    return X_train, X_test, y_train, y_test

def shape_data(X, y, data, timesteps=10):
    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # if not os.path.exists('models'):
    #     os.mkdir('models')

    # joblib.dump(scaler, 'models/scaler.dump')

    # reshape data with timesteps
    reshaped = []
    for i in range(timesteps, X.shape[0] + 1):
        reshaped.append(X[i - timesteps:i])
    
    # account for data lost in reshaping
    X = np.array(reshaped)
    y = y[timesteps - 1:]
    data = data[timesteps-1:]

    return X, y, data

def build_model(X):
    # first layer
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))

    # second layer
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))

    # fourth layer and output
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # compile layers
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model





if __name__ == '__main__':
    hist_dir = 'price_history/'
    hist_paths = []
    for path in os.listdir(hist_dir):
        hist_paths.append(hist_dir+path)
    hist_paths.sort()


    c_lst = []
    for path in hist_paths:
        with open(path) as file:
            data = json.load(file)
        c_lst += data['c']


    # load and reshape data
    X, y, data = extract_data(np.array(c_lst))
    X, y, data = shape_data(X, y, data, timesteps=10)


    p = 180
    X_test = X[-p:]
    y_test = y[-p:]
    data_test = data[-p:] 

    # ensure equal number of labels, shuffle, and split
    X_train, X_val, y_train, y_val = adjust_data(X[:-p], y[:-p])

    # binary encode for softmax function
    y_train = to_categorical(y_train, 2)
    y_val   = to_categorical(y_val, 2)
    # y_test  = to_categorical(y_test, 2)

    X_train = tf.convert_to_tensor(X_train)
    y_train = tf.convert_to_tensor(y_train)
    X_val   = tf.convert_to_tensor(X_val  )
    y_val   = tf.convert_to_tensor(y_val  )
    X_test  = tf.convert_to_tensor(X_test )
    y_test  = tf.convert_to_tensor(y_test )


    print(X_train.shape)
    print(y_train.shape)
    print(X_val  .shape)
    print(y_val  .shape)
    print(X_test .shape)
    print(y_test .shape)



    if sys.argv[1] == 'train':
        print('Training phase')
        # build and train model
        model = build_model(X)
        model.fit(X_train, y_train, epochs=10, batch_size=8, shuffle=True, validation_data=(X_val, y_val))
        model.save('lstm_model')

    if sys.argv[1] == 'test':
        print('Testing phase')
        min_model = tf.keras.models.load_model("lstm_model")

        y_pred = []
        annos  = []
        cnt = 0
        for i in range(X_test.shape[0]):
            y = min_model(X_test[i:i+1])[0].numpy()
            label = np.argmax(y)
            y_pred.append(label)
            # score = '%.2f'%(y[label])
            # annos.append(score)
            # y_true = str(y_test[i].numpy())
            # annos.append(y_true)
            if y_test[i] == 1:
                annos.append("U")
            else:
                annos.append("D")
            if y_test[i] == label:
                cnt += 1

        acc = cnt/y_test.shape[0]
        print('Test accuracy: %d%%'%(int(acc*100)))

        hull20 = mean_hull(data_test,20)
        plt.plot(hull20,color='darkviolet')
        plt.plot(data_test,color='b')
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                color = 'g'
            else:
                color = 'r'
            plt.plot(i,hull20[i],color=color, marker='.')
            # plt.annotate(annos[i], (i,data_test[i],), xytext=(0,5), 
            #     textcoords="offset points", ha='center')
        plt.show()


