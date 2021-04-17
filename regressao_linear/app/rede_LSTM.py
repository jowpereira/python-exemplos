import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import keras as k
import matplotlib.pyplot as plt 


def split_sequence(sequence, n_step_in, n_step_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_step_in
        out_end_ix = end_ix + n_step_out

        if out_end_ix > len(sequence):
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

df = pd.read_csv(r'C:\Users\Jonathan\Desktop\regressao_linear\Base\shampoo_2.csv')
df = df.fillna(df.rolling(3, min_periods=1).mean().shift(1))
df = df.set_index('date').sort_index()

raw_seq = df.values

train_size = int(len(raw_seq) * 0.7)
train, test = raw_seq[0:train_size], raw_seq[train_size:len(raw_seq)]

step_in  = 3
step_out = 3

X_train, y_train = split_sequence(train, step_in, step_out)
X_test, y_test = split_sequence(test, step_in, step_out)

feature = 1
epochs  = 200

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], feature))
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], feature))
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu', input_shape=(step_in, feature))))
model.add(Dense(step_out))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train
                   ,y_train
                   ,batch_size=2
                   ,epochs=epochs
                   ,validation_data=(X_test, y_test)
                   ,callbacks=[k.callbacks.EarlyStopping(patience=epochs // 1)
                              ,k.callbacks.ModelCheckpoint(r'C:\Users\Jonathan\Desktop\regressao_linear\Base\model_train.h5'
                                                          ,save_best_only=True
                                                          ,verbose=0)])

plt.title('Learning Curves')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

x_input = np.array([575.5, 407.6, 682.0])
x_input = x_input.reshape((1, step_in, feature))

model = k.models.load_model(r'C:\Users\Jonathan\Desktop\regressao_linear\Base\model_train.h5')

yhat = model.predict(x_input)

yhat = yhat.flatten()

print("previsoes",str(yhat))

print("mape geral -> ", str(np.mean(abs(np.array([475.3, 581.3, 646.9]) - np.array(yhat))/np.array([475.3, 581.3, 646.9]))*100))
print("mape step 1 -> ", str(np.mean(abs(np.array([475.3]) - np.array(yhat[0]))/np.array([475.3]))*100))
print("mape step 2 -> ", str(np.mean(abs(np.array([581.3]) - np.array(yhat[1]))/np.array([581.3]))*100))
print("mape step 3 -> ", str(np.mean(abs(np.array([646.9]) - np.array(yhat[2]))/np.array([646.9]))*100))