import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


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


raw_seq = [10,20,30,40,50,60,70,80,90]

step_in  = 3
step_out = 1

X, y = split_sequence(raw_seq, step_in, step_out)

feature = 1

X = X.reshape((X.shape[0], X.shape[1], feature))

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(step_in, feature)))
model.add(Dense(step_out))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=50, verbose=2)

x_input = np.array([70,80,90])
x_input = x_input.reshape((1, step_in, feature))

yhat = model.predict(x_input)

print(yhat)