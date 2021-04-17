# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
import pandas as pd
import numpy as np
import keras as k

import matplotlib.pyplot as plt


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
df = pd.read_csv(r'Base\shampoo_2.csv')
df = df.fillna(df.rolling(3, min_periods=1).mean().shift(1))
df = df.set_index('date').sort_index()
raw_seq = df.values
train_size = int(len(raw_seq) * 0.7)
train, test = raw_seq[0:train_size], raw_seq[train_size:len(raw_seq)]
# choose a number of time steps
n_steps_in  = 3
n_steps_out = 3
# choose a number of epochs
epochs = 50
# split into samples
X_train, y_train = split_sequence(train, n_steps_in, n_steps_out)
X_test, Y_test = split_sequence(test, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_features)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_features)
Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1])
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(Dense(n_steps_out, activation='linear'))
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(X_train
				   ,y_train
				   ,batch_size=2
				   ,epochs=epochs
				   ,verbose=0
				   ,validation_data=(X_test, Y_test)
				   ,callbacks=[k.callbacks.EarlyStopping(patience=epochs // 1)
				   			  ,k.callbacks.ModelCheckpoint(r'Base\model.h5'
				   			  ,save_best_only=True
				   			  ,verbose=0)])

plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

# demonstrate prediction
x_input = array([575.5, 407.6, 682.0])
x_input = x_input.reshape((1, n_steps_in, n_features))

model = k.models.load_model(r'Base\model.h5')
yhat = model.predict(x_input, verbose=0)

yhat = yhat.flatten()

print("previsoes", str(yhat))

print("mape geral ->", str(np.mean(abs(np.array([475.3, 581.3, 646.9]) - np.array(yhat))/np.array([475.3, 581.3, 646.9]))*100))
print("mape por step 1 ->", str(np.mean(abs(np.array([475.3]) - np.array(yhat[0]))/np.array([475.3]))*100))
print("mape por step 2 ->", str(np.mean(abs(np.array([581.3]) - np.array(yhat[1]))/np.array([581.3]))*100))
print("mape por step 3 ->", str(np.mean(abs(np.array([646.9]) - np.array(yhat[2]))/np.array([646.9]))*100))