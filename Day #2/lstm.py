import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def plot_results_multiple(predicted_data, true_data, prediction_len):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')

	for i, data in enumerate(predicted_data):
		padding = [None for p in range(i * prediction_len)]
		plt.plot(padding + data, label='Prediction')
		plt.legend()

	plt.show()

def load_data(filename, seq_len, normalise_window):
	f = open(filename, 'r').read()
	data = f.split('\n')

	sequence_length = seq_len + 1
	result = []
	for index in range(len(data) - sequence_length):
		result.append(data[index: index + sequence_length])

	if normalise_window:
		result = normalise_windows(result)

	result = np.array(result)

	row = round(0.9 * result.shape[0])
	train = result[:int(row), :]
	np.random.shuffle(train)
	x_train = train[:, :-1]
	y_train = train[:, -1]
	x_test = result[int(row):, :-1]
	y_test = result[int(row):, -1]

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
	normalised_data = []
	for window in window_data:
		normaliseed_window = [((float(p) / float(window[0])) - 1) for p in window]
		normalised_data.append(normaliseed_window)

	return normalised_data

def build_model():
	model = Sequential()

	model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(100, return_sequences=False))
	model.add(Dropout(0.2))

	model.add(Dense(output_dim=1))
	model.add(Activation("linear"))

	start = time.time()
	model.compile(loss="mse", optimizer="rmsprop")
	print ("Compilation Time : ", time.time() - start)
	return model

def predict_sequence_full(model, data, window_size):
	curr_frame = data[0]
	predicted = []

	for i in range(len(data)):
		predicted.append(model.predict(curr_frame[newaxis, :, :])[0,0])
		curr_frame = curr_frame[1:]
		curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)

	return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
	prediction_seqs = []

	for i in range(len(data) // prediction_len):
		curr_frame = data[i*prediction_len]
		predicted = []

		for j in range(prediction_len):
			predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)

		prediction_seqs.append(predicted)

	return prediction_seqs