from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm

#Loading the data
x_train, y_train, x_test, y_test = lstm.load_data('sp500.csv', 50, True)

if __name__ == "__main__":
	#Build model
	model = lstm.build_model()

	#Train model
	model.fit(x_train,
		y_train,
		batch_size=512,
		nb_epoch=1,
		validation_split=0.05)

	#Plot the predictions
	predictions = lstm.predict_sequences_multiple(model, x_test, 50, 50)
	lstm.plot_results_multiple(predictions, y_test, 50)