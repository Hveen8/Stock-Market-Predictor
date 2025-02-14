import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os


# Enabling multi-GPU useage on 1 node
# gpu_strategy = tf.distribute.MirroredStrategy()
#print(f"Number of GPUs Available: {gpu_strategy.num_replicas_in_sync}")

gpus = len(tf.config.list_physical_devices('GPU'))
print(f"Num GPUs Available: {gpus}")
#print(f"Worker (1 task per node) {os.environ.get('SLURM_PROCID', 'N/A')} sees {len(gpus)} GPU(s).")

# Tensorflow distributed compute strategy (some shit that makes a distributed environment/rule-set)
# gpu_strategy = tf.distribute.MultiWorkerMirroredStrategy()

# LSTM macro parameters
#look_back = 1000 # Could be called shift (when compairing X to Y)
#batch_size = 256
neurons = 100
#epochs = 100

look_back_list = list(range(6000, 6001, 1000))

# look_back_list = [6000]

#look_back_list = list(range(3000, 5001, 1000))

#avoid_list_64 = list(range(100, 901, 100))

# batch_size_list = [512]

batch_size_list = [128, 256]

#epoch_list = list(range(100, 1001, 100))

epoch_list = list(range(5, 21, 5))

# headroom_list = np.arange(1.0, 0.2, -0.1)
# headroom_list = np.linspace(0.4, 0.1, num=2)
headroom_list = [1.5]

total_param_comb = len(look_back_list) * len(batch_size_list) * len(epoch_list) * len(headroom_list)

print('Total Macroparameters to work through: ', total_param_comb)


futureSteps = 2000	# Number of future steps to predict

# Custom scale to leave headroom for predictions
class BufferedMinMaxScaler(MinMaxScaler):
	"""New Subclass, inheriting from MinMaxScalar, gives custom/buffered scale"""
	def __init__(self, headroom=0.5):
		super().__init__()
		# 50% headroom above training max
		# WILL need to customize/work-around as forecast gets limited by the scalar max, could also get limited by the min
		self.headroom = headroom

	def fit(self, X, y=None):
		X = np.asarray(X)
		# 1. Store original data min/max
		self.orig_data_min_ = X.min(axis=0)
		self.orig_data_max_ = X.max(axis=0)
		#self.data_min_ = X.min(axis=0)
		#self.data_max_ = X.max(axis=0)

		# 2. Calculate buffer-adjusted max
		data_range = self.orig_data_max_ - self.orig_data_min_
		self.data_max_ = self.orig_data_max_ + data_range * self.headroom
		self.data_min_ = self.orig_data_min_  # Keep original min (for now, potentially will need to change)

		# 3. Calculate parent class parameters (data_range_ is for parent class, incorporating the headroom)
		self.data_range_ = self.data_max_ - self.data_min_
		self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
		self.min_ = self.feature_range[0] - self.data_min_ * self.scale_

		return self

## Splitting up (Shard) the train data across nodes (For multi-node strategy)
## will split it up based on the current worker index
#def shard_dataset(data_x, data_y, num_workers, worker_index):
#    total_size = len(data_x)
#    shard_size = math.floor(total_size / num_workers)
#	# Worker index will vary per node (I am assuming a worker is a node)
#    start = worker_index * shard_size
#	#cool lambda function
#    end = start + shard_size if worker_index != num_workers - 1 else total_size
#    return data_x[start:end], data_y[start:end]

# Combine historical (from dataset) and forecast data for plot
def create_plot_array(historical, forecasts):
	full_length = len(historical) + len(forecasts)
	plot_array = np.full((full_length, 1), np.nan)
	plot_array[:len(historical)] = historical
	plot_array[len(historical):] = forecasts
	return plot_array

# convert our data into an X (i) and Y (i + shift), specified by LSTM
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):
		# Note the 0 here indicated to put into a 1D array [1, 2, 3]
		# Instead of 2D [[1], [2], [3]]
		a = dataset[i:(i+look_back), 0]
		# ***THIS will appened into shape(X, 3)
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])

	return np.array(dataX), np.array(dataY)

# trim data (that will be X and Y depending on look back) for batch size compatibility
# (removing one data point / time step should not alter the training/pridicting in any meaningful way)
def trim_XY(dataX, dataY, batch_size):
	trim_size = len(dataX) - (len(dataX) % batch_size) # to remove any odd data depending on batch size
	return dataX[:trim_size], dataY[:trim_size]

def forecast(model, start_input, steps, look_back, batch_size):
	# with gpu_strategy.scope():
	# Definining New Model -> idea is to only have batch size = 1 for predictions (but may take a LONG time to predict, although I am not fitting)	
	forecast_model = Sequential()
	forecast_model.add(InputLayer(batch_input_shape=(batch_size, look_back, 1)))
	forecast_model.add(LSTM(neurons, activation='relu', stateful=True, return_sequences=True))
	# forecast_model.add(LSTM(neurons, activation='tanh', stateful=True, return_sequences=True))
	forecast_model.add(Dense(1))
	forecast_model.compile(loss='mean_squared_error', optimizer='adam')

	forecast_model.summary()

	forecast_model.set_weights(model.get_weights())



	new_predictions = []
	# current_batch = start_input[-batch_size:]
	
	# Need to ensure is in 3D shape ([bs, lb, fetr]), as start_input[-batch_size:] gives 2D
	# current_batch = np.repeat(start_input[-1:], batch_size, axis=0)
	current_batch = start_input[-batch_size:]

	print('first batch shape: ', current_batch.shape)
	
	print('Total Batch computes for prediction: ', math.ceil(steps/batch_size))

	for i in range(math.ceil(steps / batch_size)):
		print(f'predicting: {i+1}')
		pred = forecast_model.predict(current_batch, batch_size=batch_size)

		for b in range(batch_size):
			# Should be the ONLY predicted value
			new_predictions.append(pred[b, -1, 0])

		print('Predictions shape: ', pred[i, -1, :].shape)
		print('New Inference (Prediction): ', pred[-1, -1, 0])
		
		# Update each sequence in the batch
		new_batch = np.zeros_like(current_batch)

		for i in range(batch_size):
			# Roll (or shift) and update with previous prediction					 V
			rolled = np.roll(current_batch[i], -1) # shifting the look_back buffer [bs, lb, fetr]
			#print('rolled: ', rolled)
			rolled[-1] = pred[i, -1, 0]
			new_batch[i] = rolled

		current_batch = new_batch
		print('new batch shape: ', current_batch.shape)
	
	# [:steps] trims the array to the exact number of forecasted steps
	print('new_predictions shape: ', np.array(new_predictions).shape)
	return np.array(new_predictions).reshape(-1, 1)

# fix random seed for reproducibility
tf.random.set_seed(7)

# load the dataset
# dataframe = read_csv("UTD_Load_sorted.csv", header=0, engine='python')
dataframe = read_csv("/mnt/slurm_nfs/ece498_w25_20/LSTM_VENV_tmp__V3_10_12/testing_data/UTD_Load_sorted.csv", header=0, engine='python')
# Skip the first column
# columns = dataframe.columns[]
# this 0:0 shit is fucking stupid
#system_names = dataframe.iloc[0:0, 1:]
system_names = dataframe.columns[1:].tolist()
# print(f"Given the following systems: {system_names}")
dataset = dataframe.iloc[:, 1:].astype('float32')

# normalize the dataset (Needs to be done, transformation matrix is inverted after model layers (training))
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# # fix to use Numpy type array later on (right now is Pandas type dataframe)
# dataset_value = dataset.values


for hm in headroom_list:
	for lb in look_back_list:
		for bs in batch_size_list:
			for ep in epoch_list:

				look_back = lb
				batch_size = bs
				epochs = ep
				headroom = hm
				print(f"==========================\n   Current Headroom: {headroom}\n==========================\n")
				#if (lb == 5000) and (bs == 64) and (ep in avoid_list_64):
				#	continue

				# For each system/grid data
				
				for curr_col, curr_system in enumerate(system_names):
					# while True:
					# 	try:
					if curr_system != 'B1':
						continue
					# Starting timer for debug, only good for execl data process and LSTM script
					start_time = time.time()
																																																											
					print(f"Processing column: {curr_system}")

					# Prepare data for the current column
					# curr_dataset = dataset.iloc[:, curr_col].values.reshape(-1, 1)
					curr_dataset = dataframe.iloc[:, curr_col+1].values.reshape(-1, 1).astype('float32')

					# curr_dataset = scaler.fit_transform(curr_dataset)

					# Need to do headroom=0.5 as is a inhereted class definition, not a function
					headroom = headroom
					scaler = BufferedMinMaxScaler(headroom=headroom)
					# Calling overriden method for MinMaxScaler
					scaler.fit(curr_dataset)
					curr_dataset = scaler.transform(curr_dataset)

					print(f'curr_dataset shape: ', curr_dataset.shape)

					# train_size = int(len(curr_dataset) * 0.67)
					# test_size = len(curr_dataset) - train_size
					# train, test = curr_dataset[0:train_size, :], curr_dataset[train_size:len(curr_dataset), :]

					# I have this set temporarily, used to have a test portion, unsure whether test is needed for forecast
					# train_size = len(curr_dataset)
					# train = curr_dataset[0:train_size, :]

					# NOTE for batch and look back size: smaller size better for noisy sequence, larger better for longer characteristics (load profiles)

					# reshape into X=t and Y=t+1
					trainX, trainY = create_dataset(curr_dataset, look_back)
					# testX, testY = create_dataset(test, look_back)

					# In stateful LSTMs, the batch size must evenly divide the dataset size <- chatGPT
					trainX, trainY = trim_XY(trainX, trainY, batch_size)
					# testX, testY = trim_XY(testX, testY, batch_size)
					print('trainX shape (Before Reshape): ', trainX.shape)
					print('trainY shape (Before Reshape): ', trainY.shape)

					# reshape input to be [samples, time steps (points used/looked back at for predict), features]
					# ********* For now features is 1, however if want to include climate and economy data, that may change (would need better data, or combine data)
					# Combining timesteps using number of look_back, more accurate returns
					# LSTMs require data in a specific 3D format to capture relationships between time steps <- chatGPT, but I check elsewhere (keras/TF docs)
					trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
					print('trainX shape (After Reshape): ', trainX.shape)

					# testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

					# with gpu_strategy.scope():
					# create and fit the LSTM network
					model = Sequential()
					# Hard coding each layer: number of samples in a batch | number of time steps in a sample | number of features in a time step
					# batch_input_shape (batch_size, num_steps, features)
					model.add(InputLayer(batch_input_shape=(batch_size, look_back, 1)))
					# LSTM blocks/neurons, I remember reading that odd numbers are not good/useless
					# the more complex the data -> more neurons needed
					model.add(LSTM(neurons, activation='relu', stateful=True, return_sequences=False))
					# model.add(LSTM(neurons, activation='tanh', stateful=True, return_sequences=False))
					model.add(Dense(1))
					model.compile(loss='mean_squared_error', optimizer='adam')
		
					model.summary()

					# **Fitting the Model**
																
					for layer in model.layers:
						if isinstance(layer, LSTM):
							layer.reset_states()

					for i in range(epochs): # it look like its basically trial and error getting the number of epochs right
						# Unlike in CNNs, for RNNs (LSTM fitted) we do not shuffle
						model.fit(trainX, trainY, 
									epochs=1, 
									batch_size=batch_size, 
									shuffle=False, 
									verbose=2)
						
						# reset_states is property of EACH layer, CANNOT use on entire model (this makes it TRUMENDOUSLY slow, need opt)
						# Resetting states after each epoch for stateful LSTM
						for layer in model.layers:
							if isinstance(layer, LSTM):
								layer.reset_states()
						print(f"Epoch {i+1}/{epochs} --- Completed")

					# make predictions, on both training and test data
					# BATCH SIZE: must be the same as the one used in (Keras requires an explicit batch size for stateful RNN)
					trainPredict = model.predict(trainX, batch_size=batch_size)
					print(trainPredict.shape)

					# print(scaler.inverse_transform(trainPredict[:, 0]).shape)
					# reset_states is property of EACH layer, CANNOT use on entire model
					for layer in model.layers:
						if isinstance(layer, LSTM):
							layer.reset_states()
					# # ****** Need to REALLY figure out what the fuck happens here, it shouldn't look at the actual data (data portion that is testX)
					# testPredict = model.predict(testX, batch_size=batch_size)

					# This is to actually predict the future load forecast
					# This gives the last `look_back` values from training data
					# startInput = train[-look_back:]

					print('trainX shape: ', trainX.shape)
					futurePredictions = forecast(model, trainX, futureSteps, look_back, batch_size)
					print(futurePredictions.shape)

					# ====================|
					# Printing time for execl and LSTM script
					end_time = time.time()
					runtime = end_time - start_time
					# print(f"Excel and LSTM training finished - Runtime: {runtime:.4f}")
					hours = int(runtime // 3600)
					minutes = int((runtime % 3600) // 60)
					seconds = int(runtime % 60)
					milliseconds = int((runtime - int(runtime)) * 1000)
					runtime_str = " ".join(
						f"{int(unit)}{label}" for unit, label in
						zip([hours, minutes, seconds, milliseconds], ['h', 'm', 's', 'ms'])
						if unit > 0 or (label=='s' or label=='ms' and hours==0 and minutes==0)
					)
					print(f"LSTM training finished - Runtime: {runtime_str}")
					# ====================|

					# invert predictions
					trainPredict = scaler.inverse_transform(trainPredict)
					trainY = scaler.inverse_transform([trainY])
					# testPredict = scaler.inverse_transform(testPredict)
					# testY = scaler.inverse_transform([testY])
					futurePredictions = scaler.inverse_transform(futurePredictions)

					# calculate root mean squared error
					trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

					print('Train Score: %.2f RMSE' % (trainScore))
					# testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
					# print('Test Score: %.2f RMSE' % (testScore))

					# shift train predictions for plotting (first set up to like the original dataset)
					trainPredictPlot = np.empty_like(curr_dataset)
					trainPredictPlot[:, :] = np.nan
					trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


					# # shift test predictions for plotting
					# testPredictPlot = np.empty_like(curr_dataset)
					# testPredictPlot[:, :] = np.nan
					# testPredictPlot[len(trainPredict)+(look_back):len(trainPredict)+(look_back)+len(testPredict), :] = testPredict

					# shift Forecast predictions for plotting
					# futurePredictPlot = np.empty_like(curr_dataset)
					#futurePredictPlot = np.empty((len(curr_dataset) + len(futurePredictions), curr_dataset.shape[1]))
					#futurePredictPlot[:, :] = np.nan
					# using function for the forecase, could just have a general one
					# futurePredictPlot = create_plot_array(scaler.inverse_transform(curr_dataset), scaler.inverse_transform(futurePredictions))
					# Start plotting predictions after the last dataset point
					#start_idx = len(curr_dataset)
					# print(testPredict)
					#futurePredictPlot[start_idx:len(futurePredictions)+start_idx, :] = futurePredictions

					# For time indices (x-axis)
					full_time = np.arange(len(curr_dataset) + math.ceil(futureSteps/batch_size)*batch_size)

					# Creates plot array with NaNs
					#plot_array = np.full((len(full_time), 1), np.nan)

					plot_array = np.zeros((len(curr_dataset) + len(futurePredictions), 1))
					plot_array[:len(curr_dataset)] = scaler.inverse_transform(curr_dataset)
					plot_array[len(curr_dataset):] = futurePredictions


					# plot baseline and predictions
					plt.plot()
					# Original dataset
					# plt.plot(scaler.inverse_transform(curr_dataset), color='blue', label='Given Data')
					# Training and test data
					# plt.plot(trainPredictPlot, color='green', alpha=0.75, label='Training Data (Prediction)')
					# plt.plot(testPredictPlot, color='red', alpha=0.75, label='Test Data (Prediction)')
					#plt.plot(futurePredictPlot, color='purple', alpha=0.75, linestyle='--', label='Future Predictions')

					plt.plot(full_time[:len(curr_dataset)], plot_array[:len(curr_dataset)], color='blue', linewidth=1.5, label='Given Data')
					plt.plot(full_time[len(curr_dataset):], plot_array[len(curr_dataset):], color='red', linestyle='--', linewidth=1.5, label='Forecast')
					# plot axes
					# Add labels and title
					plt.xlabel('Time')
					plt.ylabel('Value')
					plt.title(f'LSTM Predictions vs. Given Data for {curr_system}')
					plt.legend()

					save_dir = f"/mnt/slurm_nfs/ece498_w25_20/test_slurm5_(relu)_{headroom}_results_batched/"
					# save_dir = f'/mnt/slurm_nfs/ece498_w25_20/test_slurm5_{headroom}_results_batched/'

					plt.savefig(f"{save_dir}{curr_system}_predictions (H_{headroom}) L_{look_back} B_{batch_size} N_{neurons} E_{epochs}.png")
					# plt.savefig(f"{save_dir}{curr_system}_predictions L_{look_back} B_{batch_size} N_{neurons} E_{epochs}.png")
					plt.close()
					print(f"Saved plot for column: {curr_system}")
							
						# 	# For the catch statement
						# 	break
						# except ValueError as e:
						# 	if "Input contains NaN" in str(e):
						# 		print("NaN in prediction, Retrying")
						# 	else:
						# 		raise
