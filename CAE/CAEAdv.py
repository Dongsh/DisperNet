import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt
import h5py
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from dsPltTool import *

def main():

	EPOCHS = 30
	
##### Load Data	 

	with h5py.File('###File Path1###','r') as data_file:

		data_80 = np.array(data_file.get('image'))
		mask_80 = np.array(data_file.get('mask'))

		

	with h5py.File('###File Path2###','r') as data_file:
		data_all = np.array(data_file.get('image'))
		mask_all = np.array(data_file.get('mask'))
		
	with h5py.File('###File Path3###','r') as data_file:

		data_Fu = np.array(data_file.get('image'))
		mask_Fu = np.array(data_file.get('mask'))
		
	
	fMin = 0.0015384615384615385
	fMax = 0.5246153846153846
	cMin = 2007.8277886497065
	cMax = 6000.0
	
	print("[info]Data1 Loaded...")
	print(data_80.shape)
	data_80 = np.transpose(data_80, (2, 0, 1))
	mask_80 = np.transpose(mask_80, (2, 0, 1))

	print("[info]Data2 Loaded...")
	print(data_all.shape)
	data_all = np.transpose(data_all, (2, 0, 1))
	mask_all = np.transpose(mask_all, (2, 0, 1))
	
	print("[info]Data3 Loaded...")
	print(data_Fu.shape)
	data_Fu = data_Fu[:100,:,:]
	mask_Fu = mask_Fu[:100,:,:]
	print(data_Fu.shape)

	print("----")
	
#####  Data Set Covert

	input_width = 512

	x_train_80 = tf.convert_to_tensor(data_80,dtype=tf.float32)
	x_train_80 = tf.expand_dims(x_train_80, -1)
	x_train_80 = tf.image.resize(x_train_80, (512, 512))
	y_train_80 = tf.convert_to_tensor(mask_80)
	y_train_80 = tf.expand_dims(y_train_80, -1)
	
	x_train_all = tf.convert_to_tensor(data_all,dtype=tf.float32)
	x_train_all = tf.expand_dims(x_train_all, -1)
#	x_train_all = tf.image.resize(x_train_all, (512, 512))
	y_train_all = tf.convert_to_tensor(mask_all)
	y_train_all = tf.expand_dims(y_train_all, -1)

	x_train_Fu = tf.convert_to_tensor(data_Fu,dtype=tf.float32)
	x_train_Fu = tf.expand_dims(x_train_Fu, -1)
	x_train_Fu = tf.image.resize(x_train_Fu, (512, 512))
	y_train_Fu = tf.convert_to_tensor(mask_Fu)
	y_train_Fu = tf.expand_dims(y_train_Fu, -1)
		
	x_train = tf.concat([x_train_80, x_train_all, x_train_Fu], axis=0)
	y_train = tf.concat([y_train_80, y_train_all, y_train_Fu], axis=0)

	
	print("[info]Total set shape:")
	print(x_train.shape)

	print("[info]Network Structure...")
	print("training set shape:")
	print(x_train.shape)
	print("network structure:")
	
	inputs = layers.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), name='inputs')
	print(inputs.shape)

	code = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
	code = layers.MaxPool2D((2,2), padding='same')(code)
	print(code.shape)
	
	code = layers.Conv2D(32, (3,3), activation='relu', padding='same')(code)
	code = layers.MaxPool2D((2,2), padding='same')(code)
	print(code.shape)
	
	code = layers.Conv2D(64, (3,3), activation='relu', padding='same')(code)
	code = layers.MaxPool2D((2,2), padding='same')(code)
	print(code.shape)
	
	code = layers.Conv2D(128, (3,3), activation='relu', padding='same')(code)
	code = layers.MaxPool2D((2,2), padding='same')(code)
	print(code.shape)
	
	
#	-----  

	decoded = layers.Conv2D(128, (3,3), activation='relu', padding='same')(code)
	decoded = layers.UpSampling2D((2,2))(decoded)
	print(decoded.shape)
	
	decoded = layers.Conv2D(64, (3,3), activation='relu', padding='same')(decoded)
	decoded = layers.UpSampling2D((2,2))(decoded)
	print(decoded.shape)
	
	decoded = layers.Conv2D(32, (3,3), activation='relu', padding='same')(decoded)
	decoded = layers.UpSampling2D((2,2))(decoded)
	print(decoded.shape)
	
	decoded = layers.Conv2D(16, (3,3), activation='relu', padding='same')(decoded)
	decoded = layers.UpSampling2D((2,2))(decoded)
	print(decoded.shape)

#	decoded = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same')(code)
#	print(decoded.shape)
#	
#	decoded = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', padding='same')(decoded)
#	print(decoded.shape)
#	
#	decoded = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), activation='relu', padding='same')(decoded)
#	print(decoded.shape)
#	
#	decoded = layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), activation='relu', padding='same')(decoded)
#	print(decoded.shape)

	outputs = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(decoded)
	print(outputs.shape)

#### Construct Model

	auto_encoder = keras.Model(inputs, outputs)

	auto_encoder.compile(optimizer=keras.optimizers.Adam(),
						loss=keras.losses.BinaryCrossentropy())
						
	keras.utils.plot_model(auto_encoder, show_shapes=True)
	
	n = 5
	class earlyStopAndDraw(keras.callbacks.EarlyStopping):
		def on_epoch_end(self, epoch, logs=None):
			plt.figure(figsize=(10, 4))
			reverse_direction()
			for i in range(n):
				# display original
				ax = plt.subplot(2, n, i+1)
				plt.imshow(tf.reshape(x_train[i+15],(input_width, input_width)), extent=[fMin,fMax,cMin,cMax], aspect=(fMax-fMin)/(cMax-cMin))

				x_bf(ax=ax, major=0.1,num=5)			
				if i == 0:
					mst(ax=ax, label=['Freq','Velo'], fontsize=5)
					
					
				else:
					mst(ax=ax, label=['Freq',''], fontsize=5)

				ax = plt.subplot(2, n, i + n+1)
				test_data = tf.expand_dims(tf.reshape(x_train[i+15],(input_width, input_width)), -1)
				test_data = tf.expand_dims(test_data, 0)
				predict_data = auto_encoder.predict(test_data)
				plt.imshow(np.squeeze(predict_data), extent=[fMin,fMax,cMin,cMax], aspect=(fMax-fMin)/(cMax-cMin))
				x_bf(ax=ax, major=0.1,num=5)
				ax.spines['bottom'].set_color('w')
				ax.spines['left'].set_color('w')

				if i == 0:
					mst(ax=ax, label=['Freq','Velo'], fontsize=5)
				else:
					mst(ax=ax, label=['Freq',''], fontsize=5)

			plt.show()
			file_name = './un/epoch'+str(epoch)+'.png'
			plt.savefig(file_name, dpi=300)
			plt.close()

#	early_stop = keras.callbacks.EarlyStopping(patience=4, monitor='loss')
	early_stop = earlyStopAndDraw(patience=4, monitor='loss')
	
	print(x_train.shape)
	print(y_train.shape)
	model_history = auto_encoder.fit(x_train, y_train, batch_size=4, epochs=EPOCHS, validation_split=0.1, validation_freq=1,
			callbacks=[early_stop],shuffle=True)

	print("[info]Finish Trainning...")
	auto_encoder.save('auto_encoder_hyper_m.h5')
	
	loss = model_history.history['loss']
	val_loss = model_history.history['val_loss']
	epochs = range(EPOCHS)
	plt.figure(figsize=(5, 5))
	plt.plot(epochs, loss, 'r', label='Training loss')
	plt.plot(epochs, val_loss, 'bo', label='Validation loss')
	plt.title('Training and Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss Value')
	plt.ylim([0, 1])
	plt.legend()
	plt.show()
	plt.savefig('accuracy.png')
	plt.close()


if __name__ == "__main__":
	main()
