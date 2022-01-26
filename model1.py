from re import X
from tensorflow.keras.models import Sequential
import tensorflow as tf
import os
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from preprocessing import normalisasi, train_data, train_label, test_data, test_label, label_encod
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPool2D, Flatten, GlobalMaxPooling2D, AveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam



# link path dan pemanggilan fungsi
train_dir=os.path.join("data/train")
testi_dir=os.path.join('data/test')

#pemaggilan fungsi
dt_train = train_data(train_dir)
dt_train_label = train_label(train_dir)
dt_test = test_data(testi_dir)
dt_test_label = test_label(testi_dir)

X_train, X_test = normalisasi(dt_train, dt_test)
y_train, y_test = label_encod(dt_test_label, dt_test_label)

# print("Training Data = ", dt_train.shape)
# print("Training Label = ", dt_train_label.shape)
# print("Testing Data = ", dt_test.shape)
# print("Testing Label = ", dt_test_label.shape)

# print(X_train)
# print(y_test)

ACCURACY_THRESHOLD = 0.92

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('val_accuracy') > ACCURACY_THRESHOLD):
			print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
			self.model.stop_training = True

# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model = Sequential()
model.add(Conv2D(input_shape=(250, 250, 3), filters=64,strides=1, kernel_size=(3, 3), activation='elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=512, activation='elu'))
model.add(Dense(units=128, activation='elu'))
model.add(Dense(units=1, activation='sigmoid'))
model.summary() 

best_model_gender_path = 'model_best.h5'
checkpoint_callback = ModelCheckpoint(best_model_gender_path,
                                     monitor='val_accuracy',
                                     save_best_only=True,
                                     verbose=1)
reduce_callback = ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, min_lr=0.000003, verbose=1)
Early_Stopping = myCallback()
callbacks_list = [checkpoint_callback, reduce_callback,Early_Stopping]

# Compile the model
learning_rate=0.000001
model.compile(loss='SparseCategoricalCrossentropy',
             optimizer=Adam(lr=learning_rate),
             metrics='accuracy')

# Fit the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test),callbacks=[callbacks_list])

