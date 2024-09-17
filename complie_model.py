import numpy as np
from keras.models import Sequential  # type: ignore
from keras.layers import LSTM, Dropout, Dense, Activation, Bidirectional # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from keras.optimizers import Adam  # type: ignore
import tensorflow as tf
# Load the data
X = np.load('X.npy').astype(np.float16)
y = np.load('y.npy').astype(np.float16)
note_to_int = np.load('note_to_int.npy', allow_pickle=True).item()
int_to_note = np.load('int_to_note.npy', allow_pickle=True).item()
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(len(note_to_int))) 
model.add(Activation('softmax'))
optimizer = Adam(clipnorm=1.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
early_stopping = EarlyStopping(monitor='loss', patience=5)  
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.00001)  
checkpoint = ModelCheckpoint('best_model.keras', monitor='loss', save_best_only=True, mode='min')  #
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(128).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
model.fit(dataset, epochs=10, callbacks=[early_stopping, reduce_lr, checkpoint])
model.save('MusicModdel.keras')
