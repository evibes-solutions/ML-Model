# ML-Model

# steps to improve the mae

# do the following changes to the Building the model

from keras.layers import BatchNormalization

agemodel = Sequential()
agemodel.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
agemodel.add(BatchNormalization())
agemodel.add(MaxPooling2D((2,2)))

agemodel.add(Conv2D(64, (3,3), activation='relu'))
agemodel.add(BatchNormalization())
agemodel.add(MaxPooling2D((2,2)))

agemodel.add(Conv2D(128, (3,3), activation='relu'))
agemodel.add(BatchNormalization())
agemodel.add(MaxPooling2D((2,2)))

# agemodel.add(Conv2D(256, (3,3), activation='relu')) # Added an extra convolutional layer

agemodel.add(BatchNormalization())
agemodel.add(MaxPooling2D((2,2)))

agemodel.add(Flatten())

# agemodel.add(Dense(128, activation='relu')) # Increased dense layer size

agemodel.add(Dropout(0.5))
agemodel.add(Dense(64, activation='relu'))
agemodel.add(Dense(1, activation='relu'))

# agemodel.compile(loss='mean_absolute_error', # Changed loss function to MAE directly

                 optimizer=optimizers.Adam(learning_rate=0.0001))
