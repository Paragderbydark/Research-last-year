import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Sample Data Preprocessing
def preprocess_data(X, y, max_length):
    # Pad sequences to ensure consistent input length
    X_padded = pad_sequences(X, maxlen=max_length, padding='post', value=0)
    
    # Reshape X for Conv1D input
    X_padded = X_padded.reshape((X_padded.shape[0], X_padded.shape[1], 1))
    
    return X_padded, y

# Define your data and labels
# X = ... # Your input data, should be a list or array of sequences
# y = ... # Your labels, should be a list or array of label sequences

# Example Data (Replace with actual data)
X = np.array([
    [0, 1, 0, 1, 1, 0],  # Example sequence 1
    [1, 0, 1, 0, 1, 0]   # Example sequence 2
])
y = np.array([
    [1, 1, 0, 0, 0, 1],  # Labels for sequence 1
    [0, 1, 0, 1, 0, 1]   # Labels for sequence 2
])

# Define the maximum sequence length
max_length = X.shape[1]

# Preprocess data
X_padded, y_padded = preprocess_data(X, y, max_length)

# Define the model
model = Sequential()

# Add convolutional layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(max_length, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))  # Dropout layer for regularization
model.add(Flatten())
model.add(Dense(units=max_length, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # Output layer with L2 regularization

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_padded, y_padded, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Example predictions (replace with actual test data)
# X_test = ... # Your test data
# X_test_padded, _ = preprocess_data(X_test, None, max_length)
# predictions = model.predict(X_test_padded)

# For evaluating predictions, you can compare with true labels
# y_test = ... # Your test labels
# evaluation = model.evaluate(X_test_padded, y_test)
