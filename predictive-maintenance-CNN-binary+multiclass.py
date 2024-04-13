#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


# Load the data
data = pd.read_csv('/Users/marinaelwy/Documents/Bachelor thesis/Machine Predictive Maintenance Classification by Shivam/predictive_maintenance.csv')

# Define features and target
X = data.drop(columns=['Target', 'Failure Type', "Product ID"])  # Features
# y = data['Target']
y_binary = data['Target']  # Binary target (failure or not)
y_multiclass = data['Failure Type']  # Multiclass target (type of failure)

features_num = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
features_cat = ["Type"]


# In[ ]:


# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# In[ ]:


# Preprocessing pipelines
transformer_num = make_pipeline(
    SimpleImputer(strategy="mean"),  # Impute missing values with the mean
    StandardScaler()  # Standardize numerical features
)

transformer_cat = make_pipeline(
    OneHotEncoder(handle_unknown='ignore')
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat)
)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)


# In[ ]:


# # Reshape the data for input into the CNN model
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)

# # Check the shape of the data
# print("Shape of X_train:", X_train.shape)
# print("Shape of X_valid:", X_valid.shape)


# In[ ]:


# Define the input shape
input_shape = (X_train.shape[1], 1)

# Define the shared layers
input_layer = layers.Input(shape=input_shape)
conv1d_layer1 = layers.Conv1D(32, 3, activation='relu')(input_layer)
maxpooling_layer1 = layers.MaxPooling1D(2)(conv1d_layer1)
conv1d_layer2 = layers.Conv1D(64, 3, activation='relu')(maxpooling_layer1)
flatten_layer = layers.Flatten()(conv1d_layer2)
dense_layer = layers.Dense(64, activation='relu')(flatten_layer)


# In[ ]:


# Define the binary classification output layer
binary_output = layers.Dense(1, activation='sigmoid', name='binary_output')(dense_layer)


# In[ ]:


# Define the multiclass classification output layer
num_failure_types = len(data['Failure Type'].unique())
multiclass_output = layers.Dense(num_failure_types, activation='softmax', name='multiclass_output')(dense_layer)


# In[ ]:


# Define the model
model = keras.Model(inputs=input_layer, outputs=[binary_output, multiclass_output])


# In[ ]:


# Compile the model with appropriate loss functions
model.compile(optimizer='adam',
              loss={'binary_output': 'binary_crossentropy', 'multiclass_output': 'sparse_categorical_crossentropy'},
              metrics=['accuracy'])


# In[ ]:


# Define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True
)


# In[ ]:


# Train the model
history = model.fit(X_train, {'binary_output': y_train_binary, 'multiclass_output': y_train_multiclass},
                    validation_data=(X_valid, {'binary_output': y_valid_binary, 'multiclass_output': y_valid_multiclass}),
                    batch_size=512,
                    epochs=200,
                    callbacks=[early_stopping])


# In[ ]:


# print(len(features_num))


# In[ ]:


# model = keras.Sequential([
#     # layers.Reshape((len(features_num)+1, 1), input_shape=(len(features_num)+1,)),
#     layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
#     layers.Conv1D(32, 3, activation='relu'),
#     layers.MaxPooling1D(2),
#     layers.Conv1D(64, 3, activation='relu'),
#     layers.MaxPooling1D(2),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
# Define the CNN model
# model = keras.Sequential([
#     layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
#     layers.MaxPooling1D(2),
#     layers.Conv1D(64, 3, activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])



# In[ ]:


# # Compile the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])


# In[ ]:


# # Train the model
# history = model.fit(X_train, y_train,
#                     validation_data=(X_valid, y_valid),
#                     batch_size=512,
#                     epochs=200,
#                     callbacks=[early_stopping])


# In[ ]:


# Plot training history
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy")

