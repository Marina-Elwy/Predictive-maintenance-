#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# Load the data
data = pd.read_csv('/Users/marinaelwy/Documents/Bachelor thesis/Machine Predictive Maintenance Classification by Shivam/predictive_maintenance.csv')

# Define features and target
X = data.drop(columns=['Target', 'Failure Type', "Product ID"])  # Features
y = data['Target']
features_num = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
features_cat = ["Type"]


# In[3]:


# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# In[4]:


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


# In[5]:


# Reshape the data for input into the CNN model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)

# Check the shape of the data
print("Shape of X_train:", X_train.shape)
print("Shape of X_valid:", X_valid.shape)


# In[6]:


print(len(features_num))


# In[7]:


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
model = keras.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])



# In[8]:


# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[9]:


# Define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True
)


# In[10]:


# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    batch_size=512,
                    epochs=200,
                    callbacks=[early_stopping])


# In[11]:


# Plot training history
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy")

