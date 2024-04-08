#!/usr/bin/env python
# coding: utf-8

# In[85]:


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
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


# In[86]:


# Load the data
data = pd.read_csv('/Users/marinaelwy/Documents/Bachelor thesis/Machine Predictive Maintenance Classification by Shivam/predictive_maintenance.csv')

# Define features and target
X = data.drop(columns=['Target', 'Failure Type', "Product ID"])  # Features
# y = data['Target']
y_binary = data['Target']  # Binary target (failure or not)
y_multiclass = data['Failure Type']  # Multiclass target (type of failure)

features_num = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
features_cat = ["Type"]


# In[87]:


# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# In[88]:


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


# In[89]:


# # Split the data into training and validation sets
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)

# Split the data into training and validation sets for binary classification
X_train, X_valid, y_train_binary, y_valid_binary = train_test_split(X, y_binary, stratify=y_binary, train_size=0.75)

# Split the data into training and validation sets for multiclass classification
_, _, y_train_multiclass, y_valid_multiclass = train_test_split(X, y_multiclass, stratify=y_multiclass, train_size=0.75)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Convert string labels to integer labels for multiclass classification
y_train_multiclass_encoded = label_encoder.fit_transform(y_train_multiclass)
y_valid_multiclass_encoded = label_encoder.transform(y_valid_multiclass)



# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)



# In[90]:


print("Shape of y_valid_multiclass_encoded:", y_valid_multiclass_encoded.shape)
print("Data type of y_valid_multiclass_encoded:", y_valid_multiclass_encoded.dtype)


# In[91]:


# Define the input shape
input_shape = (X_train.shape[1], 1)

# Define the shared layers
input_layer = layers.Input(shape=input_shape)
conv1d_layer1 = layers.Conv1D(32, 3, activation='relu')(input_layer)
maxpooling_layer1 = layers.MaxPooling1D(2)(conv1d_layer1)
conv1d_layer2 = layers.Conv1D(64, 3, activation='relu')(maxpooling_layer1)
flatten_layer = layers.Flatten()(conv1d_layer2)
dense_layer = layers.Dense(64, activation='relu')(flatten_layer)



# In[92]:


# Define the binary classification output layer
binary_output = layers.Dense(1, activation='sigmoid', name='binary_output')(dense_layer)


# In[93]:


# Define the multiclass classification output layer
num_failure_types = len(data['Failure Type'].unique())
multiclass_output = layers.Dense(num_failure_types, activation='softmax', name='multiclass_output')(dense_layer)



# In[94]:


# Define the model
model = keras.Model(inputs=input_layer, outputs=[binary_output, multiclass_output])



# In[95]:


# Compile the model with appropriate metrics for each output
model.compile(optimizer='adam',
              loss={'binary_output': 'binary_crossentropy', 'multiclass_output': 'sparse_categorical_crossentropy'},
              metrics={'binary_output': 'accuracy', 'multiclass_output': 'accuracy'})


# In[96]:


# Define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    patience=50,
    min_delta=0.001,
    restore_best_weights=True
)


# In[97]:


# Step 1: For Binary Classification
smote_binary = SMOTE()
X_train_resampled_binary, y_train_resampled_binary = smote_binary.fit_resample(X_train, y_train_binary)

# Step 2: For Multiclass Classification
smote_multiclass = {}
X_train_resampled_multiclass = {}
y_train_resampled_multiclass = {}

# Loop through each unique class label
for class_label in np.unique(y_train_multiclass_encoded):
    # Apply SMOTE to balance the dataset for each class label
    smote_multiclass[class_label] = SMOTE()
    X_train_resampled_multiclass[class_label], y_train_resampled_multiclass[class_label] = smote_multiclass[class_label].fit_resample(X_train, (y_train_multiclass_encoded == class_label))

# Combine resampled multiclass datasets
X_train_resampled_combined = np.concatenate([X_train_resampled_multiclass[label] for label in np.unique(y_train_multiclass_encoded)], axis=0)
y_train_resampled_combined = np.concatenate([y_train_resampled_multiclass[label] for label in np.unique(y_train_multiclass_encoded)], axis=0)


# Ensure both target arrays have the same number of samples
min_samples = min(X_train_resampled_combined.shape[0], len(y_train_resampled_binary), len(y_train_resampled_combined))
X_train_resampled_combined = X_train_resampled_combined[:min_samples]
y_train_resampled_binary = y_train_resampled_binary[:min_samples]
y_train_resampled_combined = y_train_resampled_combined[:min_samples]
print ("y_train_resampled_binary",y_train_resampled_binary.shape)
print (" y_train_resampled_combined", y_train_resampled_combined.shape)
# Train the model with the resampled data
history = model.fit(X_train_resampled_combined, {'binary_output': y_train_resampled_binary, 'multiclass_output': y_train_resampled_combined},
                    validation_data=(X_valid, {'binary_output': y_valid_binary, 'multiclass_output': y_valid_multiclass_encoded}),
                    batch_size=512,
                    epochs=200,
                    callbacks=[early_stopping])


# In[98]:


# # Train the model
# history = model.fit(X_train, {'binary_output': y_train_binary, 'multiclass_output': y_train_multiclass_encoded},
#                     validation_data=(X_valid, {'binary_output': y_valid_binary, 'multiclass_output': y_valid_multiclass_encoded}),
#                     batch_size=512,
#                     epochs=200,
#                     callbacks=[early_stopping])


