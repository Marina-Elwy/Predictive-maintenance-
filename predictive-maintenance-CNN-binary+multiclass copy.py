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
from sklearn.preprocessing import LabelEncoder


# In[2]:


# Load the data
data = pd.read_csv('/Users/marinaelwy/Documents/Bachelor thesis/Machine Predictive Maintenance Classification by Shivam/predictive_maintenance.csv')

# Define features and target
X = data.drop(columns=['Target', 'Failure Type', "Product ID"])  # Features
# y = data['Target']
y_binary = data['Target']  # Binary target (failure or not)
y_multiclass = data['Failure Type']  # Multiclass target (type of failure)

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


# In[27]:


print("Shape of y_valid_multiclass_encoded:", y_valid_multiclass_encoded.shape)
print("Data type of y_valid_multiclass_encoded:", y_valid_multiclass_encoded.dtype)


# In[5]:


# # Reshape the data for input into the CNN model
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)

# # Check the shape of the data
# print("Shape of X_train:", X_train.shape)
# print("Shape of X_valid:", X_valid.shape)


# In[6]:


# Define the input shape
input_shape = (X_train.shape[1], 1)

# Define the shared layers
input_layer = layers.Input(shape=input_shape)
conv1d_layer1 = layers.Conv1D(32, 3, activation='relu')(input_layer)
maxpooling_layer1 = layers.MaxPooling1D(2)(conv1d_layer1)
conv1d_layer2 = layers.Conv1D(64, 3, activation='relu')(maxpooling_layer1)
flatten_layer = layers.Flatten()(conv1d_layer2)
dense_layer = layers.Dense(64, activation='relu')(flatten_layer)


# In[7]:


# Define the binary classification output layer
binary_output = layers.Dense(1, activation='sigmoid', name='binary_output')(dense_layer)


# In[8]:


# Define the multiclass classification output layer
num_failure_types = len(data['Failure Type'].unique())
multiclass_output = layers.Dense(num_failure_types, activation='softmax', name='multiclass_output')(dense_layer)


# In[9]:


# Define the model
model = keras.Model(inputs=input_layer, outputs=[binary_output, multiclass_output])


# In[10]:


# Compile the model with appropriate metrics for each output
model.compile(optimizer='adam',
              loss={'binary_output': 'binary_crossentropy', 'multiclass_output': 'sparse_categorical_crossentropy'},
              metrics={'binary_output': 'accuracy', 'multiclass_output': 'accuracy'})


# In[11]:


# Define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True
)


# In[12]:


# Train the model
history = model.fit(X_train, {'binary_output': y_train_binary, 'multiclass_output': y_train_multiclass_encoded},
                    validation_data=(X_valid, {'binary_output': y_valid_binary, 'multiclass_output': y_valid_multiclass_encoded}),
                    batch_size=512,
                    epochs=200,
                    callbacks=[early_stopping])


# In[13]:


# print(len(features_num))


# In[14]:


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



# In[15]:


# # Compile the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])


# In[16]:


# # Train the model
# history = model.fit(X_train, y_train,
#                     validation_data=(X_valid, y_valid),
#                     batch_size=512,
#                     epochs=200,
#                     callbacks=[early_stopping])


# In[17]:


# # Plot training history
# history_df = pd.DataFrame(history.history)
# history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
# history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy")


# In[28]:


# Evaluate the model on validation data for binary classification
binary_losses_and_metrics = model.evaluate(X_valid, y_valid_binary)

# Extract relevant values for binary classification
binary_losses = binary_losses_and_metrics[0]
binary_accuracy = binary_losses_and_metrics[1]

print("Validation Binary Loss:", binary_losses)
print("Validation Binary Accuracy:", binary_accuracy)




# In[35]:


import numpy as np

# Reshape X_valid to match the expected input shape
X_valid_reshaped = np.expand_dims(X_valid, axis=-1)

# Check the new shape
print("Shape of X_valid after reshaping:", X_valid_reshaped.shape)
print("Expected input shape:", model.input_shape)



# In[33]:


print("Expected input shape:", model.input_shape)
print("Shape of X_valid:", X_valid.shape)


# In[36]:


# Check the shape of y_valid_multiclass_encoded
print("Shape of y_valid_multiclass_encoded:", y_valid_multiclass_encoded.shape)

# Check the unique values in y_valid_multiclass_encoded
unique_labels = np.unique(y_valid_multiclass_encoded)
print("Unique labels in y_valid_multiclass_encoded:", unique_labels)

# Check the number of unique labels
num_unique_labels = len(unique_labels)
print("Number of unique labels:", num_unique_labels)


# In[34]:


# Evaluate the model on validation data for multiclass classification
multiclass_losses_and_metrics = model.evaluate(X_valid_reshaped, y_valid_multiclass_encoded)

# Extract relevant values for multiclass classification
multiclass_losses = multiclass_losses_and_metrics[0]
multiclass_accuracy = multiclass_losses_and_metrics[1]

print("Validation Multiclass Loss:", multiclass_losses)
print("Validation Multiclass Accuracy:", multiclass_accuracy)


# In[ ]:





# In[ ]:




