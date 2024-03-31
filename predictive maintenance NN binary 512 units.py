#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer


# In[2]:


data=pd.read_csv('/Users/marinaelwy/Documents/Bachelor thesis/Machine Predictive Maintenance Classification by Shivam/predictive_maintenance.csv')
#print(data.head())
X = data.drop(columns=['Target', 'Failure Type',"Product ID"])  # Features
y = data['Target']
features_num = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
features_cat = ["Type"]

transformer_num = make_pipeline(
    SimpleImputer(strategy="mean"),  # Impute missing values with the mean
    StandardScaler()  # Standardize numerical features
)
# transformer_cat = make_pipeline(
#     SimpleImputer(strategy="constant", fill_value="NA"),
#     OneHotEncoder(handle_unknown='ignore'),
# )

transformer_cat = make_pipeline(
    OneHotEncoder(handle_unknown='ignore')
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
    # remainder="passthrough" ,
)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)
print("Shape of X_train after splitting:", X_train.shape)
print("Shape of X_valid after splitting:", X_valid.shape)

# X_train = preprocessor.fit_transform(X_train)
# X_valid = preprocessor.transform(X_valid)

# After preprocessing numerical features
X_train_num = transformer_num.fit_transform(X_train[features_num])
X_valid_num = transformer_num.transform(X_valid[features_num])
print("Shape of X_train_num after preprocessing numerical features:", X_train_num.shape)
print("Shape of X_valid_num after preprocessing numerical features:", X_valid_num.shape)

# After preprocessing categorical features
X_train_cat = transformer_cat.fit_transform(X_train[features_cat])
X_valid_cat = transformer_cat.transform(X_valid[features_cat])
print("Shape of X_train_cat after preprocessing categorical features:", X_train_cat.shape)
print("Shape of X_valid_cat after preprocessing categorical features:", X_valid_cat.shape)

print("Shape of X_train:", X_train.shape)
print("Shape of X_valid:", X_valid.shape)
print("input_shape",X_train.shape[1])

input_shape = [X_train.shape[1]]


# In[3]:


# Define a dictionary to map the categories to numeric values
type_mapping = {'L': 0, 'M': 1, 'H': 2}

# Map the values in the 'Type' column using the dictionary
X_train['Type'] = X_train['Type'].map(type_mapping)
X_valid['Type'] = X_valid['Type'].map(type_mapping)


# In[4]:


# Inspect categorical features in the training set
print("Unique values of categorical features in the training set:")
for feature in features_cat:
    print(f"Feature: {feature}")
    print(X_train[feature].unique())

# Inspect categorical features in the validation set
print("\nUnique values of categorical features in the validation set:")
for feature in features_cat:
    print(f"Feature: {feature}")
    print(X_valid[feature].unique())


# In[5]:


from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential([
    layers.BatchNormalization(input_shape=[7]),
    layers.Dense(units=512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(units=512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(1,activation='sigmoid')
    
])


# In[6]:


model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],)


# In[7]:


early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)



# In[8]:


import numpy as np
print("Data type of X_train:", X_train.values.dtype)
print("Data type of y_train:", y_train.values.dtype)
print("Data type of X_valid:", X_valid.values.dtype)
print("Data type of y_valid:", y_valid.values.dtype)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int32)
X_valid = X_valid.astype(np.float32)
y_valid = y_valid.astype(np.int32)


# In[9]:


history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
)


# In[10]:


history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")


# In[18]:


get_ipython().system('git')


# In[ ]:




