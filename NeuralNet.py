import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Your script continues here...

# Load and prepare data
df_climate = pd.read_csv('climate_data.csv', parse_dates=['dt_iso']).set_index('dt_iso')
df_hurricane = pd.read_csv('hurricane_data.csv', parse_dates=['datetime']).set_index('datetime')
# Rename 'pressure' column in df_climate
df_climate = df_climate.rename(columns={'pressure': 'pressure_climate'})

# Rename 'pressure' column in df_hurricane
df_hurricane = df_hurricane.rename(columns={'pressure': 'pressure_hurricane'})

# Merge the two DataFrames on datetime
combined_df = pd.merge(df_climate.reset_index(), df_hurricane.reset_index(), 
                       left_on='dt_iso', right_on='datetime', 
                       how='inner').drop(columns=['datetime', 
                                                  'tropicalstorm_force_diameter', 
                                                  'hurricane_force_diameter', 'rain_1h', 'rain_3h'])




# Convert 'status' into a binary target for hurricane occurrence
combined_df['is_hurricane'] = combined_df['status'].apply(lambda x: 1 if x.lower() == 'hurricane' else 0)

# Drop 'status' and other non-relevant features from input features
X = combined_df.drop(['status', 'is_hurricane'], axis=1)

# Extract numerical features from 'X' and encode categorical features
X_numerical = X.select_dtypes(include=['float64', 'int64'])
X_categorical = pd.get_dummies(X.select_dtypes(include=['object']))

# Combine numerical and categorical features
X_processed = pd.concat([X_numerical, X_categorical], axis=1)

# Fill NaN values with the median of each column
X_processed = X_processed.fillna(X_processed.median())

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# Define the target variable
y = combined_df['is_hurricane']

# Splitting the dataset for the binary classification task
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Building and training the binary classification model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping], batch_size=32)
# Evaluate binary classification model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Binary Classification - Loss: {loss}, Accuracy: {accuracy}')


hurricanes_only_df = combined_df[combined_df['status'].str.lower() == 'hurricane']

# Use the 'category' column as your target variable
y = hurricanes_only_df['category'].values
X = hurricanes_only_df.drop(['status', 'category'], axis=1)  # Ensure 'category' is also dropped here, not just 'is_hurricane'

# Proceed with encoding categorical variables and scaling as before
X_numerical = X.select_dtypes(include=['float64', 'int64'])
X_categorical = pd.get_dummies(X.select_dtypes(include=['object']))
X_processed = pd.concat([X_numerical, X_categorical], axis=1)
X_processed = X_processed.fillna(X_processed.median())
X_scaled = scaler.fit_transform(X_processed)

# Now X_scaled is correctly aligned with y_multi

# Ensure 'category' is integer-typed
y = y.astype(int)

# Label encoding from 1-5 to 0-4 (necessary for to_categorical)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert 'category' column to one-hot encoded format
y_multi = to_categorical(y_encoded)

# Splitting the dataset for multi-class classification task
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_scaled, y_multi, test_size=0.2, random_state=42)

# Model adjustment for multi-class classification
model_multi = Sequential([
    Input(shape=(X_train_multi.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(y_train_multi.shape[1], activation='softmax')  # Output layer for multi-class
])

model_multi.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to mitigate overfitting
early_stopping_multi = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit the model
model_multi.fit(X_train_multi, y_train_multi, epochs=100, validation_split=0.2, callbacks=[early_stopping_multi], batch_size=32)

# Evaluate the model
loss_multi, accuracy_multi = model_multi.evaluate(X_test_multi, y_test_multi)
print(f'Multi-Class Classification - Loss: {loss_multi}, Accuracy: {accuracy_multi}')
