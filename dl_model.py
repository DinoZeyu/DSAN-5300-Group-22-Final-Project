import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

df1 = pd.read_csv('Data/cleaned.csv')

# chi-squared selected feature
X=df1[['host_response_time', 'host_response_rate', 'host_is_superhost',
       'room_type', 'accommodates', 'minimum_nights', 'availability_30',
       'availability_60', 'availability_90', 'availability_365',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_value', 'instant_bookable', 'reviews_per_month',
       'host_years', 'review_years_range']]
Y=df1['popularity']

# sanity check
X.isna().sum()

# Split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_val_scaled = scaler.transform(x_val)

# Display the shapes of the training and testing data
x_train_scaled.shape, x_test_scaled.shape, x_val_scaled.shape,y_train.shape, y_test.shape,y_val.shape

y_train.unique()

class MyModel:
    def __init__(self, params, verbose=True):
        self.verbose = verbose
        for key, val in params.items():
            command = f'self.{key} = {val!r}' if isinstance(val, (float, int)) else f'self.{key} = "{val}"'
            if self.verbose: print(command)
            exec(command)

        num_classes = np.unique(y_train).shape[0]  # Assuming y_train is globally accessible

        # Dynamically build the model
        self.model = Sequential()
        self.model.add(Dense(self.nodes, activation=self.activation, kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2)))
        self.model.add(Dropout(self.dropout))
        if self.hidden_layers == 3:
            self.model.add(Dense(self.nodes, activation=self.activation, kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2)))
            self.model.add(Dropout(self.dropout))
        self.model.add(Dense(num_classes, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=50, batch_size=32):
        callbacks = EarlyStopping(monitor='val_loss', patience=5)
        self.history = self.model.fit(x_train_scaled, y_train, epochs=epochs, batch_size=batch_size,
                                      validation_data=(x_val_scaled, y_val), callbacks=[callbacks])

    def plot_confusion_matrix(self, y_true, y_pred, title):
        matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def evaluate_model(self):
        # Predict and evaluate
        y_train_pred = np.argmax(self.model.predict(x_train_scaled), axis=1)
        y_val_pred = np.argmax(self.model.predict(x_val_scaled), axis=1)
        y_test_pred = np.argmax(self.model.predict(x_test_scaled), axis=1)

        # Calculate accuracy
        train_loss, train_accuracy = self.model.evaluate(x_train_scaled, y_train, verbose=0)
        val_loss, val_accuracy = self.model.evaluate(x_val_scaled, y_val, verbose=0)
        test_loss, test_accuracy = self.model.evaluate(x_test_scaled, y_test, verbose=0)

        print('\nTraining Set Classification Report:\n', classification_report(y_train, y_train_pred))
        print('Training Accuracy: {:.4f}'.format(train_accuracy))
        print('\nValidation Set Classification Report:\n', classification_report(y_val, y_val_pred))
        print('Validation Accuracy: {:.4f}'.format(val_accuracy))
        print('\nTest Set Classification Report:\n', classification_report(y_test, y_test_pred))
        print('Test Accuracy: {:.4f}'.format(test_accuracy))

        self.plot_loss()
        self.plot_confusion_matrix(y_train, y_train_pred, 'Training Confusion Matrix')
        self.plot_confusion_matrix(y_val, y_val_pred, 'Validation Confusion Matrix')
        self.plot_confusion_matrix(y_test, y_test_pred, 'Test Confusion Matrix')

# Example usage:
params = {
    'nodes': 64,
    'activation': 'relu',
    'dropout': 0.2,
    'l1': 0.01,
    'l2': 0.01,
    'optimizer': 'adam',
    'hidden_layers': 2  # This should be a boolean now or handle this param differently.
}

model = MyModel(params)
model.train(epochs=100, batch_size=32)
model.evaluate_model()

params2 = {
    'nodes': 64,
    'activation': 'relu',
    'dropout': 0.2,
    'l1': 0,
    'l2': 0.01,
    'optimizer': 'adam',
    'hidden_layers': 3  # This should be a boolean now or handle this param differently.
}

model2 = MyModel(params2)
model2.train(epochs=100, batch_size=32)
model2.evaluate_model()

params3 = {
    'nodes': 64,
    'activation': 'relu',
    'dropout': 0.2,
    'l1': 0,
    'l2': 0.01,
    'optimizer': 'adam',
    'hidden_layers': 5
}

model3 = MyModel(params3)
model3.train(epochs=100, batch_size=32)
model3.evaluate_model()

params4 = {
    'nodes': 64,
    'activation': 'relu',
    'dropout': 0.2,
    'l1': 0,
    'l2': 0.01,
    'optimizer': 'adam',
    'hidden_layers': 4
}

model4 = MyModel(params4)
model4.train(epochs=100, batch_size=32)
model4.evaluate_model()

"""### best model so far

"""

optimal_params = {
    'nodes': 64,
    'activation': 'relu',
    'dropout': 0.2,
    'l1': 0,
    'l2': 0.01,
    'optimizer': 'adam',
    'hidden_layers': 4
}

optimal_model = MyModel(optimal_params)
optimal_model.train(epochs=100, batch_size=32)
optimal_model.evaluate_model()