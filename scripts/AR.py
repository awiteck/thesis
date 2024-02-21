from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt


class AR:
    def __init__(self, p):
        self.p = p
        self.model = LinearRegression()
        self.sigma = None

    def generate_train_x(self, data):
        # Number of rows in the output matrix
        rows = len(data) - self.p

        # Initialize the output matrix with ones
        # Each row will start with a 1, followed by p elements from the original array
        output_matrix = np.ones((rows, self.p + 1))

        for i in range(rows):
            # For each row, set the next p elements in reverse order starting from i+p
            output_matrix[i, 1:] = data[i : i + self.p][::-1]

        return output_matrix

    def generate_train_y(self, data):
        return data[self.p :]

    def fit(self, data):
        self.sigma = np.std(data)
        train_x = self.generate_train_x(data)
        train_y = self.generate_train_y(data)
        self.model.fit(train_x, train_y)

    def predict(self, data, num_preds):
        # Initialize the prediction set with the last p data points
        current_set = np.hstack(([1], data[-self.p :][::-1]))
        predictions = []

        for _ in range(num_preds):
            # Predict the next value
            next_pred = self.model.predict([current_set])[0]
            predictions.append(next_pred)

            # Update the current set for the next prediction
            current_set = np.hstack(([1], np.hstack((next_pred, current_set[1:-1]))))

        return predictions

    def validate(self, data, num_preds):
        rmses = []
        # Iterate through the data in steps of num_preds, starting from 0 up to len(data) - self.p - num_preds
        for i in range(0, len(data) - self.p - num_preds + 1):
            # The current chunk to predict on
            current_chunk = data[i : i + self.p]
            # The actual future values to compare against
            actual_values = data[i + self.p : i + self.p + num_preds]
            # Predict the next num_preds values
            predictions = self.predict(current_chunk, num_preds)
            # Calculate RMSE for the current set of predictions
            rmse = sqrt(mean_squared_error(actual_values, predictions))
            rmses.append(rmse)
        # Calculate the average RMSE
        average_rmse = np.mean(rmses)
        return average_rmse
