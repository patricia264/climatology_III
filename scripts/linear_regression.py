from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd


class LinearRegression:
    def __init__(self):
        """
        Initialize the LinearRegression class.
        """
        self.model = SklearnLinearRegression()

    def apply_regression(self, dataset1, dataset2):
        """
        Combine two datasets and apply linear regression.

        Parameters:
            dataset1 (pd.DataFrame): First dataset containing temperature data.
            dataset2 (pd.DataFrame): Second dataset containing temperature data.

        Returns:
            dict: Regression results including model coefficients and mean squared error.
        """
        # Combine datasets
        combined_data = pd.merge(dataset1, dataset2, on='date', how='inner')

        # Ensure the date column is of datetime type
        combined_data['date'] = pd.to_datetime(combined_data['date'])

        # Drop rows with NaN in the target
        combined_data = combined_data.dropna(subset=['Value'])

        # Feature engineering: Encode 'location' column to numerical
        combined_data = pd.get_dummies(combined_data, columns=['location'], drop_first=True)

        print(combined_data)

        # Define target (temperature)
        y = combined_data['Value']

        # Define features: Use weather type and location columns only
        X = combined_data.drop(columns=['Value', 'date', 'prob'])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the linear regression model
        self.model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = self.model.predict(X_test)

        # Calculate the mean squared error
        mse = mean_squared_error(y_test, y_pred)

        # Return the model coefficients and MSE
        return {
            'feature_names': X.columns,
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_,
            'mean_squared_error': mse
        }

