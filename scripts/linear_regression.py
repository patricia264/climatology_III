from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold

class LinearRegression:
    def __init__(self):
        """
        Initialize the LinearRegression class.
        """
        self.model = SklearnLinearRegression()

    def apply_regression(self, dataset1, dataset2, reference_start, reference_end, combine_locations = False):
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
        combined_data['date'] = pd.to_datetime(combined_data['date'], format='%Y-%m-%d')

        # Drop rows with NaN in the target
        combined_data = combined_data.dropna(subset=['Value'])

        # combine all locations and take the mean daily temperature
        if combine_locations == True:
            combined_data['mean_daily_temp'] = combined_data.groupby('date')['Value'].transform('mean')
            combined_data = combined_data.drop(columns=['location', 'Value'])
            combined_data.rename(columns={'mean_daily_temp': 'Value'}, inplace=True)
        else:
            # Feature engineering: Encode 'location' column to numerical
            combined_data = pd.get_dummies(combined_data, columns=['location'], drop_first=True)

        # Split data into reference period and non-reference period
        reference_mask = (combined_data['date'] >= reference_start) & (combined_data['date'] <= reference_end)

        ## Features reference period: weather types and locations
        X_reference = combined_data[reference_mask]
        print(X_reference)
        X_ref = X_reference.drop(columns=['Value', 'date', 'prob'])
        # Scale reference period features
        scaler = StandardScaler()
        X_ref_scaled = scaler.fit_transform(X_ref)

        ## Target (temperature) reference period
        y_reference = combined_data[reference_mask]
        y_ref = y_reference['Value']

        ## Features other period: weather types and locations
        X_other = combined_data[~reference_mask]
        X = X_other.drop(columns=['Value', 'date', 'prob'])
        # Scale other period features
        X_scaled = scaler.transform(X)

        ## Target (temperature) other period
        y_other = combined_data[~reference_mask]
        y = y_other['Value']

        # Train the regression model on the reference period
        self.model.fit(X_ref_scaled, y_ref)

        # Apply the model to the other period
        y_pred = self.model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)

        # Calculate MSE for reference period
        #y_reference_pred = self.model.predict(X_reference)
        #mse_reference = mean_squared_error(y_reference, y_reference_pred)

        # Return the model coefficients, intercept, and predictions for the non-reference period
        return {
            'feature_names': X.columns,
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_,
            'mean_squared_error': mse,
            'predicted_non_reference': y_pred,
            'actual_non_reference': y
        }

