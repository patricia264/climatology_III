from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn import datasets, linear_model
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        """
        Initialize the LinearRegression class.
        """
        self.model = None
        self.scaler = StandardScaler()

    def calibrated_regression(self, combined_data, reference_start, reference_end, target_year=1796):
        """
        Perform a calibrated regression to estimate winter temperatures for a specific year.

        Parameters:
            temp_data (pd.DataFrame): Temperature data with 'djf_year' and 'Value' columns.
            wt_data (pd.DataFrame): Weather type frequency data with 'djf_year', 'f1', ..., 'f9'.
            co2_data (pd.DataFrame): CO2 data with 'djf_year' and 'CO2' columns.
            reference_start (int): Start year of the reference period.
            reference_end (int): End year of the reference period.
            target_year (int): Year for which to estimate winter temperature.

        Returns:
            dict: Regression coefficients, p-values, R-squared, and predicted temperature for target year.
        """

        # Define reference period mask
        reference_mask = (combined_data['djf_year'] >= reference_start) & (combined_data['djf_year'] <= reference_end)

        # Split data into reference and target year
        reference_data = combined_data[reference_mask]
        target_data = combined_data[combined_data['djf_year'] == target_year]

        # Features and target for reference period
        X_ref = reference_data.drop(columns=['djf_year', 'Value'])
        y_ref = reference_data['Value']

        # Features for target year
        X_target = target_data.drop(columns=['djf_year', 'Value'])
        y_target = target_data['Value']

        # Convert all column names to strings before scaling
        X_ref.columns = X_ref.columns.astype(str)
        X_target.columns = X_target.columns.astype(str)

        # Standardize features
        X_ref_scaled = self.scaler.fit_transform(X_ref)
        X_target_scaled = self.scaler.transform(X_target)

        # Add constant for regression
        X_ref_with_const = sm.add_constant(X_ref_scaled, has_constant='add')
        X_target_with_const = sm.add_constant(X_target_scaled, has_constant='add')

        # Train regression model
        self.model = sm.OLS(y_ref, X_ref_with_const).fit()

        # Predict temperature for target year
        y_predicted = self.model.predict(X_target_with_const)

        mse = mean_squared_error(y_target, y_predicted)

        return {
            'mean_squared_error': mse,
            'coefficients': self.model.params,
            'p_values': self.model.pvalues,
            'r_squared': self.model.rsquared,
            'predicted_temperature': y_predicted,
            'reference_temperature': y_ref,
        }

    def find_best_reference_period(self, combined_data):
        start_year = 1760
        end_year = 1790
        max_end_year = 1860

        best_start = None
        best_end = None
        lowest_mse = float('inf')  # Start with a very high value

        # Initialize lists to store reference periods and MSEs
        evaluated_reference_periods = []
        mse_results = []

        while end_year <= max_end_year:
            # Format the dates as strings
            reference_start = start_year
            reference_end = end_year

            # Apply regression and get results
            results = self.calibrated_regression(combined_data, reference_start, reference_end)

            # Extract MSE
            mse = results['mean_squared_error']

            # Collect the current period and its MSE
            evaluated_reference_periods.append((reference_start, reference_end))
            mse_results.append(mse)

            # Check for the best MSE
            if mse < lowest_mse:
                lowest_mse = mse
                best_start = reference_start
                best_end = reference_end
                coefficients = results['coefficients']
                p_values = results['p_values']
                r_squared = results['r_squared']
                y_predicted = results['predicted_temperature']
                y_ref = results['reference_temperature']

            # Increment the range by 10 years
            start_year += 10
            end_year += 10

        # Add results to the return dictionary
        return {
            'best_start': best_start,
            'best_end': best_end,
            'lowest_mse': lowest_mse,
            'coefficients': coefficients,
            'p_values': p_values,
            'r_squared': r_squared,
            'y_predicted': y_predicted,
            'y_ref': y_ref,
            'evaluated_reference_periods': evaluated_reference_periods,
            'mse_results': mse_results
        }



