import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


class LinearRegressionPlots:

    def __init__(self):
        """
        Initialize the LinearRegressionPlots class.
        """
        self.scaler = StandardScaler()
        self.model = None

    def coefficients_bar_plot(self, results):
        coefficients = results['coefficients']
        p_values = results['p_values']

        # Convert to series for easier plotting
        coeff_series = coefficients[1:]  # Exclude the intercept
        p_series = p_values[1:]

        # Highlight significant coefficients
        significant = p_series < 0.05
        colors = np.where(significant, 'blue', 'grey')

        plt.bar(coeff_series.index, coeff_series, color=colors, alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel('Predictor Variables')
        plt.ylabel('Regression Coefficient')
        plt.title('Regression Coefficients and Significance')
        plt.xticks(rotation=45)
        plt.show()

    def predictions_reference_plot(self, results):
        y_ref = results['y_ref']
        y_predicted = results['y_predicted']

        # If y_predicted is an array, ensure it's converted to a scalar (if it contains a single value)
        if isinstance(y_predicted, np.ndarray) and y_predicted.size == 1:
            y_predicted = y_predicted.item()  # Extract the scalar value

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Calculate the histogram and normalize to percentages
        counts, bins, bars = plt.hist(
            y_ref, bins=15, alpha=0.7, color='sandybrown', edgecolor='sienna',
            weights=np.ones(len(y_ref)) / len(y_ref) * 100,  # Normalize counts to percentages
            label='Reference Temperatures (1780–1810)')

        # Add a vertical line for predicted temperature
        plt.axvline(y_predicted, color='red', linestyle='--', linewidth=2,
                    label=f'Predicted Temperature (1796): {y_predicted:.1f}°C')

        # Improve axis labels and title
        plt.xlabel('Temperature [°C]', fontsize=12)
        plt.ylabel('Frequency [%]', fontsize=12)  # Update y-axis label
        plt.title('Predicted Temperature vs. Reference Period', fontsize=14)

        # Ensure y-axis ticks are formatted as percentages
        max_percentage = int(np.ceil(counts.max()))  # Get the maximum percentage
        plt.yticks(range(0, max_percentage + 1, 10))  # Adjust step if needed

        # Customize ticks
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Remove grid lines overlapping with the bars
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Add legend with improved placement
        plt.legend(loc='upper left', fontsize=10)

        # Show the plot
        plt.tight_layout()
        plt.show()

    def plot_mse_across_reference_periods(self, evaluated_reference_periods, mse_results):
        """
        Plot the Mean Squared Error (MSE) across different reference periods.

        Parameters:
            evaluated_reference_periods (list of tuples): List of reference period (start_year, end_year) tuples.
            mse_results (list of floats): Corresponding MSE values for each reference period.
        """
        reference_periods = [f"{start}-{end}" for start, end in evaluated_reference_periods]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(reference_periods, mse_results, marker='o', linestyle='-', color='b', label='MSE')
        plt.xlabel('Reference Period')
        plt.ylabel('Mean Squared Error')
        plt.title('MSE Across Reference Periods')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_residuals_for_best_period(self, combined_data, best_start, best_end):
        """
        Plot residuals for the best reference period identified.

        Parameters:
            combined_data (pd.DataFrame): The combined data including weather types, temperature, and other predictors.
            best_start (int): Start year of the best reference period.
            best_end (int): End year of the best reference period.
        """
        # Filter data for the best reference period
        reference_mask = (combined_data['djf_year'] >= best_start) & (combined_data['djf_year'] <= best_end)
        reference_data = combined_data[reference_mask]

        # Features and target for the best reference period
        X_ref = reference_data.drop(columns=['djf_year', 'Value'])
        y_ref = reference_data['Value']

        # Convert all column names to strings before scaling
        X_ref.columns = X_ref.columns.astype(str)

        # Standardize features
        X_ref_scaled = self.scaler.fit_transform(X_ref)

        # Add constant for regression
        X_ref_with_const = sm.add_constant(X_ref_scaled, has_constant='add')

        # Predict using the trained model
        predicted_values = self.model.predict(X_ref_with_const)

        # Calculate residuals
        residuals = y_ref - predicted_values

        # Plot residuals
        plt.figure(figsize=(8, 6))
        plt.scatter(predicted_values, residuals, alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot for Best Reference Period: {best_start}-{best_end}')
        plt.grid(alpha=0.3)
        plt.show()
