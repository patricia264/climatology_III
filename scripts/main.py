from data_wrangling import DataSelection
from linear_regression import LinearRegression
import pandas as pd


if __name__ == "__main__":

    # Define the base path and the locations to iterate through
    base_folder_path = '../data/EarlyInstrumentalTemperature'
    locations = ['Basel', 'Bern', 'Zurich', 'Geneva']

    # Initialize the DataSelection class
    data_selector = DataSelection()

    daily_temperature_data = data_selector.read_temp_locations(base_folder_path, locations)

    file_path = '../data/CAP9_reconstructions_1728-2020.csv'
    weather_types_to_filter = [3,5]
    weather_type_data = data_selector.process_weather_data(file_path, weather_types_to_filter)

    # Initialize the LinearRegression class
    lr = LinearRegression()

    # Define reference period
    reference_start = pd.to_datetime('1760-01-01', format='%Y-%m-%d')
    reference_end = pd.to_datetime('1790-12-31', format='%Y-%m-%d')

    # Apply regression
    results = lr.apply_regression(daily_temperature_data, weather_type_data, reference_start,
                                  reference_end, combine_locations=True)

    # Printing coefficients with feature names
    print("Coefficients:")
    for name, coef in zip(results['feature_names'], results['coefficients']):
        print(f"{name}: {coef}")

    print("Intercept:", results['intercept'])
    print("Mean Squared Error:", results['mean_squared_error'])
