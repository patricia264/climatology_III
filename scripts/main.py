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

    # Display the combined DataFrame
    #print(daily_temperature_data)

    file_path = '../data/CAP9_reconstructions_1728-2020.csv'
    weather_types = pd.read_csv(file_path)
    # Convert the first column to DateTime format and rename it to 'date'
    weather_types.rename(columns={weather_types.columns[0]: 'date'}, inplace=True)
    weather_types['date'] = pd.to_datetime(weather_types['date'], format='%Y-%m-%d')

    #print(weather_types)

    # Initialize the LinearRegression class
    lr = LinearRegression()

    # Define reference period
    reference_start = pd.to_datetime('1760-01-01', format='%Y-%m-%d')
    reference_end = pd.to_datetime('1790-12-31', format='%Y-%m-%d')

    # Apply regression
    results = lr.apply_regression(daily_temperature_data, weather_types, reference_start,
                                  reference_end, combine_locations=True)

    # Printing coefficients with feature names
    print("Coefficients:")
    for name, coef in zip(results['feature_names'], results['coefficients']):
        print(f"{name}: {coef}")

    print("Intercept:", results['intercept'])
    print("Mean Squared Error:", results['mean_squared_error'])
