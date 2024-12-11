from data_wrangling import DataSelection
from linear_regression import LinearRegression
import pandas as pd

from scripts.data_merging import DataHelper
from scripts.plots import LinearRegressionPlots

if __name__ == "__main__":

    # Define the base path and the locations to iterate through
    base_folder_path = '../data/EarlyInstrumentalTemperature'
    locations = ['Basel', 'Bern', 'Zurich', 'Geneva']

    # Initialize the DataSelection class
    data_selector = DataSelection()

    # get temperature data
    djf_temp_data = data_selector.read_temp_locations(base_folder_path, locations)

    # get weather data
    file_path = '../data/CAP9_reconstructions_1728-2020.csv'
    weather_type_data = data_selector.process_weather_data(file_path)
    print("Weather data: ", weather_type_data)

    # read in CO2 data
    file_path = '../data/CO2_data.txt'
    co2_data = pd.read_csv(file_path, delimiter="\t")
    co2_data = co2_data[['YR', 'CO2[ppm]']]
    #print("CO2 data: ", co2_data)

    combined_data = DataHelper.combine_data(djf_temp_data, weather_type_data, co2_data)

    print("combined data: ", combined_data)

    # Initialize the LinearRegression class
    lr = LinearRegression()
    pl = LinearRegressionPlots()

    best_period = lr.find_best_reference_period(combined_data)
    print("Best Reference Start:", best_period['best_start'])
    print("Best Reference End:", best_period['best_end'])
    print("Lowest MSE:", best_period['lowest_mse'])
    print("Coefficients:", best_period['coefficients'])
    print("p-values:", best_period['p_values'])
    print("r-squared:", best_period['r_squared'])

    pl.coefficients_bar_plot(best_period)
    pl.predictions_reference_plot(best_period)
    evaluated_reference_periods = best_period['evaluated_reference_periods']
    mse_results = best_period['mse_results']
    pl.plot_mse_across_reference_periods(evaluated_reference_periods, mse_results)

    #pl.plot_residuals_for_best_period(combined_data, best_period['best_start'], best_period['best_end'])



