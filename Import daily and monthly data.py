import os
import pandas as pd
import matplotlib.pyplot as plt

def load_and_filter_data(base_path, locations, subdirectory, file_suffix_pattern, date_col, date_format, start_date,
                         end_date):
    """
    Load and filter data for specified locations.

    Parameters:
        base_path (str): The base directory containing the data.
        locations (list): List of location names (e.g., ['Basel', 'Bern']).
        subdirectory (str or None): Subdirectory containing the data (e.g., '5_filled' or '4_merged'). None for locations without subdirectories.
        file_suffix_pattern (str): File suffix pattern with placeholders (e.g., '_{years}_monthly_filled.csv').
        date_col (str): Column name for date.
        date_format (str): Format of the date column in the file.
        start_date (str): Start date for filtering (e.g., '1780-01').
        end_date (str): End date for filtering (e.g., '1810-12').

    Returns:
        dict: A dictionary with location names as keys and filtered DataFrames as values.
    """
    filtered_data = {}

    for location in locations:
        # Get the year range based on the location
        if location == "Basel":
            years = "1755-1863"
        elif location == "Bern":
            years = "1760-1863"
        elif location == "Geneva":
            years = "1768-1863"
        elif location == "SwissPlateau":
            years = "1756-1863"
        elif location == "Zurich":
            years = "1756-1863"
        else:
            raise ValueError(f"Unknown location: {location}")

        # Construct the file path
        file_name = f"{location}_{years}{file_suffix_pattern}"
        if location == "SwissPlateau":
            file_path = os.path.join(base_path, location, file_name)
        else:
            file_path = os.path.join(base_path, location, subdirectory, file_name)

        # Load the data
        data = pd.read_csv(file_path)

        # Convert the date column to datetime
        data[date_col] = pd.to_datetime(data[date_col], format=date_format)

        # Filter the data by date range
        #filtered_data[location] = data[(data[date_col] >= start_date) & (data[date_col] <= end_date)]
        filtered_data[location] = data

    return filtered_data


# Example usage
base_path = "data/EarlyInstrumentalTemperature"
locations = ["Basel", "Bern", "Geneva", "SwissPlateau", "Zurich"]

# Load monthly data
monthly_data = load_and_filter_data(
    base_path=base_path,
    locations=locations,
    subdirectory="5_filled",
    file_suffix_pattern="_monthly_filled.csv",
    date_col="Date",
    date_format="%Y-%m",
    start_date="1780-01",
    end_date="1810-12"
)

# Load daily data
daily_data = load_and_filter_data(
    base_path=base_path,
    locations=locations,
    subdirectory="4_merged",
    file_suffix_pattern="_daily.csv",
    date_col="Date",
    date_format="%Y-%m-%d",
    start_date="1780-01-01",
    end_date="1810-12-31"
)

# Access filtered data
print(monthly_data["SwissPlateau"].head())
print(daily_data["Zurich"].head())

# %%
# ------ Plotting the daily mean temperatures -------------
plt.figure(figsize=(12, 6))

# Colors for each location
colors = {
    "Basel": "blue",
    "Bern": "orange",
    "Geneva": "green",
    "SwissPlateau": "red",
    "Zurich": "purple"
}

# Plot temperature data for each location
for location, data in daily_data.items():
    plt.plot(data['Date'], data['Ta_mean'], label=location, color=colors[location], alpha=0.6)

# Highlight the year 1796
plt.axvspan(pd.Timestamp('1796-01-01'), pd.Timestamp('1796-12-31'), color='yellow', alpha=0.3, label='1796')

# Labels and title
plt.xlabel('Date')
plt.ylabel('Mean Temperature (°C)')
plt.title('Daily Mean Temperature (1780-1810) for Swiss Locations')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'data/CAP9_reconstructions_1728-2020.csv'
weather_types = pd.read_csv(file_path)
# Convert the first column to DateTime format and rename it to 'date'
weather_types.rename(columns={weather_types.columns[0]: 'date'}, inplace=True)
weather_types['date'] = pd.to_datetime(weather_types['date'], format='%Y-%m-%d')

import pandas as pd
import matplotlib.pyplot as plt


def weather_type_frequency_analysis(
        weather_types,
        months,
        analysis_start,
        analysis_end,
        baseline_start,
        baseline_end,
        title_suffix="Weather Type Frequencies"
):
    """
    Perform a frequency analysis of weather types for a specific period and compare it
    to a climatological baseline.

    Parameters:
        weather_types (pd.DataFrame): DataFrame containing the weather type data with a 'date' column.
        months (list): List of months to filter (e.g., [11, 12, 1, 2, 3, 4] for Nov-Apr).
        analysis_start (str): Start date for the analysis period (e.g., '1795-11-01').
        analysis_end (str): End date for the analysis period (e.g., '1796-04-30').
        baseline_start (int): Start year for the climatological baseline period.
        baseline_end (int): End year for the climatological baseline period.
        title_suffix (str): Suffix for plot titles (e.g., "Weather Type Frequencies").

    Returns:
        pd.DataFrame: Comparison of frequencies for the analysis period and climatology.
    """

    # Filter for specified months
    filtered_weather_types = weather_types[weather_types['date'].dt.month.isin(months)]

    # Filter for analysis period
    analysis_data = filtered_weather_types[
        (filtered_weather_types['date'] >= analysis_start) &
        (filtered_weather_types['date'] <= analysis_end)
        ]

    # Frequency analysis for the analysis period
    freq_analysis = analysis_data['WT'].value_counts(normalize=True) * 100

    # Filter for climatological baseline period
    baseline_data = filtered_weather_types[
        (filtered_weather_types['date'].dt.year >= baseline_start) &
        (filtered_weather_types['date'].dt.year <= baseline_end)
        ]

    # Frequency analysis for the climatological baseline period
    freq_baseline = baseline_data['WT'].value_counts(normalize=True) * 100

    # Combine results into a comparison DataFrame
    comparison = pd.DataFrame({
        f'{analysis_start[:4]}/{analysis_end[:4]} (%)': freq_analysis,
        f'{baseline_start}-{baseline_end} (%)': freq_baseline
    }).fillna(0)  # Fill missing weather types with 0 frequency

    # Print the comparison table
    print(comparison)

    # Optional: Visualize the comparison
    comparison.plot(kind='bar', figsize=(10, 6), alpha=0.7)
    plt.title(f'{title_suffix}: {analysis_start[:4]}-{analysis_end[:4]} vs {baseline_start}-{baseline_end}')
    plt.xlabel('Weather Type')
    plt.ylabel('Frequency (%)')
    plt.legend(title='Period')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return comparison

weather_type_frequency_analysis(
    weather_types=weather_types,
    months=[1, 2, 3],
    analysis_start='1796-01-01',
    analysis_end='1796-03-31',
    baseline_start=1728,
    baseline_end=2020,
    title_suffix="Weather Type Frequencies (Jan-Mar)"
)
weather_type_frequency_analysis(
    weather_types=weather_types,
    months=[12, 1, 2],
    analysis_start='1795-12-01',
    analysis_end='1796-02-28',
    baseline_start=1728,
    baseline_end=2020,
    title_suffix="Weather Type Frequencies (Dec-Feb)"
)
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Merge weather types with daily temperature data and filter for winter months
combined_data = {}
for location, temp_data in daily_data.items():
    # Add a 'month' column to the temperature data for filtering
    temp_data['month'] = pd.to_datetime(temp_data['Date']).dt.month

    # Merge and filter for winter months
    combined_data[location] = temp_data.merge(
        weather_types[['date', 'WT']],
        left_on='Date',
        right_on='date',
        how='inner'
    )
    combined_data[location] = combined_data[location][combined_data[location]['month'].isin([12, 1, 2])]

# Compute average temperature by weather type for each location (winter months only)
mean_temp_by_wt = {}
for location, data in combined_data.items():
    mean_temp_by_wt[location] = data.groupby('WT')['Ta_mean'].mean()

# Create a combined DataFrame for visualization
combined_mean_temp = pd.DataFrame(mean_temp_by_wt)

# Plot the mean temperatures by weather type
combined_mean_temp.plot(kind='bar', figsize=(12, 6), alpha=0.7)
plt.title('Average Winter Temperature by Weather Type for Swiss Locations')
plt.xlabel('Weather Type')
plt.ylabel('Mean Temperature (°C)')
plt.legend(title='Location')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt


def analyze_weather_type_temperature(daily_data, weather_data, months_of_interest, period_name):
    """
    Analyze the relationship between weather types and temperatures across Switzerland.

    Parameters:
        daily_data (dict): Dictionary of DataFrames with daily temperature data for locations.
        weather_data (DataFrame): DataFrame containing weather types with date information.
        months_of_interest (list): List of months (integers) to include in the analysis (e.g., [11, 12, 1, 2, 3, 4]).
        period_name (str): Description of the period for use in plot titles (e.g., 'Winter', 'Extended Winter').

    Returns:
        None. Displays plots of the results.
    """
    # Filter weather data to include only the months of interest
    weather_filtered = weather_data[weather_data['date'].dt.month.isin(months_of_interest)]

    # Combine temperatures across all locations into one DataFrame
    temperature_data = pd.concat(
        [
            daily_data[location][['Date', 'Ta_mean']].rename(columns={'Ta_mean': location})
            for location in daily_data.keys()
        ],
        axis=1
    )
    temperature_data = temperature_data.loc[:, ~temperature_data.columns.duplicated()]

    # Merge with weather types on the date
    temperature_data['Date'] = pd.to_datetime(temperature_data['Date'])
    merged_data = weather_filtered.merge(temperature_data, left_on='date', right_on='Date')

    # Calculate mean temperature across all locations for each weather type
    merged_data['Mean_Temp'] = merged_data[daily_data.keys()].mean(axis=1)
    mean_temps_by_weather_type = merged_data.groupby('WT')['Mean_Temp'].mean()

    # Sort weather types in numerical order (1 through 9)
    mean_temps_by_weather_type = mean_temps_by_weather_type.reindex(range(1, 10), fill_value=0)

    # Plot the mean temperatures by weather type
    plt.figure(figsize=(10, 6))
    mean_temps_by_weather_type.plot(kind='bar', color='skyblue', alpha=0.8)
    plt.title(f'Mean Temperatures by Weather Type ({period_name})')
    plt.xlabel('Weather Type')
    plt.ylabel('Mean Temperature (°C)')
    plt.xticks(ticks=range(0, 9), labels=range(1, 10), rotation=0)  # Ensures correct labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Analyze the frequency of weather types
    freq_weather_types = weather_filtered['WT'].value_counts(normalize=True) * 100
    freq_weather_types = freq_weather_types.reindex(range(1, 10), fill_value=0)

    # Print results
    print(f"Mean Temperatures by Weather Type ({period_name}):")
    print(mean_temps_by_weather_type)
    print(f"\nFrequency of Weather Types ({period_name}) (%):")
    print(freq_weather_types)


# Example usage
months_of_interest_djf = [12, 1, 2]  # Meteorological winter
months_of_interest_boreal = [11, 12, 1, 2, 3, 4]  # Extended boreal winter

print("DJF Results:")
analyze_weather_type_temperature(daily_data, weather_types, months_of_interest_djf, "Winter")

print("\nBoreal Winter Results:")
analyze_weather_type_temperature(daily_data, weather_types, months_of_interest_boreal, "Extended Winter")

# %%
def plot_temperature_comparison(daily_data, start_year, end_year, winter_start, winter_end):
    """
    Plot the mean temperatures for the extended winter period (Nov-Apr) of a specific year
    and the climatological mean for the same months in a reference period.

    Parameters:
        daily_data (dict): Dictionary of daily temperature DataFrames for locations.
        start_year (int): Start year of the reference period for climatology.
        end_year (int): End year of the reference period for climatology.
        winter_start (str): Start date of the specific winter (e.g., '1795-11-01').
        winter_end (str): End date of the specific winter (e.g., '1796-04-30').

    Returns:
        None. Displays the plot.
    """
    import numpy as np

    # Combine temperature data for Basel, Bern, Geneva, Zurich (skip SwissPlateau)
    selected_locations = ["Basel", "Bern", "Geneva", "Zurich"]
    combined_data = pd.concat(
        [daily_data[loc][['Date', 'Ta_mean']] for loc in selected_locations]
    )

    # Convert to datetime and filter data for winter period
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])
    combined_data['Month'] = combined_data['Date'].dt.month
    combined_data['Year'] = combined_data['Date'].dt.year

    # Specific winter (e.g., 1795-1796)
    specific_winter = combined_data[
        (combined_data['Date'] >= winter_start) & (combined_data['Date'] <= winter_end)
    ]
    specific_winter['Month'] = specific_winter['Date'].dt.month
    specific_winter_mean = specific_winter.groupby('Month')['Ta_mean'].mean()

    # Climatological mean for the same months (e.g., 30 years before)
    climatology = combined_data[
        (combined_data['Year'] >= start_year) & (combined_data['Year'] <= end_year) &
        (combined_data['Month'].isin([11, 12, 1, 2, 3, 4]))
    ]
    climatology['Month'] = climatology['Date'].dt.month
    climatology_mean = climatology.groupby('Month')['Ta_mean'].mean()

    # Ensure proper month order
    months = [11, 12, 1, 2, 3, 4]
    specific_winter_mean = specific_winter_mean.reindex(months, fill_value=np.nan)
    climatology_mean = climatology_mean.reindex(months, fill_value=np.nan)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'],
        specific_winter_mean,
        label='Winter 1795-1796',
        color='blue',
        marker='o',
    )
    plt.plot(
        ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'],
        climatology_mean,
        label=f'Climatology ({start_year}-{end_year})',
        color='orange',
        linestyle='--',
        marker='o',
    )

    # Labels and title
    plt.title('Temperature Comparison: Winter 1795-1796 vs Climatology')
    plt.xlabel('Month')
    plt.ylabel('Mean Temperature (°C)')
    plt.legend(title='Period')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_temperature_comparison(
    daily_data=daily_data,
    start_year=1765,
    end_year=1794,
    winter_start="1795-11-01",
    winter_end="1796-04-30",
)

# %%
def compute_and_plot_winter_temperatures(daily_data, statistic='mean', time_interval=None):
    """
    Compute and plot winter (DJF) temperatures for each location based on the selected statistic.

    Parameters:
        daily_data (dict): Dictionary of DataFrames with daily temperature data for locations.
        statistic (str): Statistic to compute and plot ('mean', 'max', 'min').
        time_interval (tuple): Optional tuple specifying the start and end years for the plot (e.g., (1800, 1850)).

    Returns:
        winter_df (DataFrame): DataFrame with the selected winter (DJF) statistic temperatures for each location.
    """
    if statistic not in ['mean', 'max', 'min']:
        raise ValueError("Invalid statistic. Choose 'mean', 'max', or 'min'.")

    winter_stats = {}

    # Loop through each location and compute the desired DJF statistic per year
    for location, data in daily_data.items():
        # Convert Date column to datetime if not already
        data['Date'] = pd.to_datetime(data['Date'])

        # Extract years and filter for December, January, February
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month

        # Shift December to the next year's winter (e.g., Dec 1795 -> Winter 1796)
        data.loc[data['Month'] == 12, 'Year'] += 1

        # Filter for winter months and group by year
        winter_data = data[data['Month'].isin([12, 1, 2])]
        stats = winter_data.groupby('Year')['Ta_mean'].agg(['mean', 'max', 'min'])

        winter_stats[location] = stats

    # Combine all locations into a single DataFrame
    winter_df = pd.concat(winter_stats, axis=1)
    winter_df.columns = pd.MultiIndex.from_tuples(
        [(loc, stat) for loc, df in winter_stats.items() for stat in df.columns],
        names=['Location', 'Statistic']
    )

    # Extract the desired statistic for plotting
    plot_data = winter_df.xs(statistic, level='Statistic', axis=1)

    # Apply time interval if specified
    if time_interval:
        start_year, end_year = time_interval
        plot_data = plot_data.loc[start_year:end_year]

    # Plotting
    plt.figure(figsize=(12, 6))
    for location in plot_data.columns:
        plt.plot(plot_data.index, plot_data[location], label=location)

    # Labels, title, and legend
    interval_str = f" ({time_interval[0]}–{time_interval[1]})" if time_interval else ""
    plt.title(f'{statistic.capitalize()} Winter (DJF) Temperatures Over Time{interval_str}')
    plt.xlabel('Year')
    plt.ylabel(f'{statistic.capitalize()} Temperature (°C)')
    plt.legend(title='Location')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Return the DataFrame with all statistics for further analysis
    return winter_df


# Example usage
# Compute and plot mean winter temperatures
winter_df_mean = compute_and_plot_winter_temperatures(daily_data, statistic='mean')
winter_df_mean_selected = compute_and_plot_winter_temperatures(daily_data, statistic='mean', time_interval=(1760,1800))

# Compute and plot max winter temperatures
winter_df_max = compute_and_plot_winter_temperatures(daily_data, statistic='max', time_interval=(1760, 1800))

# Compute and plot min winter temperatures
winter_df_min = compute_and_plot_winter_temperatures(daily_data, statistic='min', time_interval=(1760, 1800))

# Identify the years where the mean winter temperature is above 2°C
years_above_2 = winter_df[winter_df['Mean_All_Locations'] > 2].index.tolist()

# Print the years with mean temperatures above 2°C
print("Years with mean winter temperature above 2°C:")
print(years_above_2)

# %%
import pandas as pd
import matplotlib.pyplot as plt

def weather_type_frequency_analysis(weather_types, years, months):
    """
    Perform frequency analysis of weather types for the specified years and months.

    Parameters:
        weather_types (DataFrame): The weather type data.
        years (list): List of years to analyze.
        months (list): List of months to filter for (e.g., [12, 1, 2] for DJF).

    Returns:
        result (DataFrame): A DataFrame with weather type frequencies for each year.
    """
    weather_type_frequencies = {}

    # Loop over the specified years
    for year in years:
        # Filter the data for the selected year and months
        data_filtered = weather_types[(weather_types['date'].dt.year == year) &
                                      (weather_types['date'].dt.month.isin(months))]

        # Perform the frequency analysis for weather types
        weather_counts = data_filtered['WT'].value_counts(normalize=True) * 100  # Percentages
        weather_type_frequencies[year] = weather_counts

    # Convert the result to a DataFrame for better visualization
    result = pd.DataFrame(weather_type_frequencies).T  # Transpose so years are rows
    return result

def plot_weather_type_frequency_analysis(weather_types, years_above_2, months=[12, 1, 2], clim_start=1728, clim_end=2020):
    """
    Plot the frequency analysis of weather types for each year in the years_above_2 list,
    along with the climatological reference values (average over entire climatological period).

    Parameters:
        weather_types (DataFrame): The weather type data.
        years_above_2 (list): List of years with mean winter temperature above 2°C.
        months (list): List of months to filter for (e.g., [12, 1, 2] for DJF).
        clim_start (int): Start year of the climatological baseline period.
        clim_end (int): End year of the climatological baseline period.

    Returns:
        None: Displays the plot.
    """
    # Perform the frequency analysis for the warm winters (above 2°C)
    freq_analysis_df = weather_type_frequency_analysis(weather_types, years_above_2, months)

    # Perform the climatological frequency analysis (using the full baseline period)
    clim_data = weather_types[(weather_types['date'].dt.year >= clim_start) &
                              (weather_types['date'].dt.year <= clim_end)]
    clim_freq_analysis_df = weather_type_frequency_analysis(clim_data, range(clim_start, clim_end + 1), months)
    # Calculate the climatological mean frequencies across all years in the baseline period
    climatological_mean = clim_freq_analysis_df.mean(axis=0)  # Mean across all years

    # Add the climatological mean as a new column for plotting
    climatological_mean.name = 'Climatological Mean'
    print(climatological_mean)
    freq_analysis_df.loc['Climatological Mean'] = climatological_mean
    print(freq_analysis_df)
    # Plot the results
    plt.figure(figsize=(14, 8))

    # Plot the frequency for the years above 2°C (stacked bar)
    ax = freq_analysis_df.plot(kind='bar', stacked=True, figsize=(12, 6), width=0.8, ax=plt.gca(), alpha=0.7)

    # Update the x-ticks to show years and add the climatological bar as the last one
    years_above_2_with_clim = years_above_2 + ['Climatological Mean']
    ax.set_xticks(range(len(years_above_2_with_clim)))
    ax.set_xticklabels(years_above_2_with_clim, rotation=45)

    # Labels, title, and legend
    plt.title(f'Weather Type Frequencies for Warm Winters (Above 2°C) with Climatological Reference')
    plt.xlabel('Year')
    plt.ylabel('Frequency (%)')

    plt.legend(title='Weather Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example usage (assuming you have your weather_types DataFrame and the years_above_2 list)
plot_weather_type_frequency_analysis(weather_types, years_above_2)
