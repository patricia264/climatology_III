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
        filtered_data[location] = data[(data[date_col] >= start_date) & (data[date_col] <= end_date)]

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

# Load the dataset
file_path = 'data/CAP9_reconstructions_1728-2020.csv'
weather_types = pd.read_csv(file_path)

# Convert the first column to DateTime format and rename it to 'date'
weather_types.rename(columns={weather_types.columns[0]: 'date'}, inplace=True)
weather_types['date'] = pd.to_datetime(weather_types['date'], format='%Y-%m-%d')

# Filter for the first three months of the year (Jan, Feb, Mar)
weather_types_first_quarter = weather_types[weather_types['date'].dt.month.isin([1, 2, 3])]

# Filter for the year 1796 and the first three months
weather_1796_q1 = weather_types_first_quarter[weather_types_first_quarter['date'].dt.year == 1796]

# Frequency analysis for 1796 (January, February, March)
freq_1796_q1 = weather_1796_q1['WT'].value_counts(normalize=True) * 100  # Percentages

# Frequency analysis for the climatological period (entire dataset or a baseline period)
clim_start, clim_end = 1728, 2020  # Define baseline period
clim_data_q1 = weather_types_first_quarter[
    (weather_types_first_quarter['date'].dt.year >= clim_start) &
    (weather_types_first_quarter['date'].dt.year <= clim_end)
]
freq_clim_q1 = clim_data_q1['WT'].value_counts(normalize=True) * 100  # Percentages

# Combine results into a comparison DataFrame
comparison_q1 = pd.DataFrame({
    '1796 (%)': freq_1796_q1,
    f'{clim_start}-{clim_end} (%)': freq_clim_q1
}).fillna(0)  # Fill missing weather types with 0 frequency

# Print the comparison table
print(comparison_q1)

# Optional: Visualize the comparison
comparison_q1.plot(kind='bar', figsize=(10, 6), alpha=0.7)
plt.title('Weather Type Frequencies (Jan-Mar): 1796 vs Climatology')
plt.xlabel('Weather Type')
plt.ylabel('Frequency (%)')
plt.legend(title='Period')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
