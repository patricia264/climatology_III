import os
import pandas as pd
import matplotlib.pyplot as plt

class DataSelection:
    def __init__(self):
        """
        Initialize the DataSelection class with a folder path.

        Parameters:
            folder_path (str): Path to the folder containing the TSV files.
        """
        self.folder_path = None
        self.cleaned_df = None

    def read_data(self):
        """
        Reads TSV files from the initialized folder, processes the data, and cleans it.

        Returns:
            pd.DataFrame: Cleaned DataFrame containing combined data from all TSV files.
        """
        # List to store the dataframes
        dataframes = []

        # Loop through all files in the folder
        for file_name in os.listdir(self.folder_path):
            # Check if the file is a TSV file
            if file_name.endswith('.tsv'):
                # Construct the full file path
                file_path = os.path.join(self.folder_path, file_name)

                # Locate the header row containing "year"
                with open(file_path, 'r') as file:
                    for i, line in enumerate(file):
                        if "year" in line.lower():  # Assuming the header contains "year"
                            header_row = i
                            break

                # Read the file starting from the header row
                df = pd.read_csv(file_path, delimiter='\t', skiprows=header_row)

                # Optionally, add a column to track the source file
                df['source_file'] = file_name

                # Append the DataFrame to the list
                dataframes.append(df)

        # Combine all DataFrames into a single DataFrame
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Remove unwanted columns
        drop_cols = ['Hour', 'Minute', 'Period', 'Meta']
        cleaned_df = combined_df.drop(columns=drop_cols, errors='ignore')
        # Filter for winter months
        djf = [12, 1, 2]
        filtered_df = cleaned_df[cleaned_df['Month'].isin(djf)]
        # create new column which combines year, month, day
        filtered_df = filtered_df.copy()
        filtered_df['date'] = pd.to_datetime(filtered_df[['Year', 'Month', 'Day']])
        # group by date apply the mean
        grouped_df = filtered_df.groupby('date', as_index=False)['Value'].mean()

        return grouped_df

    def read_temp_locations(self, base_folder_path, locations):
        """
        Read and process data from multiple locations.
        Parameters:
            base_folder_path (str): Base path to the data folders.
            locations (list): List of location names.
        Returns:
            pd.DataFrame: A combined DataFrame with data from all locations.
        """
        # Initialize an empty list to store DataFrames
        all_locations = []

        # Loop through each location and process the data
        for location in locations:
            folder_path_daily = f'{base_folder_path}/{location}/1_daily/'
            print(f'Processing data in folder: {folder_path_daily}')

            # Initialize DataSelection for the specific location
            self.folder_path = folder_path_daily

            # Read the data
            cleaned_df_daily = self.read_data()

            # Add a new column for the location
            cleaned_df_daily['location'] = location

            # Append the DataFrame to the list
            all_locations.append(cleaned_df_daily)

        # Concatenate all the DataFrames into one
        final_dataframe = pd.concat(all_locations, ignore_index=True)
        return final_dataframe





