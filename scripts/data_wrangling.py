import os
import pandas as pd
import matplotlib.pyplot as plt

class DataSelection:
    def __init__(self, folder_path):
        """
        Initialize the DataSelection class with a folder path.

        Parameters:
            folder_path (str): Path to the folder containing the TSV files.
        """
        self.folder_path = folder_path
        self.cleaned_df = None  # Placeholder for the processed DataFrame

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
        self.cleaned_df = combined_df.drop(columns=drop_cols, errors='ignore')
        return self.cleaned_df

    def plot_values_by_source(self):
        """
        Plots values over time grouped by source files.

        Requires:
            The `read_data` method must be called first to populate `self.cleaned_df`.
        """
        if self.cleaned_df is None:
            raise ValueError("No data to plot. Please call read_data() first.")

        # Group data by 'source_file'
        grouped = self.cleaned_df.groupby('source_file')

        # Initialize the plot
        plt.figure(figsize=(20, 6))

        # Plot data for each source file
        for source, data in grouped:
            # Ensure data is sorted by year for consistent plotting
            data = data.sort_values('Year')

            # Plot 'Year' vs 'Value' for the current source file
            plt.plot(data['Year'], data['Value'], label=source)

        # Add labels, title, and legend
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.title('Values Over Time by Source File')
        plt.legend(title='Source File', loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.95, 1])

        # Show the plot
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Initialize the DataSelection class
    folder_path = '../data/EarlyInstrumentalTemperature/Basel/1_daily/'
    data_selector = DataSelection(folder_path)

    # Read and process the data
    cleaned_df = data_selector.read_data()

    # Plot the data
    data_selector.plot_values_by_source()
