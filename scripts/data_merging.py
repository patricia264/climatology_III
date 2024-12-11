import pandas as pd
class DataHelper:
    """
    A helper class for data processing and merging.
    """
    @staticmethod
    def combine_data(temp_data, wt_data, co2_data):
        """
        Combines the given datasets by merging on the specified keys and cleans the resulting dataframe.

        Parameters:
            temp_data (pd.DataFrame): DataFrame containing temperature data with 'djf_year' as a key.
            wt_data (pd.DataFrame): DataFrame containing weather type data with 'djf_year' as a key.
            co2_data (pd.DataFrame): DataFrame containing CO2 data with 'YR' as a key.

        Returns:
            pd.DataFrame: The cleaned and merged dataframe.
        """
        # Merge temp_data and wt_data on 'djf_year'
        combined_data = pd.merge(temp_data, wt_data, on='djf_year', how='inner')

        # Merge co2_data using 'djf_year' from combined_data and 'YR' from co2_data
        combined_data = pd.merge(combined_data, co2_data, left_on='djf_year', right_on='YR', how='inner')

        # Drop the extra 'YR' column from the final result
        combined_data.drop(columns=['YR'], inplace=True)

        # Drop rows with NaN in the target column 'Value'
        combined_data = combined_data.dropna(subset=['Value'])

        return combined_data