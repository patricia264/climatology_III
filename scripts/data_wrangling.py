import pandas as pd

# Read a TSV file
df = pd.read_csv('../data/EarlyInstrumentalTemperature/Basel/1_daily/CHIMES_JU01_Delemont_18011222-18321230_ta_daily.tsv', sep='\t',
                 skiprows=12)

# Display the DataFrame
print(df)