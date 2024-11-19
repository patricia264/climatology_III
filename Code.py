import pandas as pd
#monthly data

data_Basel = pd.read_csv('Basel_1755-1863_monthly_filled.csv')
data_Basel['Date'] = pd.to_datetime(data_Basel['Date'], format='%Y-%m')
Basel_filtered = data_Basel[(data_Basel['Date'] >= '1780-01') & (data_Basel['Date'] <= '1810-12')]

data_Bern = pd.read_csv('Bern_1760-1863_monthly_filled.csv')
data_Bern['Date'] = pd.to_datetime(data_Bern['Date'], format='%Y-%m')
Bern_filtered = data_Bern[(data_Bern['Date'] >= '1780-01') & (data_Bern['Date'] <= '1810-12')]

data_Geneva = pd.read_csv('Geneva_1768-1863_monthly_filled.csv')
data_Geneva['Date'] = pd.to_datetime(data_Geneva['Date'], format='%Y-%m')
Geneva_filtered = data_Geneva[(data_Geneva['Date'] >= '1780-01') & (data_Geneva['Date'] <= '1810-12')]

data_Plateau = pd.read_csv('SwissPlateau_1756-1863_monthly_filled.csv')
data_Plateau['Date'] = pd.to_datetime(data_Plateau['Date'], format='%Y-%m')
Plateau_filtered = data_Plateau[(data_Plateau['Date'] >= '1780-01') & (data_Plateau['Date'] <= '1810-12')]

data_Zurich = pd.read_csv('Zurich_1756-1863_monthly_filled.csv')
data_Zurich['Date'] = pd.to_datetime(data_Zurich['Date'], format='%Y-%m')
Zurich_filtered = data_Zurich[(data_Zurich['Date'] >= '1780-01') & (data_Zurich['Date'] <= '1810-12')]

print(Basel_filtered.head())
print(Bern_filtered.head())
print(Geneva_filtered.head())
print(Plateau_filtered.head())
print(Zurich_filtered.head())

# Daily Data
ddata_Ba = pd.read_csv('Basel_1755-1863_daily.csv')
ddata_Ba['Date'] = pd.to_datetime(ddata_Ba['Date'], format='%Y-%m-%d')
daily_Basel = ddata_Ba[(ddata_Ba['Date'] >= '1780-01-01') & (ddata_Ba['Date'] <= '1810-12-31')]

ddata_Be = pd.read_csv('Bern_1760-1863_daily.csv')
ddata_Be['Date'] = pd.to_datetime(ddata_Be['Date'], format='%Y-%m-%d')
daily_Bern = ddata_Be[(ddata_Be['Date'] >= '1780-01-01') & (ddata_Be['Date'] <= '1810-12-31')]

ddata_Ge = pd.read_csv('Geneva_1768-1863_daily.csv')
ddata_Ge['Date'] = pd.to_datetime(ddata_Ge['Date'], format='%Y-%m-%d')
daily_Geneva = ddata_Ge[(ddata_Ge['Date'] >= '1780-01-01') & (ddata_Ge['Date'] <= '1810-12-31')]

ddata_SP = pd.read_csv('SwissPlateau_1756-1863_daily.csv')
ddata_SP['Date'] = pd.to_datetime(ddata_SP['Date'], format='%Y-%m-%d')
daily_Plateau = ddata_SP[(ddata_SP['Date'] >= '1780-01-01') & (ddata_SP['Date'] <= '1810-12-31')]

ddata_Zu = pd.read_csv('Zurich_1756-1863_daily.csv')
ddata_Zu['Date'] = pd.to_datetime(ddata_Zu['Date'], format='%Y-%m-%d')
daily_Zurich = ddata_Zu[(ddata_Zu['Date'] >= '1780-01-01') & (ddata_Zu['Date'] <= '1810-12-31')]

print(daily_Basel.head())
print(daily_Bern.head())
print(daily_Geneva.head())
print(daily_Plateau.head())
print(daily_Zurich.head())

import matplotlib.pyplot as plt
# ------ Plotting the daily mean temperatures -------------

plt.figure(figsize=(12, 6))

# Plot temperature data for each location
plt.plot(daily_Basel['Date'], daily_Basel['Ta_mean'], label='Basel', color='blue', alpha=0.6)
plt.plot(daily_Bern['Date'], daily_Bern['Ta_mean'], label='Bern', color='orange', alpha=0.6)
plt.plot(daily_Geneva['Date'], daily_Geneva['Ta_mean'], label='Geneva', color='green', alpha=0.6)
plt.plot(daily_Plateau['Date'], daily_Plateau['Ta_mean'], label='Swiss Plateau', color='red', alpha=0.6)
plt.plot(daily_Zurich['Date'], daily_Zurich['Ta_mean'], label='Zurich', color='purple', alpha=0.6)

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