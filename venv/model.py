import pandas as pd




# Load the CSV without parsing dates to inspect column names
climate_df = pd.read_csv('climatedata.csv')

# Print the columns to check for any discrepancies
print(climate_df.columns)

# Strip leading/trailing spaces from column names
climate_df.columns = climate_df.columns.str.strip()

# Now, set 'dt_iso' as a datetime column
climate_df['dt_iso'] = pd.to_datetime(climate_df['dt_iso'])


climate_df.set_index('dt_iso', inplace=True)

daily_climate_df = climate_df.resample('D').mean()

daily_climate_df[['lat', 'lon']] = climate_df[['lat', 'lon']].resample('D').first()
daily_climate_df.to_csv('dailyclimate.csv')
