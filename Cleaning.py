import pandas as pd
from datetime import datetime
import glob
import os

# === 1. DATA CLEANING ===
def read_data(file):
    df = pd.read_csv(file)
    return df
path = "Data/"
df = pd.concat(map(read_data, glob.glob(os.path.join(path, "*_weather_data.csv"))))
df = df.reset_index()
coords_df = read_data(path + "country_coordinates.csv")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed', errors='coerce')
df['Sunshine_Duration'] = round((df['Sunshine_Duration'] / 3600), 2)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Country'] = df['Country'].replace({
    "USA": "United States",
    "UK": "United Kingdom"
})
continents = {'South America': ['Argentina', 'Brazil'],
              'Asia': ['China', 'India', 'Indonesia','Myanmar', 'Pakistan', 'Philippines', 'Russia', 'Thailand', 'Vietnam'],
              'North America': ['United States', 'Mexico'],
              'Europe': ['France', 'Italy', 'United Kingdom', 'Turkey', 'Spain'],
              'Africa': ['South_Africa', 'Tanzania', 'Nigeria']}
country_to_continent = {country: cont for cont, countries in continents.items() for country in countries}
df['Continent'] = df['Country'].map(country_to_continent)
df.drop(columns=['index'], inplace=True)

def get_season(date):
    y = date.year
    seasons = {'Summer':(datetime(y,6,21), datetime(y,9,22)),
               'Autumn':(datetime(y,9,23), datetime(y,12,20)),
               'Spring':(datetime(y,3,21), datetime(y,6,20))}
    for season,(season_start, season_end) in seasons.items():
        if (date >= season_start) and (date <= season_end):
            return season
    else:
        return 'Winter'

# Assuming df has a date column of type `datetime`

df['Season'] = df['Date'].map(get_season)
