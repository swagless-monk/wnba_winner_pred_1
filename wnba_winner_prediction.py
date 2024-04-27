from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib

# Read in dataframes
season1 = pd.read_csv('E:\data\datasets\wnba_archive\_1997_2000_officialBoxScore.csv')
season2 = pd.read_csv('E:\data\datasets\wnba_archive\_2001_2010_officialBoxScore.csv')
season3 = pd.read_csv('E:\data\datasets\wnba_archive\_2011_2020_officialBoxScore.csv')

# Check for unique teams for all dataframes
wnba_teams1 = [season1['matchWinner'].unique()]
wnba_teams2 = [season2['matchWinner'].unique()]
wnba_teams3 = [season3['matchWinner'].unique()]

# Create mappings for string columns
home_away = {'Away':0, 'Home':1}
win_loss = {'Loss':0, 'Win':1}
wnba = {'HOU':0, 'NYL':1, 'SAC':2, 
        'PHO':3, 'LAS':4, 'CLE':5, 
        'CHA':6, 'UTA':7, 'WAS':8, 
        'DET':9, 'MIN':10, 'ORL':11, 
        'IND':12, 'MIA':13, 'POR':14, 
        'SEA':15, 'SAS':16, 'CON':17,
        'CHI': 18, 'ATL':19, 'TUL':20}
season_type = {'Regular':0, 'Playoffs':1}

# Merge dataframes
all_seasons = [season1, season2, season3]
season = pd.concat(all_seasons)

print(season)

# Drop dates and seasonType
season = season.drop(columns=['gmDate',])

# Convert string columns into numeric columns
season['matchWinner'] = season['matchWinner'].map(wnba)
season['teamLoc'] = season['teamLoc'].map(home_away)
season['opptLoc'] = season['opptLoc'].map(home_away)
season['teamAbbr'] = season['teamAbbr'].map(wnba)
season['opptAbbr'] = season['opptAbbr'].map(wnba)
season['teamRslt'] = season['teamRslt'].map(win_loss)
season['opptRslt'] = season['opptRslt'].map(win_loss)
season['seasonType'] = season['seasonType'].map(season_type)

# Split data in X and y
X = np.array(season.drop(columns=['matchWinner']))
y = np.array(season['matchWinner'])

# Drop null rows
X = np.nan_to_num(X)
y = np.nan_to_num(y)

model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

model.fit(X_train, y_train)

prediction = model.score(X_test, y_test)

print(prediction)

joblib.dump(model, 'wnba_prediction2.joblib')
