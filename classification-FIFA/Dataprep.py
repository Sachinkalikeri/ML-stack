
import pandas as pd

data = pd.read_csv('../input/fifa-soccer-dataset/FIFA 2018 Statistics.csv')

# Dropping unnecessary columns
data = data.drop(['Date', 'Team', 'Opponent', 'Ball Possession %', 'Blocked', 'Offsides', 'Saves', 'Pass Accuracy %', 'Fouls Committed', 'Yellow Card', 'Yellow & Red', 'Red', '1st Goal', 'Round', 'PSO', 'Goals in PSO', 'Own goals', 'Own goal Time'], axis=1)

X = data[['Goal Scored', 'Attempts', 'On-Target', 'Off-Target', 'Corners', 'Free Kicks', 'Passes', 'Distance Covered (Kms)']]
y = data['Man of the Match']

