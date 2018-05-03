

import pandas as pd
import numpy as np
import os

raw_data_path = os.path.join(os.path.pardir, 'data', 'raw')
train_data_path = os.path.join(raw_data_path, 'train.csv')
test_data_path = os.path.join(raw_data_path, 'test.csv')

def read_data():
    # Set File Path
    raw_data_path = os.path.join(os.path.pardir, 'data', 'raw')
    train_data_path = os.path.join(raw_data_path, 'train.csv')
    test_data_path = os.path.join(raw_data_path, 'test.csv')

    # Read Data with index column as PassengerId
    train_df = pd.read_csv(train_data_path,index_col='PassengerId')
    test_df = pd.read_csv(test_data_path,index_col='PassengerId')

    # Mark all Blank Survival data from test dataset to -888.
    test_df['Survived'] = -888

    # Concat test and train data.
    df = pd.concat((train_df,test_df))

    # Return data set.
    return df

def getTitle(name):
    # Creating Map of different Titles.
    titie_group = {
        'mr' : 'Mr',
        'mrs' : 'Mrs',
        'miss' : 'Miss',
        'master' :'Master',
        'don' :'Sir',
        'rev' :'Sir',
        'dr' :'Officer',
        'mme' :'Mrs',
        'ms' :'Mrs',
        'major' :'Sir',
        'lady' :'Lady',
        'sir' :'Sir',
        'mlle' :'Miss',
        'col' :'Sir',
        'capt' :'Sir',
        'the countess' :'Lady',
        'jonkheer' :'Mr',
        'dona' :'Lady'
    }
    f_name_with_title = name.split(',')[1]
    title = f_name_with_title.split('.')[0]
    title = title.strip().lower()
    return titie_group[title]

def fill_missing_values(df):

    # Filling Missing Embarked value
    df.Embarked.fillna('C', inplace=True)

    # Calculating Median Fare and replacing with missing value
    median_fare = df.loc[(df.Pclass==3) & (df.Embarked=='S'),'Fare'].median()
    df.Fare.fillna(median_fare, inplace=True)

    # Calculating Median Age and replacing with missing value
    title_age_median = df.groupby(['Title']).Age.transform('median')
    df.Age.fillna(title_age_median, inplace=True)

    # Return data set.
    return df

def getDeck(Cabin):
    deck = np.where(pd.notnull(Cabin), str(Cabin)[0].upper(), 'Z')
    return deck

def reOrderColumns(df):
    columns = [column for column in df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    df = df [columns]
    return df

def write_data(df):
    processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
    write_train_path = os.path.join(processed_data_path, 'train.csv')
    write_test_path = os.path.join(processed_data_path, 'test.csv')

    # Write train_data without Survived Columns
    df.loc[df.Survived != -888].to_csv(write_train_path)

    # Write test_data
    columns = [column for column in df.columns if column != 'Survived']
    df.loc[df.Survived == -888, columns].to_csv(write_test_path)

def process_data(df):
    return (df
    .assign(Title = lambda x: x.Name.map(getTitle))
    .pipe(fill_missing_values)
    .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels=['very low', 'low', 'high', 'very high']))
    .assign(Age_State = lambda x: np.where(x.Age >= 18, 'Adults', 'Child'))
    .assign(Family_Size = lambda x: x.Parch + x.SibSp + 1)
    .assign(IsMother = lambda x: np.where(((x.Age > 18) & (x.Sex == 'female') & (x.Title != 'Miss') & (x.Parch > 0 )), 1,0))
    .assign(Cabin = lambda x: np.where(x.Cabin=='T', np.nan, x.Cabin))
    .assign(Deck = lambda x: x.Cabin.map(getDeck))
    .assign(isMale = lambda x: np.where(x.Sex == 'male', 1, 0))
    .pipe(pd.get_dummies, columns=['Embarked', 'Title', 'Age_State', 'Deck', 'Fare_Bin', 'Pclass'])
    .drop(['Cabin', 'Name', 'Parch', 'SibSp', 'Sex', 'Ticket'], axis=1)
    .pipe(reOrderColumns)
    )

if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)