import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logging

def read_data():
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    raw_data_path = os.path.join(project_dir, 'data', 'raw')
    train_data_path = os.path.join(raw_data_path, 'train.csv')
    test_data_path = os.path.join(raw_data_path, 'test.csv')

    train_df = pd.read_csv(train_data_path,index_col='PassengerId')
    test_df = pd.read_csv(test_data_path,index_col='PassengerId')

    female_passenger_first_class = train_df.loc[((train_df.Sex=='female') & (train_df.Pclass==1))]

    #train_df.info()
    train_df.describe()
    plt.boxplot(female_passenger_first_class.Age)

def plot_data():
    read_data()
    #train_df.info()

def main():
    logger.info('Step 1')
    plot_data()

if __name__ == '__main__':
    logger  = logging.getLogger(__name__)
    main()
