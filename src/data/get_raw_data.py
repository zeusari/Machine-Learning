from dotenv import load_dotenv, find_dotenv
import os
import requests
from requests import session
import logging

def extract_data(url, filepath):
    with session() as c:
        print(payload)
        c.post('https://www.kaggle.com/account/login', data=payload)
        with open(filepath, 'wb') as handle:
            response = c.get(url, stream=True)
            for block in response.iter_content(1024):
                handle.write(block)

def main(project_dir):

    raw_data_path = os.path.join(project_dir, 'data', 'raw')
    train_data_path = os.path.join(raw_data_path, 'train.csv')
    test_data_path = os.path.join(raw_data_path, 'test.csv')

    logger.info('Step 1')

    urltest='https://www.kaggle.com/c/3136/download/test.csv'
    urltrain='https://www.kaggle.com/c/3136/download/train.csv'

    extract_data(urltrain,train_data_path)
    logger.info('Step 2')
    extract_data(urltest,test_data_path)
    logger.info('Step 3')

if __name__ == '__main__':
    logger  = logging.getLogger(__name__)
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    # payload
    payload = {
        'action': 'login',
        'username': os.environ.get('KAGGLE_USER_NAME'),
        'password': os.environ.get('KAGGLE_PASSWORD')
    }
    main(project_dir)
