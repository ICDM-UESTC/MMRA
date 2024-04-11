import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    initial_data_frame = pd.read_pickle(r'data.pkl')

    train_data_frame, others_data_frame = train_test_split(initial_data_frame, test_size=0.2, random_state=9)

    test_data_frame, valid_data_frame = train_test_split(others_data_frame, test_size=0.5, random_state=18)

    train_data_frame.reset_index(drop=True, inplace=True)

    test_data_frame.reset_index(drop=True, inplace=True)

    valid_data_frame.reset_index(drop=True, inplace=True)

    train_data_frame.to_pickle(r'train.pkl')

    test_data_frame.to_pickle(r'test.pkl')

    valid_data_frame.to_pickle(r'valid.pkl')
