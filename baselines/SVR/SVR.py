import numpy as np
from sklearn.svm import SVR
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd


def main(train_path, test_path, dataset_id):
    train_data = pd.read_pickle(train_path)

    test_data = pd.read_pickle(test_path)

    visual_feature_train = np.array(train_data['visual_feature_embedding_cls'].tolist())

    textual_feature_train = np.array(train_data['textual_feature_embedding'].tolist())

    X_train_ = np.concatenate((visual_feature_train, textual_feature_train), axis=1)

    Y_train_ = (np.array(train_data['label'].tolist()))

    visual_feature_test = np.array(test_data['visual_feature_embedding_cls'].tolist())

    textual_feature_test = np.array(test_data['textual_feature_embedding'].tolist())

    X_test_ = np.concatenate((visual_feature_test[:, 0], textual_feature_test), axis=1)

    Y_test_ = np.array(test_data['label'].tolist())

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_)

    X_test_scaled = scaler.transform(X_test_)

    svr_model = SVR(kernel='rbf')

    svr_model.fit(X_train_scaled, Y_train_)

    Y_pred = svr_model.predict(X_test_scaled)

    MAE = mean_absolute_error(Y_test_, Y_pred)

    SRC, _ = spearmanr(Y_pred, Y_test_)

    nMSE = np.mean(np.square(Y_pred - Y_test_)) / (Y_test_.std() ** 2)


if __name__ == "__main__":
    main(train_path=r'train.pkl',
         test_path=r'test.pkl',
         dataset_id='MicroLens-100k')
