import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import pandas as pd

train_data = pd.read_pickle(r'train.pkl')

valid_data = pd.read_pickle(r'valid.pkl')

test_data = pd.read_pickle(r'test.pkl')

visual_feature_train = np.array(train_data['visual_feature_embedding_cls'].tolist())

textual_feature_train = np.array(train_data['textual_feature_embedding'].tolist())

X_train_ = np.concatenate((visual_feature_train.reshape(len(train_data), -1), textual_feature_train), axis=1)

Y_train_ = np.array(train_data['label'].tolist())

visual_feature_valid = np.array(valid_data['visual_feature_embedding_cls'].tolist())

textual_feature_valid = np.array(valid_data['textual_feature_embedding'].tolist())

X_valid_ = np.concatenate((visual_feature_valid.reshape(len(valid_data), -1), textual_feature_valid), axis=1)

Y_valid_ = np.array(valid_data['label'].tolist())

visual_feature_test = np.array(test_data['visual_feature_embedding_cls'].tolist())

textual_feature_test = np.array(test_data['textual_feature_embedding'].tolist())

X_test_ = np.concatenate((visual_feature_test.reshape(len(test_data), -1), textual_feature_test), axis=1)

Y_test_ = np.array(test_data['label'].tolist())

batch_size = 256

X_test_1 = X_test_[:batch_size, :]

Y_test_1 = Y_test_[:batch_size]

X_test_2 = X_test_[batch_size:batch_size * 2, :]

Y_test_2 = Y_test_[batch_size:batch_size * 2]

X_test_3 = X_test_[batch_size * 2:batch_size * 3, :]

Y_test_3 = Y_test_[batch_size * 2:batch_size * 3]

X_test_4 = X_test_[batch_size * 3:batch_size * 4, :]

Y_test_4 = Y_test_[batch_size * 3:batch_size * 4]

X_test_5 = X_test_[batch_size * 4:batch_size * 5, :]

Y_test_5 = Y_test_[batch_size * 4:batch_size * 5]

X_test_6 = X_test_[batch_size * 5:batch_size * 6, :]

Y_test_6 = Y_test_[batch_size * 5:batch_size * 6]

X_test_7 = X_test_[batch_size * 6:batch_size * 7, :]

Y_test_7 = Y_test_[batch_size * 6:batch_size * 7]

X_test_8 = X_test_[batch_size * 7:, :]

Y_test_8 = Y_test_[batch_size * 7:]

catboost_model = CatBoostRegressor(iterations=500, depth=10, learning_rate=0.03, loss_function='RMSE')

catboost_model.fit(
    X_train_, Y_train_,
    eval_set=(X_valid_, Y_valid_),
    early_stopping_rounds=20,
    verbose=100
)

Y_pred_1 = catboost_model.predict(X_test_1)

Y_pred_2 = catboost_model.predict(X_test_2)

Y_pred_3 = catboost_model.predict(X_test_3)

Y_pred_4 = catboost_model.predict(X_test_4)

Y_pred_5 = catboost_model.predict(X_test_5)

Y_pred_6 = catboost_model.predict(X_test_6)

Y_pred_7 = catboost_model.predict(X_test_7)

Y_pred_8 = catboost_model.predict(X_test_8)

MAE_1 = mean_absolute_error(Y_test_1, Y_pred_1)

MAE_2 = mean_absolute_error(Y_test_2, Y_pred_2)

MAE_3 = mean_absolute_error(Y_test_3, Y_pred_3)

MAE_4 = mean_absolute_error(Y_test_4, Y_pred_4)

MAE_5 = mean_absolute_error(Y_test_5, Y_pred_5)

MAE_6 = mean_absolute_error(Y_test_6, Y_pred_6)

MAE_7 = mean_absolute_error(Y_test_7, Y_pred_7)

MAE_8 = mean_absolute_error(Y_test_8, Y_pred_8)

MAE = (MAE_1 + MAE_2 + MAE_3 + MAE_4 + MAE_5 + MAE_6 + MAE_7 + MAE_8) / 8

SRC_1, _ = spearmanr(Y_pred_1, Y_test_1)

SRC_2, _ = spearmanr(Y_pred_2, Y_test_2)

SRC_3, _ = spearmanr(Y_pred_3, Y_test_3)

SRC_4, _ = spearmanr(Y_pred_4, Y_test_4)

SRC_5, _ = spearmanr(Y_pred_5, Y_test_5)

SRC_6, _ = spearmanr(Y_pred_6, Y_test_6)

SRC_7, _ = spearmanr(Y_pred_7, Y_test_7)

SRC_8, _ = spearmanr(Y_pred_8, Y_test_8)

SRC = (SRC_1 + SRC_2 + SRC_3 + SRC_4 + SRC_5 + SRC_6 + SRC_7 + SRC_8) / 8

nMSE_1 = np.mean(np.square(Y_pred_1 - Y_test_1)) / (Y_test_1.std() ** 2)

nMSE_2 = np.mean(np.square(Y_pred_2 - Y_test_2)) / (Y_test_2.std() ** 2)

nMSE_3 = np.mean(np.square(Y_pred_3 - Y_test_3)) / (Y_test_3.std() ** 2)

nMSE_4 = np.mean(np.square(Y_pred_4 - Y_test_4)) / (Y_test_4.std() ** 2)

nMSE_5 = np.mean(np.square(Y_pred_5 - Y_test_5)) / (Y_test_5.std() ** 2)

nMSE_6 = np.mean(np.square(Y_pred_6 - Y_test_6)) / (Y_test_6.std() ** 2)

nMSE_7 = np.mean(np.square(Y_pred_7 - Y_test_7)) / (Y_test_7.std() ** 2)

nMSE_8 = np.mean(np.square(Y_pred_8 - Y_test_8)) / (Y_test_8.std() ** 2)

nMSE = (nMSE_1 + nMSE_2 + nMSE_3 + nMSE_4 + nMSE_5 + nMSE_6 + nMSE_7 + nMSE_8) / 8

print(f"Mean Absolute Error on Test Set: {MAE}")

print(f"Spearman Correlation on Test Set: {SRC}")

print(f"Normalized Mean Squared Error on Test Set: {nMSE}")
