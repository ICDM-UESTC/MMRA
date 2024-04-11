import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor


def main(alpha_lgbm, alpha_tabnet):
    train_data = pd.read_pickle(r'train.pkl')

    valid_data = pd.read_pickle(r'valid.pkl')

    test_data = pd.read_pickle(r'test.pkl')

    visual_feature_train = np.array(train_data['visual_feature_embedding_cls'].tolist())

    textual_feature_train = np.array(train_data['textual_feature_embedding'].tolist())

    X_train = np.concatenate((visual_feature_train.reshape(len(train_data), -1), textual_feature_train), axis=1)

    Y_train = np.array(train_data['label'].tolist())

    visual_feature_valid = np.array(valid_data['visual_feature_embedding_cls'].tolist())

    textual_feature_valid = np.array(valid_data['textual_feature_embedding'].tolist())

    X_valid = np.concatenate((visual_feature_valid.reshape(len(valid_data), -1), textual_feature_valid), axis=1)

    Y_valid = np.array(valid_data['label'].tolist())

    visual_feature_test = np.array(test_data['visual_feature_embedding_cls'].tolist())

    textual_feature_test = np.array(test_data['textual_feature_embedding'].tolist())

    X_test = np.concatenate((visual_feature_test.reshape(len(test_data), -1), textual_feature_test), axis=1)

    Y_test = np.array(test_data['label'].tolist())

    lgbm_model = LGBMRegressor()

    lgbm_model.fit(
        X_train, Y_train,
        eval_set=(X_valid, Y_valid),
    )

    tabnet_model = TabNetRegressor()

    tabnet_model.fit(
        X_train, Y_train.reshape(-1, 1),
        eval_set=[(X_valid, Y_valid.reshape(-1, 1))],
        patience=20,
        max_epochs=200
    )

    Y_pred_lgbm = lgbm_model.predict(X_test)

    Y_pred_tabnet = tabnet_model.predict(X_test)

    Y_pred_tabnet = Y_pred_tabnet.reshape(-1)

    Y_pred = alpha_lgbm * Y_pred_lgbm + alpha_tabnet * Y_pred_tabnet

    MAE = mean_absolute_error(Y_test, Y_pred)

    SRC, _ = spearmanr(Y_pred, Y_test)

    nMSE = np.mean(np.square(Y_pred - Y_test)) / (Y_test.std() ** 2)

    print(f"Alpha_LGBM : {alpha_lgbm}, Alpha_Tabnet : {alpha_tabnet}")

    print(f"Mean Absolute Error on Test Set: {MAE}")

    print(f"Spearman Correlation on Test Set: {SRC}")

    print(f"Normalized Mean Squared Error on Test Set: {nMSE}")



if __name__ == "__main__":
    main(0.5, 0.5)
