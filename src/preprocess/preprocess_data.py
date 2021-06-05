from sklearn.preprocessing import StandardScaler, MinMaxScaler
from quple.components.data_preparation import prepare_train_val_test

# We apply 2 data preprocessors:
# 1. StandardScaler to standardize features by removing the mean and scaling to unit variance
# 2. MinMaxScaler to bound the data in the range [-1, +1]
preprocessors = [StandardScaler(), MinMaxScaler((-1, 1))]


def preprocess_data(ds_x, ds_y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, preprocessors=preprocessors):
    return prepare_train_val_test(ds_x, ds_y, train_size=train_ratio, val_size=val_ratio, test_size=test_ratio, preprocessors=preprocessors, shuffle=True)
