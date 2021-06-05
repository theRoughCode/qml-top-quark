from .data import load_data
from .preprocess import preprocess_data
import tensorflow as tf


def train(model, filepath, ranked_vars_filepath, batch_size=64, epochs=10, num_vars=9, num_samples=None):
    ds_x, ds_y = load_data.load_data(
        filepath, ranked_vars_filepath, num_vars, num_samples)

    x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data.preprocess_data(
        ds_x, ds_y)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=10, restore_best_weights=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[callback])
