# Loss Function
import keras.backend as K


def skewed_absolute_error(y_true, y_pred, tau):
    dy = y_pred - y_true
    return K.mean((1.0 - tau) * K.relu(dy) + tau * K.relu(-dy), axis=-1)


def quantile_loss(y_true, y_pred, taus):
    y_pred = K.reshape(y_pred, shape=(-1, len(taus)))
    e = skewed_absolute_error(
        K.flatten(y_true), K.flatten(y_pred[:, 0]), taus[0])
    for i, tau in enumerate(taus[1:]):
        e += skewed_absolute_error(K.flatten(y_true),
                                   K.flatten(y_pred[:, i + 1]),
                                   tau)
    return e


class QuantileLoss:
    def __init__(self, quantiles):
        self.__name__ = "QuantileLoss"
        self.quantiles = quantiles

    def __call__(self, y_true, y_pred):
        return quantile_loss(y_true, y_pred, self.quantiles)

    def __repr__(self):
        return "QuantileLoss(" + repr(self.quantiles) + ")"
