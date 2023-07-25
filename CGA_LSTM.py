import abc
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Input, Bidirectional, concatenate, Conv1D, MaxPool1D
from keras import Model
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from LossFunc import QuantileLoss
from GAT_layer import GAT
from keras.regularizers import l2
from keras.layers import Reshape


class QRRecurrentModel(object):
    # constructor
    def __init__(self, trainX1, trainY, trainX2, modelName, hyperParameters=None):
        self.trainX1 = trainX1
        self.trainX2 = trainX2
        self.trainY = trainY
        self.hyperParameters = hyperParameters
        self.depth = hyperParameters['depth']
        self.nodeNum = hyperParameters['nodeNum']
        self.quantiles = hyperParameters['quantiles']
        self.activation = hyperParameters['activation']
        self.dropout = hyperParameters['dropout']
        self.recurrentModel = None
        self.modeName = modelName

        # construct model based on hyperParameters

    @abc.abstractmethod
    def constructModel(self, **kwargs):
        pass

    # train model
    def fit(self, **kwargs):
        # Handle parameters
        # bs:  batch size
        # ce:  convergence epochs
        # ilr: initial learning rate
        # lrd: learning rate decay
        # lrm: learning rate minimum
        # me:  maximum number of epochs
        # ts:  split ratio of training set
        bs, ce, ilr, lrd, lrm, me, ts, kwargs = self.__fit_params__(kwargs)

        self.xMean1 = np.mean(self.trainX1, axis=0, keepdims=True)
        self.xSig1 = np.std(self.trainX1, axis=0, keepdims=True)
        self.xMean2 = np.mean(self.trainX2, axis=0, keepdims=True)
        self.xSig2 = np.std(self.trainX2, axis=0, keepdims=True)
        self.yMean = np.mean(self.trainY, axis=0, keepdims=True)
        self.ySig = np.std(self.trainY, axis=0, keepdims=True)
        n = self.trainX1.shape[0]
        nTrain = round(ts * n)

        xTrain1 = self.trainX1[:nTrain, :, :]
        xTrain2 = self.trainX2[:nTrain, :, :]
        yTrain = self.trainY[:nTrain, :]
        xValid1 = self.trainX1[nTrain:, :, :]
        xValid2 = self.trainX2[nTrain:, :, :]
        yValid = self.trainY[nTrain:, :]

        # xValid1 = self.trainX1[1000:1002, :, :]
        # xValid2 = self.trainX2[1000:1002, :, :]
        # yValid = self.trainY[1000:1002, :]

        xTrain1 = np.array(xTrain1)
        xTrain2 = np.array(xTrain2)

        xTrain1 = (xTrain1 - self.xMean1) / self.xSig1
        xTrain2 = (xTrain2 - self.xMean2) / self.xSig2
        xValid1 = (xValid1 - self.xMean1) / self.xSig1
        xValid2 = (xValid2 - self.xMean2) / self.xSig2

        yTrain = (yTrain - self.yMean) / self.ySig

        xTrain = [xTrain1, xTrain2]
        xValid = [xValid1, xValid2]

        self.trainX = xTrain
        self.trainY = yTrain
        self.validationX = xValid
        self.validationY = yValid

        # if self.modeName not in ['QRVMDCNNBILSTM(nwp)']:
        loss = QuantileLoss(self.quantiles)
        optimizer = tf.keras.optimizers.Adam(learning_rate=ilr)
        self.recurrentModel.compile(loss=loss, optimizer=optimizer)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=1 / lrd, patience=ce)
        callbacks = [early_stopping, lr_reducer]
        self.recurrentModel.fit(xTrain, yTrain, batch_size=32, epochs=200, validation_split=0.1, callbacks=callbacks)

    # predict for validationX
    def predict(self, isDic=False, validationX=None):
        if validationX is None:
            validationX = self.validationX
        # validationX = (validationX - self.xMean) / self.xSig
        # validationX = np.array(validationX)
        # validationX= np.reshape(validationX, [validationX.shape[1], validationX.shape[2], validationX.shape[3]])

        predictions = self.recurrentModel.predict(validationX)
        predictions = self.yMean + self.ySig*predictions
        predictions = np.array(predictions).reshape(-1, self.quantiles.shape[0])

        if isDic:
            # 采用字典的形式输出，'分为数':[x的预测值]
            dic = {}
            for i in range(self.quantiles.shape[0]):
                key = round(self.quantiles[i], 3)
                dic[self.quantiles[i]] = np.array(predictions[:, i]).reshape(-1,1)
            return dic
        return predictions

    # default parameters
    def __fit_params__(self, kwargs):
        batch_size = kwargs.pop("batch_size", 32)
        convergence_epochs = kwargs.pop("convergence_epochs", 10)
        initial_learning_rate = kwargs.pop('initial_learning_rate', 0.01)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.9)
        learning_rate_minimum = kwargs.pop('learning_rate_minimum', 1e-4)
        maximum_epochs = kwargs.pop("maximum_epochs", 100)
        training_split = kwargs.pop("training_split", 0.9)
        return batch_size, convergence_epochs, initial_learning_rate, \
            learning_rate_decay, learning_rate_minimum, maximum_epochs, \
            training_split, kwargs


class CGA_LSTM(QRRecurrentModel):

    def constructModel(self, **kwargs):
        input1 = Input(shape=(self.trainX1.shape[1], self.trainX1.shape[2]))
        l1 = LSTM(64, return_sequences=True, **kwargs)(input1)
        l1 = Dropout(self.dropout)(l1)
        l1 = LSTM(64, return_sequences=True, **kwargs)(l1)
        l1 = Dropout(self.dropout)(l1)
        l1 = LSTM(64, activation='relu')(l1)

        input2 = Input(shape=(self.trainX2.shape[1], self.trainX2.shape[2]))
        cnn = Conv1D(filters=16, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_uniform')(input2)
        cnn = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_uniform')(cnn)
        ga1 = GAT(32,
                  attn_heads=4,
                  attn_heads_reduction='average',
                  activation='relu',
                  kernel_regularizer=l2(5e-4/2),
                  attn_kernel_regularizer=l2(5e-4/2))(cnn)

        ga2 = GAT(64,
                  attn_heads=8,
                  attn_heads_reduction='average',
                  activation='softmax',
                  kernel_regularizer=l2(5e-4/2),
                  attn_kernel_regularizer=l2(5e-4/2))(ga1)
        l3 = Reshape((64, 11))(ga2)
        l3 = LSTM(64, return_sequences=True, **kwargs)(l3)
        l3 = Dropout(self.dropout)(l3)
        l3 = LSTM(64, activation='relu')(l3)

        added = concatenate([l1, l3], axis=1)
        # added = Flatten()(added)
        total_output = Dense(128, activation='relu')(added)
        total_output = Dense(64, activation='relu')(total_output)
        output = Dense(units=len(self.quantiles), activation=None)(total_output)
        self.recurrentModel = Model(inputs=[input1, input2], outputs=output)
