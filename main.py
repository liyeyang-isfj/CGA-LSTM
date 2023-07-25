import numpy as np
import pandas as pd
import time
import os
from GAT_LSTM import GAT_LSTM1
from CGA_LSTM import CGA_LSTM
from datetime import datetime
from Evaluation import Evaluation
from Draw import Draw
from scipy import stats
import math
from KernelDensityEstimation import KernelDensityEstimation


quantiles = np.arange(0.005, 1, 0.005)
step = 11
ahead = 4
start = 0
end = 2400


df = pd.read_excel(r"D:\PycharmProjects\5.Energy2审修改\CGA-LSTM\DATA\duochangzhan.xlsx")

# input1
data1 = df.loc[:, ('风速', '风速1', '风速2', '风速3', '风速4', '风速5', '风速6', '风速7', '风速8', '风速9', '风速10')].values

x1_train = []
for i in np.arange(end + 20):
    x1_train.append(data1[i:i + step, :])
x1_train = np.array(x1_train)
x1_train = x1_train[:end, :]
x1_train = x1_train.reshape([x1_train.shape[0], x1_train.shape[1], x1_train.shape[2]])
x12_train = x1_train.transpose(0, 2, 1)

# input2 + outputWP
data2 = df.loc[:, ('输出功率')].values

x2_train = []
y_train = []
for i in np.arange(end+20):
    x2_train.append(data2[i:i + step])
    y_train.append(data2[i + step + ahead - 1])
x2_train = np.array(x2_train)
x2_train = x2_train[:end, :]
x2_train = x2_train.reshape([x2_train.shape[0], x2_train.shape[1], 1])

y_train = np.array(y_train)
y_train = y_train[:end]
y_train = y_train.reshape([y_train.shape[0], 1])

# inputA
A = np.array(np.ones((x1_train.shape[0], x1_train.shape[0])))


isDrawResults = True
isDrawPIT = True
isDrawPDF = True
isSave = True
isShow = False

resultSaveBasePath = '../results/'
plotSaveBasePath = '../plots/'
modelSaveBasePath = '../model_save/'
timeFlag = str(int(time.time()))

metricSavePath = resultSaveBasePath + timeFlag + '/metrics_' + timeFlag + '.xlsx'
metricWriter = pd.ExcelWriter(metricSavePath)
metrics = []
metricIndex = []

resultSavePath = resultSaveBasePath + timeFlag + '/dataset_results_' + timeFlag + '.xlsx'
resultWriter = pd.ExcelWriter(resultSavePath)

observationSavePath = resultSaveBasePath + timeFlag + '/dataset_observations_' + timeFlag + '.xlsx'
observationsWriter = pd.ExcelWriter(observationSavePath)

plotSavePath = plotSaveBasePath + timeFlag + '/dataset_'

if not os.path.exists(resultSaveBasePath + timeFlag):
    os.makedirs(resultSaveBasePath + timeFlag)
    os.makedirs(plotSaveBasePath + timeFlag)

modelNames = ['GAT-LSTM', 'CGA-LSTM']

for modelName in modelNames:
    hyperParameters = dict()
    hyperParameters['depth'] = 3
    hyperParameters['nodeNum'] = 32
    hyperParameters['quantiles'] = quantiles
    hyperParameters['activation'] = 'tanh'
    hyperParameters['dropout'] = 0.2

    # model
    if modelName == 'GAT-LSTM':
        model = GAT_LSTM1(x2_train, y_train, x12_train, modelName, hyperParameters)
    elif modelName == 'CGA-LSTM':
        model = CGA_LSTM(x2_train, y_train, x12_train, modelName, hyperParameters)

    model.constructModel()
    startTime = datetime.now()
    model.fit(initial_learning_rate=0.01,
              learning_rate_decay=1.5,
              convergence_epochs=5,
              batch_size=32,
              maximum_epochs=100,
              learning_rate_minimum=1e-4,
              training_split=0.8)
    endTime = datetime.now()

    predictions = model.predict(isDic=True)
    trainingTime = (endTime - startTime).seconds

    observations = model.validationY[:, 0]

    # 预测值数组
    predictionsArray = np.zeros(shape=(len(predictions[0.5]), len(model.quantiles)))
    j = 0
    for key, value in predictions.items():
        predictionsArray[:, j] = value.reshape(value.shape[0])
        j = j + 1

    resultsDataFrame = pd.DataFrame(predictionsArray, columns=quantiles)
    resultsDataFrame.to_excel(resultWriter, modelName, index=False)
    resultWriter.save()
    resultWriter.close()

    pd.DataFrame(observations, columns=['observation']).to_excel(observationsWriter, 'dataset', index=False)
    observationsWriter.save()
    observationsWriter.close()

    # point prediction
    evaluation = Evaluation()
    mean = np.reshape(predictions[0.5], (len(observations),))
    pointMetrics = evaluation.getPointPredictionMetric(predictions=mean, observations=observations)
    print(pointMetrics)

    # interval prediction
    lower = np.reshape(predictions[0.025], (len(observations),))
    upper = np.reshape(predictions[0.975], (len(observations),))
    intervalMetrics = evaluation.getIntervalPredictionMetric(lower, upper, observations)
    print(intervalMetrics)

    metric = np.array([pointMetrics['RMSE'], pointMetrics['NMAPE'], pointMetrics['R2'],
                       intervalMetrics['PICP'], intervalMetrics['PINAW'], intervalMetrics['ACE'], intervalMetrics['IS']])

    metrics.append(metric)
    metricIndex.append(modelName)

    reliabilityMetrics = evaluation.getReliabilityMetric(predictionsArray, observations, quantiles)

    # plot
    draw = Draw()
    if isDrawResults:
        locArray = np.array([[1.0, 1.0]])
        draw.drawPredictions(predictions=mean, observations=observations, lower=lower, upper=upper, alpha='90%',
                             isInterval=True, xlabel='Period', ylabel='Wind Power(MW)',
                             title=modelName + '_WindPower', legendLoc=locArray[0, :], isShow=isShow,
                             isSave=isSave,
                             savePath=plotSavePath + modelName + '_results.jpg')
    if isDrawPIT:
        draw.drawPIT(reliabilityMetrics['PIT'], cdf=stats.uniform, xlabel='Uniform Distribution',
                     ylabel='PIT', title=modelName, isShow=isShow, isSave=isSave,
                     savePath=plotSavePath+modelName+'_pit.jpg')

resultWriter.save()
resultWriter.close()
metricDataFrame = pd.DataFrame(metrics, columns=['RMSE', 'NMAPE', 'R2',
                                                 'PICP', 'PINAW', 'ACE', 'IS'], index=metricIndex)
metricDataFrame.to_excel(metricWriter, 'dataset')
metricWriter.save()
metricWriter.close()

pd.DataFrame(observations, columns=['observation']).to_excel(observationsWriter, 'dataset', index=False)
observationsWriter.save()
observationsWriter.close()
