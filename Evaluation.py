import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Evaluation(object):

    def getPointPredictionMetric(self, predictions, observations, metricNames=None):
        if metricNames is None:
            metricNames = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MAPE', 'R2', 'NMAPE']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'MAE':
                metric = Evaluation.getMAE(predictions, observations)
            elif metricName == 'MSE':
                metric = Evaluation.getMSE(predictions, observations)
            elif metricName == 'RMSE':
                metric = Evaluation.getRMSE(predictions, observations)
            elif metricName == 'MAPE':
                metric = Evaluation.getMAPE(predictions, observations)
            elif metricName == 'R2':
                metric = Evaluation.getRsquare(predictions, observations)
            elif metricName == 'NMAPE':
                metric = Evaluation.getNMAPE(predictions, observations)

            else:
                raise Exception('unknown point prediction metric name: '+metricName)
            metrics[metricName] = metric
        return metrics

    @staticmethod
    def getMAE(predictions, observations):
        MAE = np.mean(np.abs(predictions-observations))
        return MAE

    @staticmethod
    def getMSE(predictions, observations):
        MSE = np.mean(np.power(predictions-observations, 2))
        return MSE

    @staticmethod
    def getRMSE(predictions, observations):
        MSE = Evaluation.getMSE(predictions, observations)
        RMSE = np.sqrt(MSE)
        return RMSE

    @staticmethod
    def getMAPE(predictions, observations):
        MAPE = np.mean(np.true_divide(np.abs(predictions-observations), np.abs(observations)))
        return MAPE

    def getNMAPE(predictions, observations):
        datamax = np.max(observations)
        NMAPE = np.mean(np.true_divide(np.abs(predictions-observations), datamax))
        return NMAPE



    @staticmethod
    def getRsquare(predictions, observations):
        mean = np.mean(observations)
        numerator = np.sum(np.power(observations-predictions, 2))
        denominator = np.sum(np.power(observations-mean, 2))
        Rsquare = 1-numerator/denominator
        return Rsquare

    def getIntervalPredictionMetric(self, lower, upper, observations, metricNames=None):
        if metricNames==None:
            metricNames = ['PICP', 'PINAW', 'ACE', 'IS']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'PICP':
                metric = Evaluation.getCP(lower, upper, observations)
            elif metricName == 'PINAW':
                metric = Evaluation.getPINAW(lower, upper, observations)
            elif metricName == 'CM':
                metric = Evaluation.getCM(lower, upper, observations)
            elif metricName == 'ACE':
                metric = Evaluation.getACE(lower, upper, observations)
            elif metricName == 'IS':
                metric = Evaluation.getIS(lower, upper, observations)

            else:
                raise Exception('unknown interval prediction metric name: '+metricName)
            metrics[metricName] = metric
        return metrics

    @staticmethod
    def getCP(lower, upper, observations):
        N = observations.shape[0]
        count = 0
        for i in range(N):
            if observations[i]>=lower[i] and observations[i]<=upper[i]:
                count = count + 1
        CP = count/N
        return CP

    def getACE(lower, upper, observations):
        N = observations.shape[0]
        count = 0
        for i in range(N):
            if observations[i]>=lower[i] and observations[i]<=upper[i]:
                count = count + 1
        ACE = (count/N - 0.95)*100
        return ACE


    def getIS(lower, upper, observations):
        N = observations.shape[0]
        a = 0.05
        IS = 0
        for i in range(N):
            if observations[i]>=lower[i] and observations[i]<=upper[i]:
                IS = IS - (upper[i]-lower[i])*2*a
            if observations[i] < lower[i]:
                IS = IS - (upper[i] - lower[i]) * 2 * a - 4*(lower[i]-observations[i])
            if observations[i] > upper[i]:
                IS = IS - (upper[i] - lower[i]) * 2 * a - 4*(observations[i]-upper[i])
        IS = IS/N
        return IS



    @staticmethod
    def getPINAW(lower, upper, observations):
        N = observations.shape[0]
        PINAW = 0
        R = np.max(observations) - np.min(observations)
        for i in range(N):
            if upper[i] < lower[i]:
                print(i)
            PINAW = PINAW + (upper[i]-lower[i])/R
        PINAW = PINAW/N
        return PINAW

    @staticmethod
    def getCM(lower, upper, observations):
        CM = Evaluation.getCP(lower, upper, observations)/Evaluation.getMWP(lower, upper, observations)
        return CM

    def getProbabilityPredictionMetric(self, predictionsArray, observations, quantiles, metricNames=None):
        if metricNames is None:
            metricNames = ['CRPS']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'CRPS':
                metric = Evaluation.getCRPS(predictionsArray, observations, quantiles)
            else:
                raise Exception('unknown probability prediction metric name: '+metric)
            metrics[metricName] = metric
        return metrics

    @staticmethod
    def cdf(predictionsArray, quantiles):
        y_cdf = np.zeros((predictionsArray.shape[0], quantiles.size + 2))
        y_cdf[:, 1:-1] = predictionsArray
        y_cdf[:, 0] = 2.0 * predictionsArray[:, 1] - predictionsArray[:, 2]
        y_cdf[:, -1] = 2.0 * predictionsArray[:, -2] - predictionsArray[:, -3]
        qs = np.zeros((1, quantiles.size + 2))
        qs[0, 1:-1] = quantiles
        qs[0, 0] = 0.0
        qs[0, -1] = 1.0
        return y_cdf, qs

    @staticmethod
    def getCRPS(predictionsArray, observations, quantiles):
        y_cdf, qs = Evaluation.cdf(predictionsArray, quantiles)
        ind = np.zeros(y_cdf.shape)
        ind[y_cdf < observations.reshape(-1, 1)] = 1.0
        scaler1 = MinMaxScaler()
        y_cdf = scaler1.fit_transform(y_cdf.T)
        CRPS = np.trapz((qs - ind) ** 2.0, y_cdf.T)
        CRPS = np.mean(CRPS)
        return CRPS

    # 获取可靠性指标
    def getReliabilityMetric(self, predictionsArray, observations, quantiles, metricNames=None):
        if metricNames is None:
            metricNames = ['PIT']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'PIT':
                metric = Evaluation.getPIT(predictionsArray, observations, quantiles)
            else:
                raise Exception('unknown probability prediction metric name: '+metric)
            metrics[metricName] = metric
        return metrics

    @staticmethod
    def getPIT(predictionsArray, observations, quantiles):
        y_cdf, qs = Evaluation.cdf(predictionsArray, quantiles)
        PIT = np.zeros(observations.shape)
        for i in range(observations.shape[0]):
            PIT[i] = np.interp(np.squeeze(observations[i]), np.squeeze(y_cdf[i, :]), np.squeeze(qs))
        return PIT
