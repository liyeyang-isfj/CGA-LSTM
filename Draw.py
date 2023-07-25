from statsmodels.graphics.api import qqplot
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
plt.rc('font', family='Times New Roman')


class Draw(object):
    def drawPIT(self, data, cdf, xlabel, ylabel, title, isShow, isSave, savePath):
        lw = 4
        fontsize = 40
        fig = plt.figure(figsize=(12, 12))
        axs = fig.add_subplot(111)
        fig = qqplot(data, dist=cdf, line='45', ax=axs, fit=False)
        deta = 1.358 / (len(data)) ** 0.5 * (2 ** 0.5)
        print(len(data), deta)
        axs.plot([deta, 1], [0, 1 - deta], '--', color='blueviolet', lw=lw, label='Kolmogorov 5% significance band')
        axs.plot([0, 1 - deta], [deta, 1], '--', color='blueviolet', lw=lw)
        axs.set_title(title, loc="center", fontsize=fontsize)
        axs.set_xlabel(xlabel, fontsize=fontsize)
        axs.set_ylabel(ylabel, fontsize=fontsize)
        axs.set_xlim([0, 1])
        axs.set_ylim([0, 1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        plt.legend(fontsize=25)
        if isShow:
            plt.show()
        if isSave:
            fig.savefig(savePath, bbox_inches="tight", dpi=300)
        plt.close()

    def drawPredictions(self, predictions, observations, lower, upper, alpha,
                        isInterval, xlabel, ylabel, title, legendLoc, isShow, isSave, savePath):
        lw = 2
        fontsize = 20
        index = [i for i in range(observations.shape[0])]

        fig, axs = plt.subplots(1, 1, figsize=(12, 6))

        axs.plot(index, observations, marker="o", markersize=3, label="observations", lw=lw, color='r')
        preColor = (0.11765, 0.56471, 1)
        axs.plot(index, predictions, marker="D", markersize=3, label="predictions", lw=lw, color=preColor)

        np.reshape(np.array(lower), (len(index), 1))
        np.reshape(np.array(upper), (len(index), 1))
        if isInterval:
            axs.plot(index, lower, color=preColor, lw=lw / 2)
            axs.plot(index, upper, color=preColor, lw=lw / 2)
            axs.fill_between(index, lower, upper, alpha=1, color=(0.9, 0.9, 0.9),
                             label='' + alpha + ' prediction interval')
            for i in range(len(index)):
                if observations[i] > upper[i] or observations[i] < lower[i]:
                    axs.plot(index[i], observations[i], marker="o", markersize=3, color='chartreuse')

        axs.grid()


        xmajorLocator = MultipleLocator(20)
        xminorLocator = MultipleLocator(10)
        ymajorLocator = MultipleLocator(5.0)
        yminorLocator = MultipleLocator(2.5)

        axs.xaxis.set_major_locator(xmajorLocator)
        axs.yaxis.set_major_locator(ymajorLocator)

        axs.xaxis.set_minor_locator(xminorLocator)
        axs.yaxis.set_minor_locator(yminorLocator)

        axs.xaxis.grid(True, which='major')
        axs.yaxis.grid(True, which='minor')

        axs.set_title(title, loc="center", fontsize=fontsize)
        axs.set_xlim([min(index), max(index)])
        axs.set_ylim([round(min(lower) * 1.2), round(max(upper) * 1.2)])
        axs.set_xlabel(xlabel, fontsize=fontsize)
        axs.set_ylabel(ylabel, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

        fig.legend(fontsize=fontsize - 5, bbox_to_anchor=legendLoc, bbox_transform=axs.transAxes)
        plt.tight_layout()
        if isShow:
            plt.show()
        if isSave:
            fig.savefig(savePath, bbox_inches="tight", dpi=300)
        plt.close()

    def drawKD(self, kde, samples, observation, xlabel, ylabel, title, isShow, isSave, savePath):
        lw = 4
        fontsize = 24
        mi = min(samples) - 0.5 * (max(samples) - min(samples))
        ma = max(samples) + 0.5 * (max(samples) - min(samples))
        X_plot = np.array(np.linspace(mi, ma, 1000)).reshape(-1, 1)
        log_dens = kde.score_samples(X_plot)
        fig, ax = plt.subplots()
        ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
                label="probability density curve", lw=lw)
        ax.plot(np.array([observation, observation]),
                np.array([0, max(np.exp(log_dens))]),
                'r-', label='observation', lw=lw)
        ax.fill_between(X_plot[:, 0], np.zeros(len(log_dens)), np.exp(log_dens), alpha=1, color=[0.9, 0.9, 0.9])
        xyRange = plt.axis()
        ax.set_ylim([0, xyRange[3] * 1.01])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        ax.text(observation * 1.02, 0.0002, str(format(observation, '.2f')), color='blueviolet', fontsize=fontsize, fontweight='bold')
        ax.set_title(title, loc="center", fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=10)
        plt.grid()
        if isShow:
            plt.show()
        if isSave:
            fig.savefig(savePath, bbox_inches="tight", dpi=300)
        plt.close()
