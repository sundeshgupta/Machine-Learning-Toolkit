import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)

from sklearn.metrics import precision_recall_fscore_support
def plot_classification_report(y_true, y_pred, figsize=(10, 10), ax=None):
    np.set_printoptions(suppress=True)
    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = list(np.unique(y_true))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_true, y_pred)).T
    # print(rep)
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,annot=True,cbar=False,xticklabels=xticks,yticklabels=yticks,ax=ax, cmap='GnBu')

    # plt.show()
    return plt

def plot_confusion_matrix(cm):
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cmap='GnBu')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    return plt
