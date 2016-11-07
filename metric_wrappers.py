from abc import ABCMeta, abstractmethod
from scipy import stats
from sklearn import metrics

class MetricWrapper(object):

	__metaclass__ = ABCMeta
	
	@classmethod
	@abstractmethod
	def measure(cls, label, prediction):
		raise NotImplementedError()


class RSquare(MetricWrapper):

	@classmethod
	def measure(cls, label, prediction):
		r, _ = stats.pearsonr(label, prediction)
		return r**2

class AUC(MetricWrapper):

	@classmethod
	def measure(cls, label, prediction):
		return metrics.roc_auc_score(label, prediction)


class RMSE(MetricWrapper):

	@classmethod
	def measure(cls, label, prediction):
		return metrics.mean_squared_error(label, prediction)**0.5

