"""Summary
"""
from abc import ABCMeta, abstractmethod
from scipy import stats
from sklearn import metrics

class MetricWrapper(object):
	"""Summary
	"""
	__metaclass__ = ABCMeta
	
	@classmethod
	@abstractmethod
	def measure(cls, label, prediction):
		"""Summary

		Parameters
		----------
		label : TYPE
		    Description
		prediction : TYPE
		    Description

		Returns
		-------
		TYPE
		    Description
		"""
		raise NotImplementedError()


class RSquare(MetricWrapper):
	"""Summary
	"""
	@classmethod
	def measure(cls, label, prediction):
		"""Summary

		Parameters
		----------
		label : TYPE
		    Description
		prediction : TYPE
		    Description

		Returns
		-------
		TYPE
		    Description
		"""
		r, _ = stats.pearsonr(label, prediction)
		return r**2

class AUC(MetricWrapper):
	"""Summary
	"""
	@classmethod
	def measure(cls, label, prediction):
		"""Summary

		Parameters
		----------
		label : TYPE
		    Description
		prediction : TYPE
		    Description

		Returns
		-------
		TYPE
		    Description
		"""
		return metrics.roc_auc_score(label, prediction)


class RMSE(MetricWrapper):
	"""Summary
	"""
	@classmethod
	def measure(cls, label, prediction):
		"""Summary

		Parameters
		----------
		label : TYPE
		    Description
		prediction : TYPE
		    Description

		Returns
		-------
		TYPE
		    Description
		"""
		return metrics.mean_squared_error(label, prediction)**0.5

