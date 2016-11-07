"""Summary
"""
import pandas as pd

pd.options.mode.chained_assignment = None

class PredResult(object):
    """Summary

    Attributes
    ----------
    result : TYPE
        Description
    """
    def __init__(self, index_len, columns):
        """Summary

        Parameters
        ----------
        index_len : TYPE
            Description
        columns : TYPE
            Description
        """
        self.result = pd.DataFrame(index = range(index_len), columns = columns)

    def set_col(self, data, col_name, idx):
        """Summary

        Parameters
        ----------
        data : TYPE
            Description
        col_name : TYPE
            Description
        idx : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        self.result[col_name].ix[idx] = data



class BinaryPredResult(PredResult):
    """Summary
    """
    def  __init__(self, index_len):
        """Summary

        Parameters
        ----------
        index_len : TYPE
            Description
        """
        PredResult.__init__(self, index_len, ['label', 'pred_prob', 'pred_label'])

    @property
    def label(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.result['label']

    @property
    def pred_prob(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.result['pred_prob']

    @property
    def pred_label(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.result['pred_label']

