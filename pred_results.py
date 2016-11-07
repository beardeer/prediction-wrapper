"""'Container' classes to store prediction results for easy access
"""

# Author: Xiaolu Xiong <beardeer@gmail.com>

import pandas as pd

# Disalbe DataFrame overwriting warning
pd.options.mode.chained_assignment = None

class PredResults(object):
    """Base class to store prediction results.

    Attributes
    ----------
    results : DataFrame
        DataFrame that used to store data
    """
    def __init__(self, index_len, columns):
        self.results = pd.DataFrame(index=range(index_len), columns=columns)

    def set_col(self, data, col_name, idx):
        """Set DataFrame column values by column name and indexes

        Parameters
        ----------
        data : array
            Input data
        col_name : str
            A column name in the results DataFrame
        idx : list
            A list of indexes of value locations

        Returns
        -------
        None
        """
        self.results[col_name].ix[idx] = data



class BinaryPredResults(PredResults):
    """An implementation to store binary prediction results.

    It contains the labels, the predicted probabilities, predicted labels.
    """
    def  __init__(self, index_len):
        PredResults.__init__(self, index_len, ['label', 'pred_prob', 'pred_label'])

    @property
    def label(self):
        """Get the 'label' array

        Returns
        -------
        pandas.Series
            The stored labels
        """
        return self.results['label']

    @property
    def pred_prob(self):
        """Get the predicted probability values

        Returns
        -------
        pandas.Series
            The stored probability values
        """
        return self.results['pred_prob']

    @property
    def pred_label(self):
        """Get the predicted probability labels

        Returns
        -------
        pandas.Series
            The stored probability labels
        """
        return self.results['pred_label']
