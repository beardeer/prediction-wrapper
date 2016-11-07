import pandas as pd

class PredResult(object):

    def __init__(self, index_len, columns):
        self.result = pd.DataFrame(index = range(index_len), columns = columns)

    def set_col(self, data, col_name, idx):
        self.result[col_name].ix[idx] = data



class BinaryPredResult(PredResult):

    def  __init__(self, index_len):
        PredResult.__init__(self, index_len, ['label', 'pred_prob', 'pred_label'])

    @property
    def label(self):
        return self.result['label']

    @property
    def pred_prob(self):
        return self.result['pred_prob']

    @property
    def pred_label(self):
        return self.result['pred_label']

