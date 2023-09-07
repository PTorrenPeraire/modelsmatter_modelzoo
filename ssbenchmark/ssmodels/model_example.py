import os
import sys

from ssbenchmark.ssmodels.base_model import Registries, SSMethod
from ssbenchmark.utils import canonicalize_smiles, split_reactions


@Registries.ModelChoice.register(key="example_model")
class model_test(SSMethod):
    def __init__(self, module_path=None):
        self.model_name = "ExampleModel"
        if module_path is not None:
            sys.path.insert(len(sys.path), os.path.abspath(module_path))

    def preprocess(self):
        pass
    def process_input(self):
        pass

    def preprocess_store(self):
        pass

    def process_output(self):
        pass

    def model_setup(self, use_gpu=False, **kwargs):
        pass


    def _model_call(self, X):

        mc = {'A':['D', 'G', 'I'],
              'B':['E', 'E', 'J'],
              'C':['F', 'H', 'K'],
              }

        smiles = [mc[x] for x in X]
        prob = [[0.97, 0.02, 0.01] for s in smiles]

        return smiles, prob
    
    def model_call(self, X):
        return self._model_call(X)