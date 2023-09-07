import os
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
# from aidd_codebase.utils.metacoding import DictChoiceFactory
from registry_factory.factory import Factory


class Registries(Factory):
    ModelChoice = Factory.create_registry(shared=False)

# class ModelChoice(DictChoiceFactory):
#     pass


class SSMethod(ABC):
    """Base class for single-step methods"""

    def __init__(self, module_path: str) -> None:
        pass

    @abstractmethod
    def process_input(self, data: pd.DataFrame, reaction_col: str) -> pd.DataFrame:
        """Abstract method for processing input data.

        Args:
            data (pd.DataFrame): Data to be processed.
            reaction_col (str): Name of the reaction column.

        Returns:
            pd.DataFrame: Processed data.
        """
        pass

    @abstractmethod
    def preprocess_store(self, data: pd.DataFrame, preprocess_root: str) -> None:
        """Abstract method for storing preprocessed data.

        Args:
            data (pd.DataFrame): Data to be stored.
            preprocess_root (str): Path to store data.
        """
        pass

    @abstractmethod
    def model_setup(self, use_gpu: bool):
        """Abstract method for setting up model.

        Args:
            use_gpu (bool): Whether to use GPU.
        """
        pass

    @abstractmethod
    def model_call(self, X: list):
        """Abstract method for calling model.

        Args:
            X (list): List of SMILES strings.
        """
        pass

    @abstractmethod
    def process_output(self):
        pass

    def check_path(self, path: str) -> bool:
        return os.path.exists(path)

    def make_path(self, path: str) -> None:
        os.makedirs(path)

    def get_model_name(self) -> str:
        return self.model_name

    def read_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def read_pickle(self, path: str) -> pd.DataFrame:
        return pd.read_pickle(path)

    def write_csv(self, df: pd.DataFrame, root: str, fname: str) -> None:
        if not self.check_path(root):
            self.make_path(root)
        df.to_csv(os.path.join(root, fname), index=False)

    def write_pickle(self, df: pd.DataFrame, root: str, fname: str) -> None:
        if not self.check_path(root):
            self.make_path(root)
        df.to_pickle(os.path.join(root, fname))

    def collate_batches(self, batches: list) -> pd.DataFrame:
        batches = pd.concat(batches, ignore_index=True)
        return batches

    def gather_batches(self, preprocess_root: str, ftype: str = "csv") -> pd.DataFrame:
        batches = pd.DataFrame()
        for i, file in enumerate(os.listdir(preprocess_root)):
            if ftype == "csv":
                batches = pd.concat([batches, self.read_csv(os.path.join(preprocess_root, file))])
            elif ftype == "pickle":
                try:
                    batches = pd.concat([batches, self.read_pickle(os.path.join(preprocess_root, file))])
                except:
                    print(f"Failed to read {file}")
        batches.reset_index(drop=True, inplace=True)
        return batches
