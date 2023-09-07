import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn import model_selection
from sklearn.model_selection import KFold
from ssbenchmark.utils import canonicalize_smiles, split_reactions, subsample

RANDOM_SEED = 42


class DataInput:
    def __init__(self, data_path, train_size=0.80, val_size=0.10, test_size=0.10, subsample_n=-1):
        try:  # TODO: Clean this up, this is very janky
            self.data = pd.read_csv(
                data_path, header=0, sep=","
            )  # TODO: Function so not only .csv (at least check if it is csv)
            # self.data = pd.read_csv(
            #     data_path, header=0, index_col=0, sep=","
            # )  # TODO: Function so not only .csv (at least check if it is csv)
        except:
            self.data = pd.read_csv(data_path, header=0, sep="\t")
        if subsample_n > 0:
            print("subsampling")
            self.data = subsample(self.data, n=subsample_n)
        print(self.data.shape)
        data_size = train_size + val_size + test_size
        self.train_size = train_size / data_size
        self.val_size = val_size / data_size
        self.test_size = test_size / data_size

        self.train_idx = None
        self.valid_idx = None
        self.test_idx = None

    def split_data(self):
        data = self.data
        train_idx, self.test_idx, _, _ = model_selection.train_test_split(
            range(len(data)), range(len(data)), test_size=self.test_size
        )
        self.train_idx, self.valid_idx, _, _ = model_selection.train_test_split(
            train_idx, train_idx, test_size=(self.val_size * len(data)) / len(train_idx)
        )

        split = np.zeros(len(data))
        split[self.valid_idx] = 1
        split[self.test_idx] = 2  # Assuming that all remaining idx are train
        data["split"] = split
        data["split"] = data["split"].map({0: "train", 1: "valid", 2: "test"})
        return data

    def k_split_data(self, k=5):
        data = self.data

        kf = model_selection.KFold(n_splits=int(k))
        for train_idx, test_idx in kf.split(data):
            train_idx, valid_idx, _, _ = model_selection.train_test_split(
                train_idx, train_idx, random_state=RANDOM_SEED, test_size=(self.val_size * len(data)) / len(train_idx)
            )

            split = np.zeros(len(data))
            split[valid_idx] = 1
            split[test_idx] = 2  # Assuming that all remaining idx are train
            data["split"] = split
            data["split"] = data["split"].map({0: "train", 1: "valid", 2: "test"})
            yield data

    def store_test_target(self, data, reaction_col, preprocess_root, name_instance):  # TODO: Move this to DataInput
        data = data.copy()
        data = data[data["split"] == "test"]
        data["reactants"], data["spectators"], data["products"] = split_reactions(data[reaction_col].tolist())
        data["reactants"] = data.reactants.apply(canonicalize_smiles)
        data["reactants"].to_csv(
            os.path.join(preprocess_root, f"target_smiles_{name_instance}.txt"), index=False, header=None
        )

    def store_split(self, data, oroot, instance_name):
        data.to_csv(os.path.join(oroot, f"rawdata_{instance_name}.csv"), index=False)


class DataOutput:
    def __init__(self, y_smiles) -> None:
        y_smiles = pd.read_csv(
            y_smiles, header=None
        ).values.tolist()  # TODO: Its a txt file, read csv may not be best way to process (having to use two sets of indices after)
        y_smiles = [
            canonicalize_smiles(smi[0]) for smi in y_smiles
        ]  # TODO: y_true should be a single set of reactants there should be several options for the the same products
        self.y_smiles = [smi.split(".") for smi in y_smiles if smi is not None]

    def _compare_smiles(self, tgt, topk_pred):
        topk_pred = [pred.split(".") if type(pred) == str else pred for pred in topk_pred]
        output = [int(set(tgt) == set(pred)) if pred is not None else -1 for pred in topk_pred]
        output = torch.Tensor(output)
        return torch.any(output == 1)

    def compare_smiles(self, pred):
        m = [
            self._compare_smiles(tgt, pred) for i, (tgt, pred) in enumerate(zip(self.y_smiles, pred)) if tgt is not None
        ]
        m = torch.Tensor(m)
        return torch.sum(m) / m.numel()
    
    def compare_retrieve_smiles(self, pred):
        m = [
            [tgt, self._compare_smiles(tgt, pred)] for i, (tgt, pred) in enumerate(zip(self.y_smiles, pred)) if tgt is not None
        ]
        m_ = torch.Tensor([m_[1] for m_ in m])
        m_=torch.sum(m_) / m_.numel()
        print(f"Accuracy: {m_}")
        return m