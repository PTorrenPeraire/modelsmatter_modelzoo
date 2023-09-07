import collections
from .base_model import ModelChoice, SSMethod
from rdkit import Chem
import os
import pandas as pd
from ssbenchmark.utils import canonicalize_smiles, split_reactions
import numpy as np


@ModelChoice.register_choice("augmented_transformer")
class model_augmented_transformer(SSMethod):
    def __init__(self):
        self.model_name = "augmented_transformer"
        self.requires_templates = False

    def augment_data(self, data):
        augmented_data = pd.DataFrame()
        smiles_enum = Enumerator(
            enumerations=4, seed=42, oversample=10, keep_original=True, only_unique=False, shuffle_reactants=True
        )
        for spl in ["train", "val", "test"]:
            d = data[data["split"] == spl]
            d = d.drop(columns=["split"])
            if spl != "test":
                d["products"] = [smiles_enum.smiles_enumeration(s) for s in d["products"].tolist()]
                d["reactants"] = [smiles_enum.smiles_enumeration(s) for s in d["reactants"].tolist()]
                d = d.explode(["products", "reactants"])
                d = smiles_enum.create_forward_reaction(d, col_names=["products", "reactants"])
            else:
                smiles_enum = Enumerator(
                    enumerations=99, seed=42, oversample=150, keep_original=True, only_unique=False
                )
                d["products"] = [smiles_enum.smiles_enumeration(s) for s in d["products"].values.tolist()]
                d["reactants"] = [smiles_enum.smiles_enumeration(s) for s in d["reactants"].values.tolist()]
                d = d.explode(["products", "reactants"])
            d["split"] = spl
            augmented_data = pd.concat([augmented_data, d])
        return augmented_data

    def preprocess(self, data):
        data["products"], data["spectators"], data["reactants"] = split_reactions(data["rxn_smiles"].tolist())
        data["products"] = data.products.apply(canonicalize_smiles)
        data["reactants"] = data.reactants.apply(canonicalize_smiles)
        data = self.augment_data(data)
        return data

    def process_input(self, data):
        print("preprocess input")
        data = self.preprocess(data)
        data = data.rename(columns={"products": "input", "reactants": "target"})
        return data[["input", "target", "split"]]

    def preprocess_store(self, data, preprocess_root, instance_name):
        print("preprocess store")
        if not os.path.exists(preprocess_root):
            os.mkdir(preprocess_root)
        for spl in ["train", "valid", "test"]:
            if spl == "valid":
                spl_ = "val"
            else:
                spl_ = spl
            d = data[data["split"] == spl][data.columns[:-1]]
            d = d.reset_index(drop=True)
            d.to_csv(os.path.join(preprocess_root, f"{instance_name}_{spl_}.csv"))

    def process_output(self, data_root, instance_name, k):
        data = pd.read_csv(os.path.join(data_root, self.model_name.lower(), f"{instance_name}.csv"))
        # col = [f"target_{i}" for i in range(1, self.k + 1)]
        # data = data[col]
        # data = data.values.tolist()
        # return [canonicalize_smiles(pred) for pred in data]
        def get_freq(d):
            i = d.index.min()
            d = d[d.columns[2:]].values.tolist()
            d = [[canonicalize_smiles(a) for a in aug] for aug in d]
            try:
                d = [".".join(sorted(smi.split("."))) if smi is not None else None for d_ in d for smi in d_]
            except:
                print(d)
            try:
                freq = collections.Counter(d)
            except:
                print(d)
            del freq[None]
            return [[smi for smi, _ in freq.most_common(k)], i]

        data = data.fillna(np.nan).replace([np.nan], [None])  # TODO: This needs to be cleaner
        data["can_input"] = [canonicalize_smiles(smi) for smi in data.input.values.tolist()]
        data = [
            get_freq(data.iloc[a : a + 100]) for a in range(0, len(data), 100)
        ]  # TODO: The 1s represent the amount of augmentation - if we were to use 10 then that would be the amount of augmentation - really don't like how I'm doing this
        # data = [get_freq(d) for inp, d in data.groupby('input')]
        order_smiles = np.array([i for smi, i in data])
        order_smiles = np.argsort(order_smiles).tolist()
        data = [data[i][0] for i in order_smiles]
        data = [[d_.split(".") for d_ in d] for d in data]
        return data

    def model_setup(self):
        pass

    def model_call(self):
        pass


from typing import List, Optional, Union
import pandas as pd
import numpy as np
import sys
import random


class Enumerator:  # Import this from coding framework instead of copy/paste (then create new class to add forward reaction funciton)
    def __init__(
        self,
        enumerations: int,
        seed: int,
        oversample: int = 0,
        max_len: Optional[int] = None,
        keep_original: bool = True,
        only_unique: bool = True,
        shuffle_reactants: bool = False,
    ) -> None:
        self.enumerations = enumerations
        self.oversample = oversample
        self.max_len = max_len
        self.keep_original = keep_original

        self.seed = seed

        self.only_unique = only_unique

        self.shuffle_reactants = shuffle_reactants

    def _shuffle_reactants(self, smile):
        smile = smile.split(".")
        random.shuffle(smile)
        return ".".join(smile)

    def _smiles_enumeration(
        self,
        smile: str,
    ) -> Union[List[str], None]:
        try:
            enum_smiles = Chem.MolToRandomSmilesVect(
                Chem.MolFromSmiles(smile),
                self.enumerations + self.oversample,
                randomSeed=self.seed,
            )
        except Exception:
            return None

        if self.keep_original:
            unique_enum_smiles = list(set([smile, *enum_smiles]))
            max_n = self.enumerations + 1
        else:
            unique_enum_smiles = list(set(enum_smiles))
            max_n = self.enumerations

        if self.max_len:
            unique_enum_smiles = [smile for smile in unique_enum_smiles if len(smile) <= self.max_len]

        if self.shuffle_reactants:
            unique_enum_smiles[1:] = [self._shuffle_reactants(smi) for smi in unique_enum_smiles[1:]]

        if self.only_unique:
            return unique_enum_smiles[:max_n] if len(unique_enum_smiles) >= max_n else None
        else:
            if len(unique_enum_smiles) < max_n:
                return (unique_enum_smiles * int(np.ceil(max_n / len(unique_enum_smiles))))[:max_n]
            else:
                return unique_enum_smiles[:max_n]

    def smiles_enumeration(self, smiles):
        try:
            smiles = [self._smiles_enumeration(smi) for smi in smiles.split(".")]
        except:
            print(smiles)
        return [".".join(smi) for smi in list(zip(*smiles))]

    def dataframe_enumeration(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.applymap(lambda x: self.smiles_enumeration(x))
        data = data.dropna()
        data = data.explode(list(data.columns))
        return data.reset_index(drop=True)

    def create_forward_reaction(self, data, col_names=["products", "reactants"]):
        assert len(col_names) == 2

        input = data[col_names[0]].values.tolist()
        output = data[col_names[-1]].values.tolist()
        fdata = data.copy()
        fdata[col_names[0]] = ["." + oup for oup in output]
        fdata[col_names[-1]] = input
        data = pd.concat([data, fdata], ignore_index=True)
        # data = data.append([["."+oup, inp]for inp, oup in zip(input, output)])
        return data
