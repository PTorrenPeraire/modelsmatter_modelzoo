import os
import sys
import numpy as np

from ssbenchmark.utils import canonicalize_smiles
import pandas as pd
from rdkit import Chem
from ssbenchmark.ssmodels.base_model import ModelChoice, SSMethod


@ModelChoice.register_choice(call_name="aizynthfinder")
class model_aizynthfinder(SSMethod):
    def __init__(self, module_path=None):
        self.model_name = "AiZynthFinder"
        if module_path is not None:
            sys.path.insert(len(sys.path), os.path.abspath(module_path))

    def preprocess(self, df, reaction_col):
        import hashlib

        def create_hash(pd_row):
            if type(pd_row) == str:
                return hashlib.md5(pd_row.encode()).hexdigest()
            else:
                return hashlib.md5(pd_row.to_json().encode()).hexdigest()

        if 'reaction_type' in df.columns: # Added by Paula: Carry over from using different types of USPTO data
            df = df.drop(columns = ['reaction_type'])

        if "_id" not in df.columns:
            df["_id"] = df.apply(create_hash, axis=1)

        df.rename(columns={reaction_col: "reaction_smiles"}, inplace=True)
        if (
            "product" in df.columns
        ):  # Added by Paula: Clean up, different data styles
            df.rename(columns={"product": "prod_smiles"}, inplace=True)
        # df.rename(columns={'dataset':'split'}, inplace=True)

        reactants, spectators, products = list(
            zip(*[s.split(">") for s in df["reaction_smiles"]])
        )
        df["reactants"] = reactants
        df["spectators"] = spectators
        df["products"] = products

        df['prod_smiles'] = df.products.apply(canonicalize_smiles)

        # extract templates

        from multiprocessing import Pool

        from rdchiral.template_extractor import extract_from_reaction

        reaction_dicts = [row.to_dict() for i, row in df.iterrows()]
        with Pool(32) as pool:
            res = pool.map(extract_from_reaction, reaction_dicts)

        # assert list(df._id) == [r['reaction_id'] for r in res] ## Its commented in USPTO full notebook

        reaction_smarts = [
            r["reaction_smarts"]
            if (r is not None) and ("reaction_smarts" in r.keys())
            else np.nan
            for r in res
        ]  # introduced by Paula: workaround since data is not working - consider editing original dataset
        df["reaction_smarts"] = reaction_smarts
        df = df.dropna(axis=0, subset=["reaction_smarts"])

        # canonicalize reactant (optionally product_can_from_reaction)
        def canonicalize_reactants(smiles, can_steps=2):
            if can_steps == 0:
                return smiles

            mol = Chem.MolFromSmiles(smiles)
            for a in mol.GetAtoms():
                a.ClearProp("molAtomMapNumber")

            smiles = Chem.MolToSmiles(mol, True)
            if can_steps == 1:
                return smiles

            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
            if can_steps == 2:
                return smiles

            raise ValueError("Invalid can_steps")

        df["reactants_can"] = [
            canonicalize_reactants(r, can_steps=2) for r in df["reactants"]
        ]

        df["template_hash"] = df.reaction_smarts.apply(create_hash)
        df["reaction_hash"] = df.reaction_smiles.apply(create_hash)

        df = df.drop(columns=["reactants", "products", "spectators"])
        df = df.rename(
            columns={
                "reactants_can": "reactants",
                "prod_smiles": "products",
                "class": "classification",
                "reaction_smarts": "retro_template",
            }
        )

        return df

    def process_input(self, data, reaction_col):
        print("preprocess input")
        data = self.preprocess(data, reaction_col)
        # if 'class' in data.columns: # Added by Paula: TODO: Clean up
        #     data = data.drop(columns = ['class'])
        return data[
            [   "reaction_hash",
                "reactants",
                "products",
                "classification",
                "retro_template",
                "template_hash",
                "split",
            ]
        ]
        return data[
            [
                "id",
                "reaction_hash",
                "reactants",
                "products",
                "classification",
                "retro_template",
                "template_hash",
                "split",
            ]
        ]

    def preprocess_store(self, data, preprocess_root, instance_name):
        print("preprocess store")
        opath = os.path.join(preprocess_root, self.model_name.lower())
        if not self.check_path(opath):
            self.make_path(opath)
        data = data.reset_index(drop=True)
        data.to_csv(os.path.join(opath, f"{instance_name}_AZF_prepro.csv"))

    def process_output(self, data_root, instance_name, k):
        pass

    def model_setup(self, use_gpu=False, **kwargs):
        pass

    def model_call(self, X):
        pass
