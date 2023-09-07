import argparse
import os
import pickle as pkl
import sys
import time
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Memory
from pyexpat import model
from rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem
from ssbenchmark.ssmodels.base_model import ModelChoice, SSMethod
from ssbenchmark.utils import canonicalize_smiles


@ModelChoice.register_choice(call_name="mhn_react_nocache")
class model_mhn_react_nocache(SSMethod):
    def __init__(self, module_path=None):
        self.model_name = "mhn_react"
        if module_path is not None:
            sys.path.insert(len(sys.path), os.path.abspath(module_path))

    def _extract_from_reaction(
        self,
        reaction_dicts,
    ):  # Added by Paula: some non-concentional reactions fail and since were using pool makes the whole batch fail
        try:
            re = extract_from_reaction(reaction_dicts)
            if "reaction_smarts" not in re:
                re["reaction_smarts"] = ""
            return re
        except:
            return {}

    def _preprocess(self, df, reaction_col):
        import hashlib

        def create_hash(pd_row):
            return hashlib.md5(pd_row.to_json().encode()).hexdigest()

        if (
            "reaction_type" in df.columns
        ):  # Added by Paula: Carry over from using different types of USPTO data
            df = df.drop(columns=["reaction_type"])

        if "_id" not in df.columns:
            df["_id"] = df.apply(create_hash, axis=1)

        df.rename(columns={reaction_col: "reaction_smiles"}, inplace=True)
        if "product" in df.columns:  # Added by Paula: Clean up, different data styles
            df.rename(columns={"product": "prod_smiles"}, inplace=True)
        # df.rename(columns={'dataset':'split'}, inplace=True)

        reactants, spectators, products = list(
            zip(*[s.split(">") for s in df["reaction_smiles"]])
        )
        df["reactants"] = reactants
        df["spectators"] = spectators
        df["products"] = products

        df["prod_smiles"] = df.products.apply(canonicalize_smiles)

        # extract templates

        from multiprocessing import get_context

        from tqdm.contrib.concurrent import process_map

        reaction_dicts = [row.to_dict() for i, row in df.iterrows()]
        njobs = (
            os.cpu_count()
        )  # IMPORTANT: THIS SHOULD ONLY BE USED IF THERE IS A SLURM QUEUE!
        res = process_map(
            self._extract_from_reaction, reaction_dicts, max_workers=njobs, chunksize=1
        )

        # t_end = time.time() + 60 * 30
        # finished = False
        # while time.time() < t_end:
        #     with get_context("fork").Pool(32) as pool:
        #         res = pool.map(self._extract_from_reaction, reaction_dicts)
        #     finished = True
        # if not finished:
        #     print("TIMED OUT")
        #     res = [self._extract_from_reaction(r) for r in reaction_dicts]
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
            canonicalize_reactants(r, can_steps=2) if r is not None else np.nan
            for r in df["reactants"]
        ]
        return df

    def preprocess(self, df, reaction_col, batched=False):
        if not batched:
            df = self._preprocess(df, reaction_col)

        def filter_by_dict(df, fil):
            for col, value in fil.items():
                if not isinstance(value, list):
                    value = [value]
                df = df[df[col].isin(value)]
            return df

        import re

        mre = ":\d+(?=])"
        # eval reaction_smarts column in df

        df["reaction_smarts"] = df["reaction_smarts"].astype(str)
        df["unmapped_template"] = df["reaction_smarts"].apply(
            lambda x: re.sub(mre, "", x)
        )
        # unmapped = [re.sub(mre, "", r) for r in df["reaction_smarts"]]
        # df["unmapped_template"] = unmapped

        unmapped2idx = {}
        labels = []
        for split in ["train", "valid", "test"]:
            sub = filter_by_dict(df, {"split": split})
            for u in sub["unmapped_template"]:
                if u not in unmapped2idx:
                    label = len(unmapped2idx)
                    unmapped2idx[u] = label

        df["label"] = [unmapped2idx[u] for u in df["unmapped_template"]]
        if "class" in df.columns:  # Added by Paula: TODO: Clean up
            df = df.drop(columns=["class"])
        return df[["prod_smiles", "reactants_can", "split", "reaction_smarts", "label"]]

    def process_input(self, data, reaction_col, chunk_size=None):
        print("preprocess input")
        data = self.preprocess(data, reaction_col, chunk_size)
        return data
        # return data[['id','prod_smiles', 'reactants_can','split', 'reaction_smarts', 'label']]

    def preprocess_store(self, data, preprocess_root, instance_name):
        print("preprocess store")
        opath = os.path.abspath(preprocess_root)
        if not self.check_path(opath):
            self.make_path(opath)
        data = data.reset_index(drop=True)
        data.to_csv(os.path.join(opath, f"{instance_name}_MHN_prepro.csv.gz"))

    def process_output(self, data_root, instance_name, k):
        with open(
            os.path.join(data_root, self.model_name.lower(), f"{instance_name}.pickle"),
            "rb",
        ) as f:
            data = pkl.load(f)
        data = [d[:k] for d in data]
        return [[canonicalize_smiles(p) for p in pred] for pred in data]

    def sort_by_template_and_flatten(
        self, template_scores, prod_idx_reactants, agglo_fun=sum
    ):  # Adapted from mhnreact
        flat_results = []
        flat_priors = []
        for ii in range(len(template_scores)):
            idx_prod_reactants = defaultdict(list)
            for k, v in prod_idx_reactants[ii].items():
                for iv in v:
                    idx_prod_reactants[iv].append(template_scores[ii, k])
            d2 = {k: agglo_fun(v) for k, v in idx_prod_reactants.items()}
            if len(d2) == 0:
                flat_results.append([])
            else:
                res = pd.DataFrame.from_dict(d2, orient="index").sort_values(
                    0, ascending=False
                )
                flat_results.append(res.index.values.tolist())
                flat_priors.append(res[0].values.tolist())
        return flat_results, flat_priors

    def model_setup(
        self,
        use_gpu=False,
        model_fn="",
        model_path="",
        template_path="",
        large_dataset=False,
        **kwargs,
    ):
        import pandas as pd
        from mhnreact.data import load_dataset_from_csv
        from mhnreact.inspect import load_clf
        from mhnreact.molutils import LargeDataset

        self.model = load_clf(model_fn, model_path, model_type="mhn", device="cpu")
        self.bs = kwargs["bs"] if "bs" in kwargs else 1024
        self.model.eval()
        X, y, template_list, test_reactants_can = load_dataset_from_csv(
            csv_path=template_path,
            split_col="split",
            input_col="prod_smiles",
            ssretroeval=True,
            reactants_col="reactants_can",
            ret_df=False,
        )
        self.templates = list(template_list.values())
        self.template_product_smarts = [str(s).split(">")[0] for s in self.templates]
        if large_dataset:  # TODO: REAL JANKY
            self.large_dataset = LargeDataset(
                kwargs["fp_type"] if "fp_type" in kwargs.keys() else "morgan",
                kwargs["template_fp_type"]
                if "template_fp_type" in kwargs.keys()
                else "rdk",
                kwargs["fp_size"] if "fp_size" in kwargs.keys() else 4096,
                kwargs["fp_radius"] if "fp_radius" in kwargs.keys() else 2,
                template_list,
                kwargs["njobs"] if "njobs" in kwargs.keys() else -1,
                kwargs["only_templates_in_batch"]
                if "only_templates_in_batch" in kwargs.keys()
                else False,
                kwargs["verbose"] if "verbose" in kwargs.keys() else False,
            )
        cachedir = "./data/cache"
        self.cachedir = os.path.join(cachedir, str(uuid.uuid4()))
        #self.cachedir_template="/home/stb/paula.torren/github/esr7aizynthfinder/experiments/multistep/retrostar_extended/paroutes/mhnreact/paroutes/paroutes/data/cached_templates"
        #self.model.template_cache={i:os.path.join(self.cachedir_template, f"tmp_{i}.pt") for i in range(32)}

    def _model_call(self, X):
        from mhnreact.molutils import smarts2appl
        from mhnreact.retroeval import run_templates

        memory = Memory(self.cachedir, verbose=0, bytes_limit=1e9)
        X = [[x] for x in X]

        # execute all template
        print("execute all templates")
        test_product_smarts = [xi[0] for xi in X]  # added later
        smarts2appl = memory.cache(smarts2appl, ignore=["njobs", "nsplits"])
        memory.reduce_size()
        appl = smarts2appl(test_product_smarts, self.template_product_smarts)
        # n_pairs = len(test_product_smarts) * len(self.template_product_smarts)
        # check if either in appl is size 0
        if (appl[0].size | appl[1].size) == 0:
            print(f"No templates were applicable for {X}, returning empty lists")
            return [], []
        y_preds = None
        # forward
        y = np.zeros(len(X)).astype(np.int)
        # clf.eval()
        if y_preds is None:
            y_preds = self.model.evaluate(
                X,
                X,
                y,
                is_smiles=True,
                split="ttest",
                only_loss=True,
                wandb=None,
                large_dataset=self.large_dataset,
                bs=self.bs,
                #cachedir=self.cachedir_template,
            )

        template_scores = y_preds  # this should allready be test

        ####
        if y_preds.shape[1] > 100000:
            kth = 200
            print(f"only evaluating top {kth} applicable predicted templates")
            # only take top kth and multiply by applicability matrix
            appl_mtrx = np.zeros_like(y_preds, dtype=bool)
            appl_mtrx[appl[0], appl[1]] = 1

            appl_and_topkth = ([], [])
            for row in range(len(y_preds)):
                argpreds = np.argpartition(
                    -(y_preds[row] * appl_mtrx[row]), kth, axis=0
                )[:kth]
                # if there are less than kth applicable
                mask = appl_mtrx[row][argpreds]
                argpreds = argpreds[mask]
                # if len(argpreds)!=kth:
                #    print('changed to ', len(argpreds))

                appl_and_topkth[0].extend([row for _ in range(len(argpreds))])
                appl_and_topkth[1].extend(list(argpreds))

            appl = appl_and_topkth
        ####

        print("running the templates")
        run_templates = run_templates  # memory.cache( ) ... allready cached to tmp
        prod_idx_reactants, prod_temp_reactants = run_templates(
            test_product_smarts, self.templates, appl
        )

        flat_results, flat_priors = self.sort_by_template_and_flatten(
            y_preds, prod_idx_reactants, agglo_fun=sum
        )
        # flat_priors = torch.Tensor(flat_priors)
        # try:
        #     flat_priors = flat_priors / flat_priors.sum(dim=1, keepdim=True)
        # except:
        #     pass
        # flat_priors = flat_priors.tolist()

        memory.clear(warn=False)
        return flat_results, flat_priors

    def model_call(self, X):
        return self._model_call(X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", type=str, default=None)
    parser.add_argument("--preprocess_batch", type=str, default=None)
    parser.add_argument("--preprocess_join", type=str, default=None)
    parser.add_argument("--reaction_col", type=str, default="reaction_smiles")
    parser.add_argument("--preprocess_output", type=str, default=None)
    parser.add_argument("--instance_name", type=str, default=None)

    args = parser.parse_args()

    model_instance = model_mhn_react(module_path=args.module_path)
    if args.preprocess_batch is not None:
        df = model_instance.read_csv(args.preprocess_batch)
        df = model_instance._preprocess(df, args.reaction_col)
        model_instance.write_csv(df, args.preprocess_output, str(uuid.uuid4()))

    if args.preprocess_join is not None:
        df = model_instance.gather_batches(args.preprocess_join)
        df = model_instance.preprocess(df, args.reaction_col, batched=True)
        model_instance.preprocess_store(df, args.preprocess_output, args.instance_name)
