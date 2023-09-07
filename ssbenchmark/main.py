import collections
import os

import numpy as np
import pandas as pd
import yaml

from ssbenchmark.process.data_processing import DataInput, DataOutput
from ssbenchmark.process.process_arguments import set_args
from ssbenchmark.ssmodels.base_model import ModelChoice
from ssbenchmark.utils import chunk_data


class SingleStepBenchmark:
    """
    Main class to benchmark single step models. Runs full pipeline for stated single step models.

    Args:
        models (list): List of single step models to benchmark.
        data_path (str): Path to raw reaction data.
        preprocess (bool): Whether to preprocess data.
        postprocess (bool): Whether to postprocess data (once model run(s) have been carried out).
    """

    def __init__(self, models, data_path, preprocess, postprocess):

        self.data_path = data_path
        self.models = [m.strip() for m in models]

        self.preprocess = preprocess
        self.postprocess = postprocess

    def run(
        self,
        models,
        preprocess_root,
        postprocess_root,
        target_smiles,
        name_instance,
        reaction_col="rxn_smiles",
        kfold=None,
        store_results=None,
    ):
        """
        Run full pipeline for single step models.

        Args:
            models (list): List of single step models to benchmark.
            preprocess_root (str): Path to store preprocessed data.
            postprocess_root (str): Path to store postprocessed data.
            target_smiles (str): Path to target SMILES.
            name_instance (str): Name of instance.
            reaction_col (str): Name of reaction column.
            kfold (int): Number of folds for k-fold cross validation (default: None).
            store_results (str): Path to store results.


        """
        self.models = []
        for m in models:
            ModelChoice.validate_choice(m)
            model = ModelChoice.get_choice(m)
            self.models.append(model())

        if self.preprocess:
            print("Preprocessing data")
            data_instance = DataInput(self.data_path)

            if kfold is None:
                data = data_instance.split_data()
                data_instance.store_split(data, preprocess_root, name_instance)
                data_instance.store_test_target(
                    data, reaction_col, preprocess_root, name_instance
                )  # TODO: Restructure this
                for m in self.models:
                    d = data.copy()
                    d = m.process_input(d, reaction_col)
                    m.preprocess_store(d, preprocess_root, name_instance)
            else:
                for i, d_fold in enumerate(data_instance.k_split_data(k=kfold)):
                    data_instance.store_split(
                        d_fold, preprocess_root, name_instance + f"{i:02}"
                    )
                    data_instance.store_test_target(
                        d_fold, reaction_col, preprocess_root, name_instance + f"{i:02}"
                    )  # TODO: Restructure this
                    for m in self.models:
                        d = d_fold.copy()
                        d = m.process_input(d, reaction_col)
                        m.preprocess_store(
                            d, preprocess_root, name_instance + f"{i:02}"
                        )

        if self.postprocess:
            results = collections.defaultdict(dict)
            for k in [1, 3, 5, 10, 50]:
                do_instance = DataOutput(target_smiles)

                for m in self.models:
                    # d = m.process_output(postprocess_root, name_instance, k)
                    try:  # TODO: THis shouldn't be a try except, this should be a check in the process output
                        d = m.process_output(postprocess_root, name_instance, k)
                    except:
                        results[m.model_name][k] = np.nan
                        continue
                    if d is None:
                        continue
                    results[m.model_name][k] = round(
                        (do_instance.compare_smiles(d) * 100).item(), 2
                    )
                # print(f"RESULTS top-{k}", results)
            print("RESULTS", results)
            if store_results:
                results_dict = pd.DataFrame.from_dict(results)
                results_dict.to_csv(
                    os.path.join(store_results, f"{name_instance}_results.csv"),
                    index=False,
                )


class SingleStepBatch:
    def __init__(self, data_path):
        self.data_path = data_path

    def run(
        self,
        name_instance=None,
        preprocess_root=None,
        reaction_col="rxn_smiles",
        batch_root=None,
        chunk_size=None,
        subsample=-1,
    ):
        data_instance = DataInput(self.data_path, subsample_n=subsample)

        if f"rawdata_{name_instance}" in self.data_path:
            print("Using pre-split data")
            data = data_instance.data
        else:
            data = data_instance.split_data()
            data_instance.store_split(data, preprocess_root, name_instance)
            data_instance.store_test_target(
                data, reaction_col, preprocess_root, name_instance
            )  # TODO: Restructure this

        # If there are files in batch_root delete them, if batch_root does not exist create folder
        if batch_root is not None:
            if os.path.exists(batch_root):
                for f in os.listdir(batch_root):
                    if f.startswith(f"{name_instance}_batch"):
                        os.remove(os.path.join(batch_root, f))
            else:
                os.makedirs(batch_root)
            print("BATCHING DATA")
            for i, df_ in enumerate(chunk_data(data, chunk_size=chunk_size)):
                df_ = df_.reset_index(drop=True)
                df_.to_csv(
                    os.path.join(batch_root, f"{name_instance}_batch{i}.csv"),
                    index=False,
                )


def main():
    args = set_args()
    if args.run_type == "single_step":
        run_type = SingleStepBenchmark(
            args.models, args.data_path, args.preprocess, args.postprocess
        )
        run_type.run(
            models=args.models,
            preprocess_root=args.preprocess_root,
            postprocess_root=args.postprocess_root,
            target_smiles=args.target_smiles,
            name_instance=args.name_instance,
            reaction_col=args.reaction_col,
            kfold=args.kfold,
            store_results=args.store_results,
        )

    elif args.run_type == "batch":
        run_type = SingleStepBatch(args.data_path)
        run_type.run(
            preprocess_root=args.preprocess_root,
            name_instance=args.name_instance,
            reaction_col=args.reaction_col,
            batch_root=args.batch_root,
            chunk_size=args.chunk_size,
            subsample=args.subsample,
        )


if __name__ == "__main__":
    main()
