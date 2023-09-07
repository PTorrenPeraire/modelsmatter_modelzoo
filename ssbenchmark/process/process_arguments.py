from argparse import ArgumentParser

import configargparse


# parse arguments using configargparse
def set_args():
    # Document function
    """
    Set arguments for the benchmark.
    Uses either commandline or config file arguments. Commandline arguments take precedence over config file arguments.

    Returns:
        argparse.Namespace: Arguments.
    """

    parser = configargparse.ArgParser()
    # add argument for config file
    parser.add_argument("-c", "--config", required=False, is_config_file=True, help="config file path")
    parser.add_argument("-ms", "--models", dest="models", nargs="+", help="model(s) to run")
    parser.add_argument(
        "-rt", "--run_type", dest="run_type", default="single_step", help="type of analysis to be carried out"
    )
    parser.add_argument("-d", "--data_path", dest="data_path", help="Location of data file")
    parser.add_argument("-n", "--name_instance", dest="name_instance")
    parser.add_argument("-pre", "--preprocess", dest="preprocess", default=False, help="run data preprocessing")
    parser.add_argument("-post", "--postprocess", dest="postprocess", default=False, help="run data postprocessing")

    parser.add_argument(
        "-pr",
        "--preprocess_root",
        dest="preprocess_root",
        default="./data",
        help="Location where preprocessed data will be stored",
    )

    parser.add_argument(
        "-pstr",
        "--postprocess_root",
        dest="postprocess_root",
        default="./output",
        help="Location where model results are stored",
    )
    parser.add_argument("-ts", "--target_smiles", dest="target_smiles", default=None, help="File path of true smiles")
    parser.add_argument(
        "-kf", "--kfold", dest="kfold", default=None, help="k number of folds to run, defaults to None for no k-fold"
    )
    parser.add_argument(
        "-rxn",
        "--reaction_column",
        dest="reaction_col",
        default="rxn_smiles",
        help="Name of column where reactions are found",
    )
    parser.add_argument("-r", "--store_results", dest="store_results", default=None, help="Root where to store results")

    parser.add_argument(
        "-cs", "--chunk_size", dest="chunk_size", type=int, default=None, help="Size of data batches to create"
    )
    parser.add_argument("-br", "--batch_root", dest="batch_root", default=None, help="Root where to store batches")

    parser.add_argument("-ss", "--subsample", dest="subsample", type=int, default=-1, help="Subsample data")
    args = parser.parse_args()

    # Output the selected arguments
    print("Selected arguments:")
    parser.print_values()
    return args
