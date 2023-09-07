import random
from typing import Generator, Union

import pandas as pd
from rdkit import Chem

RANDOM_SEED = 42


def convert_to_mol(r):
    """Convert SMILES string to RDKit molecule then back to SMILES string.
    This is to ensure that the SMILES string is canonicalized and atom mapping removed."""
    r = Chem.MolFromSmiles(r)
    if r is None:
        return None
    atom_mapped = False
    for a in r.GetAtoms():
        if a.HasProp("molAtomMapNumber"):
            atom_mapped = True
        a.ClearProp("molAtomMapNumber")
    if atom_mapped:
        r = Chem.MolFromSmiles(Chem.MolToSmiles(r))
        if r is None:
            return None
    return Chem.MolToSmiles(r)


def canonicalize_smiles(smiles, sort=False):
    """Canonicalize SMILES string."""
    if smiles is None:
        return None
    smiles = [convert_to_mol(smi) for smi in smiles.split(".")]
    try:
        if sort:
            smiles = sorted(smiles)
        return ".".join(smiles)
    except:
        return None


def split_reactions(rxn):
    """Split reaction SMILES string into reactants and products."""
    reactants, spectators, products = list(zip(*[s.split(">") for s in rxn]))
    return reactants, spectators, products


def chunk_data(df: pd.DataFrame, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
    """Split pd.DataFrame into chunks and yield them.

    Args:
        df: pd.DataFrame to be split
        chunk_size: size of each chunk

    Returns:
        Generator[pd.DataFrame, None, None]: Generator of chunks
    """
    for i in range(0, len(df), chunk_size):
        yield df[i : i + chunk_size]


# Subsample n rows from a dataframe
def subsample(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """Subsample n rows from a dataframe.

    Args:
        data: pd.DataFrame to be subsampled
        n: number of rows to be subsampled

    Returns:
        pd.DataFrame: subsampled dataframe
    """

    return data.sample(n=n, random_state=RANDOM_SEED)
