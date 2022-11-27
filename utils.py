import os
import pandas as pd
import numpy as np
from typing import Tuple


def _load_bins(
        df: pd.DataFrame,
        dataset='A',
        year=2015,
) -> np.ndarray:
    """
    Load binned data corresponding to the dataframe

    Args:
        df: source dataframe
        dataset: DRIAMS datasets, can be 'A' | 'B' | 'C' | 'D'
        year: the years of the datasets to load, can be 2015 | 2016 | 2017 | 2018

    Returns:
        bins
    """
    bins = []

    for code in df['code']:
        bins.append(np.loadtxt(
            f'./archive/DRIAMS-{dataset}/binned_6000/{year}/{code}.txt',
            skiprows=1,
            usecols=1,
        ))

    return np.array(bins)


def load_data(
        pathogen='Escherichia coli',
        drug='Ceftriaxone',
        datasets=['A'],
        years=[2015, 2016, 2017, 2018],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and build dataset from `./archive/DRIAMS-{datasets}/id/{years}/{years}_clean.csv`

    Args:
        pathogen: target pathogen
        drug: target drug
        datasets: DRIAMS datasets, can be any combination of ['A', 'B', 'C', 'D']
        years: the years of the datasets to load, can be any combination of [2015, 2016, 2017, 2018]

    Returns:
        X, y
    """
    X = []
    y = []

    for ds in datasets:
        for yr in years:
            csv_file = f'./archive/DRIAMS-{ds}/id/{yr}/{yr}_clean.csv'

            # skip non-existent files
            if not os.path.exists(csv_file):
                continue

            df = pd.read_csv(csv_file, dtype='str')
            # select data
            df = df[df['species'] == pathogen]
            df = df[(df[drug] == 'S') | (df[drug] == 'R')]
            # drop NaN rows
            df = df[['code', drug]].dropna()

            # X
            X.append(_load_bins(df, ds, yr))

            # y
            labels = df[drug]
            labels[labels == 'S'] = 0
            labels[labels == 'R'] = 1
            y.append(labels.to_numpy(dtype=int))

    return np.concatenate(X), np.concatenate(y)


# tests
if __name__ == '__main__':
    kpn_X, kpn_y = load_data(
        pathogen='Klebsiella pneumoniae',
        drug='Ciprofloxacin',
        datasets=['A', 'B', 'C', 'D'],
        years=[2015, 2016, 2017, 2018],
    )

    print(kpn_X.shape)
    print(kpn_y.shape)
