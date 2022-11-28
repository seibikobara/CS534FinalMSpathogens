import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class EvaluationResults:
    """
    results = {
        [drug]: [data]
    }

    dataframe layout:
        - cols: metrics
        - rows: folds
    """
    base_results: Dict[str, pd.DataFrame]
    smote_results: Dict[str, pd.DataFrame]

    def bar_plot(
            self,
            title: str,
            drugs: List[str] = None,
            metrics: List[str] = None,
            y_lim: Tuple[float, float] = None,
            show: bool = True,
            save_as: str = None,
            *args,
            **kwargs,
    ) -> None:
        """
        Plot a bar graph

        Args:
            title: graph's title
            drugs: list of drugs to plot, set to `None` to plot all drugs
            metrics: list of metrics to plot, set to `None` to plot all metrics
            y_lim: view limits for y-axis
            show: whether to display the image
            save_as: output file name
            *args, **kwargs: same as `plt.plot()`
        """
        fig, axes = self._subplots(len(self.base_results), *args, **kwargs)
        fig.suptitle(title)

        if drugs is None:
            drugs = list(self.base_results)

        if metrics is None:
            metrics = self.base_results[drugs[0]].columns

        for i, drug in enumerate(drugs):
            # get the subplot
            ax = axes[i]
            # get the dataframes
            df_base_med = self.base_results[drug].median()
            df_smote_med = self.smote_results[drug].median()
            # create plotting dataframe
            df_plt = pd.DataFrame({
                'Without SMOTE': df_base_med[metrics],
                'With SMOTE': df_smote_med[metrics],
            })
            # plot
            df_plt.plot.bar(
                ax=ax,
                title=drug,
                width=0.9,
                rot=0,
                legend=False,
            )
            # add bar labels
            ax.bar_label(ax.containers[0], fmt='%.3f', padding=2, fontsize=8)
            ax.bar_label(ax.containers[1], fmt='%.3f', padding=2, fontsize=8)
            # add y label
            ax.set_ylabel('Scores')
            # set y limits
            ax.set_ylim(y_lim)

        # draw legend
        legend_target = axes[0] if len(axes) == 1 else fig
        legend_target.legend(['Without SMOTE', 'With SMOTE'])

        self.save_plot(save_as)

        if show:
            plt.show()

    def box_plot(
            self,
            title: str,
            drugs: List[str] = None,
            metrics: List[str] = None,
            y_lim: Tuple[float, float] = None,
            show: bool = True,
            save_as: str = None,
            *args,
            **kwargs,
    ) -> None:
        """
        Plot a box graph

        Args:
            title: graph's title
            drugs: list of drugs to plot, set to `None` to plot all drugs
            metrics: list of metrics to plot, set to `None` to plot all metrics
            y_lim: view limits for y-axis
            show: whether to display the image
            save_as: output file name
            *args, **kwargs: same as `plt.plot()`
        """
        color_base = '#D7191C'
        color_smote = '#2C7BB6'

        if drugs is None:
            drugs = list(self.base_results)

        if metrics is None:
            metrics = self.base_results[drugs[0]].columns

        fig, axes = self._subplots(len(metrics), *args, **kwargs)
        fig.suptitle(title)

        for i, m in enumerate(metrics):
            # get the subplot
            ax = axes[i]
            # get the scores
            scores_base = np.transpose([
                self.base_results[d][m].to_numpy() for d in drugs
            ])
            scores_smote = np.transpose([
                self.smote_results[d][m].to_numpy() for d in drugs
            ])
            # plot
            pos = np.arange(scores_base.shape[1]) * 2.0
            bp_base = ax.boxplot(
                scores_base,
                sym='',
                widths=0.6,
                positions=pos - 0.4,
            )
            bp_smote = ax.boxplot(
                scores_smote,
                sym='',
                widths=0.6,
                positions=pos + 0.4,
            )
            # set plotting colors
            self._set_box_color(bp_base, color_base)
            self._set_box_color(bp_smote, color_smote)
            # draw x-ticks
            ax.set_xticks(pos, drugs)
            ax.set_xlim(-2, scores_base.shape[1] * 2)
            # set y limits
            ax.set_ylim(y_lim)
            # set sub-fig's title
            ax.set_title(m)
            # add y label
            ax.set_ylabel('Scores')

        # draw temporary red and blue lines and use them to create a legend
        axes[0].plot([], c=color_base, label='Without SMOTE')
        axes[0].plot([], c=color_smote, label='With SMOTE')

        # draw legend
        legend_target = axes[0] if len(axes) == 1 else fig
        legend_target.legend()

        self.save_plot(save_as)

        if show:
            plt.show()

    def save_to(self, directory: str):
        """
        Export the results as csv files to the specified directory

        Args:
            directory: output directory

        Returns: None
        """
        if directory is None:
            return

        os.makedirs(directory, exist_ok=True)

        for drug in self.base_results:
            df = self.base_results[drug]
            df.to_csv(f'{directory}/{drug}_base.csv')

        for drug in self.smote_results:
            df = self.smote_results[drug]
            df.to_csv(f'{directory}/{drug}_smote.csv')

    @staticmethod
    def save_plot(filepath: str):
        if filepath is None:
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, facecolor='white')

    @staticmethod
    def _subplots(count, *args, **kwargs):
        fig, axes = plt.subplots(*args, **kwargs)

        # wrap axes into ndarray
        if not isinstance(axes, np.ndarray):
            arr = np.zeros(1, dtype=object)
            arr[0] = axes
            axes = arr

        # flatten
        axes = axes.flatten()

        # remove empty subplots
        if len(axes) > count:
            for ax in axes[count:]:
                fig.delaxes(ax)

            axes = axes[:count]

        return fig, axes

    @staticmethod
    def _set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)


def _load_bins(
        df: pd.DataFrame,
        site='A',
        year=2015,
) -> np.ndarray:
    """
    Load binned data corresponding to the dataframe

    Args:
        df: source dataframe
        site: DRIAMS sites, can be 'A' | 'B' | 'C' | 'D'
        year: the years of the sites to load, can be 2015 | 2016 | 2017 | 2018

    Returns:
        bins
    """
    bins = []

    for code in df['code']:
        bins.append(np.loadtxt(
            f'./archive/DRIAMS-{site}/binned_6000/{year}/{code}.txt',
            skiprows=1,
            usecols=1,
        ))

    return np.array(bins)


def load_data(
        pathogen='Escherichia coli',
        drug='Ceftriaxone',
        sites=['A'],
        years=[2015, 2016, 2017, 2018],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and build dataset from `./archive/DRIAMS-{sites}/id/{years}/{years}_clean.csv`

    Args:
        pathogen: target pathogen
        drug: target drug
        sites: DRIAMS sites, can be any combination of ['A', 'B', 'C', 'D']
        years: the years of the sites to load, can be any combination of [2015, 2016, 2017, 2018]

    Returns:
        X, y
    """
    X = []
    y = []

    for ds in sites:
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
    def generate_results(smote=False):
        drugs = ['Ceftriaxone', 'Ciprofloxacin', 'Cefepime']
        metrics = ['AUROC', 'Accuracy', 'F1 Score']
        lower = 0.65 if smote else 0.55
        res = {}

        for d in drugs:
            data = {}
            for m in metrics:
                data[m] = lower + 0.2 * np.random.randn(10)
            res[d] = pd.DataFrame(data)

        return res

    results = EvaluationResults(
        generate_results(),
        generate_results(True),
    )

    results.bar_plot(
        nrows=2,
        ncols=2,
        sharey=True,
        figsize=(12, 8),
        title='Bar Graph Test',
        save_as='./plots/bar_test.png',
        metrics=['AUROC', 'F1 Score'],
    )

    results.box_plot(
        title='Box Graph Test',
        metrics=['AUROC'],
        save_as='./plots/box_test.png',
    )

    results.save_to('./results/test')
