import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from copy import deepcopy
from utils import load_data, EvaluationResults
from typing import Dict, List, Tuple, Any, TypeVar


T = TypeVar('T')


class Trainer:
    pathogen: str
    n_splits: int
    sites: List[str]
    years: List[int]

    base_results: Dict[str, pd.DataFrame]
    smote_results: Dict[str, pd.DataFrame]

    def __init__(
            self,
            pathogen: str,
            n_splits: int,
            sites: List[str],
            years: List[int],
    ):
        """
        Init Trainer

        Args:
            pathogen: target pathogen
            n_splits: the number of folds
            sites: DRIAMS sites, can be 'A' | 'B' | 'C' | 'D'
            years: the years of the data to load, can be 2015 | 2016 | 2017 | 2018
        """
        self.pathogen = pathogen
        self.n_splits = n_splits
        self.sites = sites
        self.years = years
        self.base_results = {}
        self.smote_results = {}

    def fit(self, drug: str, model: T, **kwargs) -> Tuple[T, T]:
        """
        Fit the model to the specified drug dataset

        Args:
            drug: name of the drug
            model: model to fit

        Returns:
            (fitted model w/o SMOTE, fitted model w/ SMOTE)
        """
        print(f'Loading {drug}...')

        X, y = load_data(
            pathogen=self.pathogen,
            drug=drug,
            sites=self.sites,
            years=self.years,
        )

        print('Training w/o SMOTE...')
        base_model, base_res = self._train_and_eval(X, y, model, False, **kwargs)
        self.base_results[drug] = base_res
        print('Training w/ SMOTE...')
        smote_model, smote_res = self._train_and_eval(X, y, model, True, **kwargs)
        self.smote_results[drug] = smote_res

        return base_model, smote_model

    def collect_results(self) -> EvaluationResults:
        """
        Collect training results

        Returns: an instance of `EvaluationResults`

        See also: EvaluationResults
        """
        return EvaluationResults(self.base_results, self.smote_results)

    def _train_and_eval(
            self,
            X: np.ndarray,
            y: np.ndarray,
            model: T,
            use_smote=False,
            **kwargs,
    ) -> Tuple[T, pd.DataFrame]:
        """
        Train and evaluate the model

        Args:
            X: input data
            y: output labels
            model: model to train
            use_smote: whether to use SMOTE for oversampling

        Returns: the result data frame
        """
        results = {
            'AUROC': [],
            'Accuracy': [],
            'F1 Score': [],
        }

        # clone the model to avoid polluting it
        model = deepcopy(model)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        i = 0

        for train_index, test_index in skf.split(X, y):
            print(f'Fold {i + 1}/{self.n_splits}...')
            i += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if use_smote:
                X_train, y_train = SMOTE().fit_resample(X_train, y_train)

            model.fit(X_train, y_train, **kwargs)
            y_pred = model.predict(X_test).flatten()

            # ensure compatibility with keras
            y_pred[y_pred <= 0.5] = 0
            y_pred[y_pred > 0.5] = 1

            auc = roc_auc_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            results['AUROC'].append(auc)
            results['Accuracy'].append(acc)
            results['F1 Score'].append(f1)
            print(f'AUC={auc}, ACC={acc}, f1={f1}')

        return model, pd.DataFrame(results)

