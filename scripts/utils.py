import sys
import os

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
from typing import *
from itertools import combinations, chain, groupby
from functools import partial


class HiddenPrints:
    """
    Class to block printing.
    Taken from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def save_pandas_table(dir: str, df: pd.DataFrame):
    plt.clf()
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table(ax, df)
    plt.savefig(dir)
    df.to_csv(dir + '.csv')


def compact_dict_print(dict: Dict[str, Any]):
    result = ''
    for index, key in enumerate(dict):
        result += f'{key}={dict[key]}{"," if index < len(dict) - 1 else ""}'.replace(' ', '_').replace(':', '-')
    return result


def select_features(df: pd.DataFrame, dim: int = -1):
    if dim == -1:
        return df[[name for name in df.columns if 'feature' in name]]
    return df[[f'feature_{i}' for i in range(dim)]]


def generate_coverage_of_model_graph(model, df: pd.DataFrame, save_dir: str):
    plt.clf()
    feature_one = df['feature_0']
    feature_two = df['feature_1']
    predictions = model.estimate_causal_effect(select_features(df))
    maximal = df['treatment_effect'].max()
    minimal = df['treatment_effect'].min()
    color_function = lambda i: [0,
                                max(0, min(1, (predictions[i] - minimal) / (maximal - minimal + 0.01))),
                                max(0, min(1, 1 - (predictions[i] - minimal) / (maximal - minimal + 0.01)))]
    plt.scatter(feature_one, feature_two, c=[color_function(i) for i in df.index])
    plt.savefig(save_dir)


def getFeatureDict(dimensions):
    return {f'feature_{i}': 'c' for i in range(dimensions)}


def powerset(set_):
    return chain.from_iterable(
        map(
            partial(combinations, set_),
            range(len(set_) + 1)
        )
    )


def powerdict(dict_):
    return map(
        dict,
        powerset(dict_.items())
    )


def featurePowerset(dimensions):
    all_f = getFeatureDict(dimensions)
    return list(map(dict, powerdict(all_f)))[::-1]


def groupedFeaturePowerset(dimensions):
    fps = featurePowerset(dimensions)
    return [list(g) for k, g in groupby(fps, key=len)]


def getATE(data):
    return (data['y1'] - data['y0']).mean()


def getATT(data):
    t1_data = data[data['treatment'] == 1.0]
    return (t1_data['y1'] - t1_data['y0']).mean()


def getATC(data):
    t0_data = data[data['treatment'] == 0.0]
    return (t0_data['y1'] - t0_data['y0']).mean()
