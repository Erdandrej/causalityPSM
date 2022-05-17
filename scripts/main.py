from enum import Enum
from data_generator import Generator
from utils import select_features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from causality.estimation.parametric import PropensityScoreMatching


def sigmoid(arg):
    return np.exp(arg) / (1. + np.exp(arg))

def generatePlot(data):
    n = list(data.keys())
    values = list(data.values())

    # creating the bar plot
    plt.plot(n, values)

    plt.title("Error proportional to hidden variables")
    plt.xlabel("Number of hidden variables")
    plt.ylabel("Error")
    plt.show()



if __name__ == '__main__':
    trueATE = 1
    # main_effect = lambda xs: np.sum(xs)
    # treatment_effect = lambda xs: np.sum(xs) + trueATE.
    # treatment_propensity = lambda xs: sigmoid(np.sum(xs) + np.random.normal())
    # noise = lambda: np.random.normal()
    # treatment_function = lambda p, n: np.random.binomial(1, p)
    # outcome_function = lambda me, t, te, n: me + te * t + n
    f_dimensions = 10
    # f_distribution = [lambda: 0.5 * np.random.normal()]
    # g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
    #               outcome_function, f_dimensions, f_distribution)
    # g.generate_data(1000)

    psm = PropensityScoreMatching()
    df = pd.read_csv("data/data_dump_116905378570/generated_dataTue_May_17_13-07-59_2022.csv")
    sel_df = select_features(df, f_dimensions)

    all_f = {f'feature_{i}': 'c' for i in range(f_dimensions)}
    psmATE = psm.estimate_ATE(df, "treatment", "outcome", all_f)

    results = {}
    for x in range(f_dimensions):
        psmATE = psm.estimate_ATE(df, "treatment", "outcome", all_f)
        results[x] = psmATE - trueATE
        all_f.pop(f'feature_{x}')
    results[f_dimensions] = psm.estimate_ATE(df, "treatment", "outcome", all_f)

    generatePlot(results)
    