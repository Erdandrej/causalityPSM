import os
import time
import warnings

from tqdm import tqdm

from data_generator import Generator
from utils import groupedFeaturePowerset, HiddenPrints

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from causality.estimation.parametric import PropensityScoreMatching


def sigmoid(arg):
    return np.exp(arg) / (1. + np.exp(arg))


def generateENHVLinePlot(data, save=True):
    n = list(data.keys())
    values = list(data.values())

    # creating the bar plot
    plt.plot(n, values, label="Difference of PSM ATE and True ATE")
    # plt.axhline(y=ground_truth, color='r', linestyle='--', label="True ATE Value")

    plt.title("Error proportional to hidden variables (ATE)")
    plt.xlabel("Number of hidden variables")
    plt.ylabel("RMSE")
    # plt.legend(loc="upper left")

    # make file
    if save:
        directory = f'figures/error_number_hidden_variables'
        os.makedirs(directory, exist_ok=True)
        filename = f'/enhv_{time.ctime()}'.replace(' ', '_').replace(':', '-')
        plt.savefig(directory + filename)
        print("Figure generated")
    else:
        plt.show()


def generate_basic_data(population, dimensions=5, gt_ate=1):
    main_effect = lambda xs: np.sum(xs)
    treatment_effect = lambda xs: np.sum(xs) + gt_ate
    treatment_propensity = lambda xs: sigmoid(np.sum(xs) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * t + n
    f_distribution = [lambda: 0.5 * np.random.normal()]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"basic_{dimensions}f")
    g.generate_data(population)
    print("Data generation done")


def experiment_number_of_hidden_variables_ATE(df, dimensions=5, gt_ate=1):
    psm = PropensityScoreMatching()
    number_hidden_variables = 0
    results = {}
    for fdl in tqdm(groupedFeaturePowerset(dimensions)):
        ires = []
        for fd in fdl:
            with HiddenPrints():
                psm_ate = psm.estimate_ATE(df, "treatment", "outcome", fd)
            ires.append(np.square(psm_ate - gt_ate))
        results[number_hidden_variables] = np.sqrt(np.mean(ires))
        number_hidden_variables += 1

    return results


if __name__ == '__main__':
    data = pd.read_csv("data/data_dump_basic_5f/generated_dataThu_May_19_17-53-03_2022.csv")
    f_dimensions = 5
    trueATE = 1
    pop = 2000

    # generate_basic_data(pop, f_dimensions, trueATE)

    res = experiment_number_of_hidden_variables_ATE(data, f_dimensions, trueATE)
    generateENHVLinePlot(res)
