import os
import time
import warnings

from tqdm import tqdm

from data_generator import Generator
from utils import groupedFeaturePowerset, HiddenPrints, getFeatureDict

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from causality.estimation.parametric import PropensityScoreMatching


def sigmoid(arg):
    return np.exp(arg) / (1. + np.exp(arg))


def generate_envh_line_plot(data, save=True):
    n = list(data.keys())
    values = list(data.values())

    # creating the bar plot
    plt.plot(n, values, label="Difference of PSM ATE and True ATE")
    # plt.axhline(y=ground_truth, color='r', linestyle='--', label="True ATE Value")

    plt.title("Error proportional to hidden variables (ATE)")
    plt.xlabel("Number of hidden variables")
    plt.ylabel("RMSE")
    # plt.legend(loc="upper left")
    plt.ylim([0, 1.4])

    # make file
    if save:
        directory = f'figures/error_number_hidden_variables'
        os.makedirs(directory, exist_ok=True)
        filename = f'/enhv_{time.ctime()}'.replace(' ', '_').replace(':', '-')
        plt.savefig(directory + filename)
        print("Figure generated")
    else:
        plt.show()


def generate_ehv_bar_plot(data, save=True):
    n = list(data.keys())
    values = list(data.values())

    # creating the bar plot
    plt.bar(n, values, color='green', label="Difference of PSM ATE and True ATE")
    # plt.axhline(y=ground_truth, color='r', linestyle='--', label="True ATE Value")

    plt.title("Error when each variable is hidden separately (ATE)")
    plt.xlabel("Variable hidden (feature_4 only affects treatment propensity without overlap)")
    plt.ylabel("RSE")
    # plt.legend(loc="upper left")?
    plt.ylim([0, 0.25])

    # make file
    if save:
        directory = f'figures/error_hidden_variables'
        os.makedirs(directory, exist_ok=True)
        filename = f'/ehv_{time.ctime()}'.replace(' ', '_').replace(':', '-')
        plt.savefig(directory + filename)
        print("Figure generated")
    else:
        plt.show()


def generate_basic_data(population, dimensions=5, gt_ate=1):
    feature_function = lambda xs: np.sum(xs)
    main_effect = lambda xs: feature_function(xs)
    treatment_effect = lambda xs: feature_function(xs) + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * t + n
    f_distribution = [lambda: 0.5 * np.random.normal()]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"basic_{dimensions}f")
    g.generate_data(population)
    print("Data generation done")


def generate_ff_sum_half_data(population, dimensions=5, gt_ate=1):
    feature_function = lambda xs: np.sum(xs[: int(len(xs) / 2 + 1)])
    main_effect = lambda xs: feature_function(xs)
    treatment_effect = lambda xs: feature_function(xs) + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * t + n
    f_distribution = [lambda: 0.5 * np.random.normal()]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"ff_sum_half_{dimensions}f")
    g.generate_data(population)
    print("Data generation done")


def generate_ff_prod_data(population, dimensions=5, gt_ate=1):
    feature_function = lambda xs: np.prod(xs)
    main_effect = lambda xs: feature_function(xs)
    treatment_effect = lambda xs: feature_function(xs) + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * t + n
    f_distribution = [lambda: 0.5 * np.random.normal()]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"ff_prod_{dimensions}f")
    g.generate_data(population)
    print("Data generation done")


def generate_l1_ome_data(population, dimensions=5, gt_ate=1):
    feature_function = lambda xs: np.sum(xs)
    main_effect = lambda xs: feature_function(xs)
    treatment_effect = lambda xs: feature_function(xs[:dimensions - 1]) + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs[:dimensions - 1]) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * t + n
    f_distribution = [lambda: 0.5 * np.random.normal()]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"l1_ome_{dimensions}f")
    g.generate_data(population)
    print("Data generation done")


def generate_l1_ote_data(population, dimensions=5, gt_ate=1):
    feature_function = lambda xs: np.sum(xs)
    main_effect = lambda xs: feature_function(xs[:dimensions - 1])
    treatment_effect = lambda xs: feature_function(xs) + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs[:dimensions - 1]) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * t + n
    f_distribution = [lambda: 0.5 * np.random.normal()]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"l1_ote_{dimensions}f")
    g.generate_data(population)
    print("Data generation done")


def generate_l1_otp_data(population, dimensions=5, gt_ate=1):
    feature_function = lambda xs: np.sum(xs)
    main_effect = lambda xs: feature_function(xs[:dimensions - 1])
    treatment_effect = lambda xs: feature_function(xs[:dimensions - 1]) + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * t + n
    f_distribution = [lambda: 0.5 * np.random.normal()]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"l1_otp_{dimensions}f")
    g.generate_data(population)
    print("Data generation done")


# def generate_l1_otetp_data(population, dimensions=5, gt_ate=1):
#     feature_function = lambda xs: np.sum(xs)
#     main_effect = lambda xs: feature_function(xs[:dimensions - 1])
#     treatment_effect = lambda xs: feature_function(xs) + gt_ate
#     treatment_propensity = lambda xs: sigmoid(feature_function(xs) + np.random.normal())
#     noise = lambda: np.random.normal()
#     treatment_function = lambda p, n: np.random.binomial(1, p)
#     outcome_function = lambda me, t, te, n: me + te * t + n
#     f_distribution = [lambda: 0.5 * np.random.normal()]
#     g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
#                   outcome_function, dimensions, f_distribution, name=f"l1_otetp_{dimensions}f")
#     g.generate_data(population)
#     print("Data generation done")
#
#
# def generate_l1_omete_data(population, dimensions=5, gt_ate=1):
#     feature_function = lambda xs: np.sum(xs)
#     main_effect = lambda xs: feature_function(xs)
#     treatment_effect = lambda xs: feature_function(xs) + gt_ate
#     treatment_propensity = lambda xs: sigmoid(feature_function(xs[:dimensions - 1]) + np.random.normal())
#     noise = lambda: np.random.normal()
#     treatment_function = lambda p, n: np.random.binomial(1, p)
#     outcome_function = lambda me, t, te, n: me + te * t + n
#     f_distribution = [lambda: 0.5 * np.random.normal()]
#     g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
#                   outcome_function, dimensions, f_distribution, name=f"l1_omete_{dimensions}f")
#     g.generate_data(population)
#     print("Data generation done")
#
#
# def generate_l1_ometp_data(population, dimensions=5, gt_ate=1):
#     feature_function = lambda xs: np.sum(xs)
#     main_effect = lambda xs: feature_function(xs)
#     treatment_effect = lambda xs: feature_function(xs[:dimensions - 1]) + gt_ate
#     treatment_propensity = lambda xs: sigmoid(feature_function(xs) + np.random.normal())
#     noise = lambda: np.random.normal()
#     treatment_function = lambda p, n: np.random.binomial(1, p)
#     outcome_function = lambda me, t, te, n: me + te * t + n
#     f_distribution = [lambda: 0.5 * np.random.normal()]
#     g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
#                   outcome_function, dimensions, f_distribution, name=f"l1_ometp_{dimensions}f")
#     g.generate_data(population)
#     print("Data generation done")

def generate_l1no_ome_data(population, dimensions=5, gt_ate=1):
    feature_function = lambda xs: np.sum(xs)
    main_effect = lambda xs: xs[dimensions - 1]
    treatment_effect = lambda xs: feature_function(xs[:dimensions - 1]) + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs[:dimensions - 1]) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * t + n
    f_distribution = [lambda: 0.5 * np.random.normal()]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"l1no_ome_{dimensions}f")
    g.generate_data(population)
    print("Data generation done")


def generate_l1no_ote_data(population, dimensions=5, gt_ate=1):
    feature_function = lambda xs: np.sum(xs)
    main_effect = lambda xs: feature_function(xs[:dimensions - 1])
    treatment_effect = lambda xs: xs[dimensions - 1] + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs[:dimensions - 1]) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * t + n
    f_distribution = [lambda: 0.5 * np.random.normal()]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"l1no_ote_{dimensions}f")
    g.generate_data(population)
    print("Data generation done")


def generate_l1no_otp_data(population, dimensions=5, gt_ate=1):
    feature_function = lambda xs: np.sum(xs)
    main_effect = lambda xs: feature_function(xs[:dimensions - 1])
    treatment_effect = lambda xs: feature_function(xs[:dimensions - 1]) + gt_ate
    treatment_propensity = lambda xs: sigmoid(xs[dimensions - 1] + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * t + n
    f_distribution = [lambda: 0.5 * np.random.normal()]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"l1no_otp_{dimensions}f")
    g.generate_data(population)
    print("Data generation done")


def experiment_number_of_hidden_variables_ATE(df, dimensions=5, gt_ate=1):
    psm = PropensityScoreMatching()
    number_hidden_variables = 0
    results = {}
    for fdl in groupedFeaturePowerset(dimensions):
        ires = []
        for fd in tqdm(fdl):
            with HiddenPrints():
                psm_ate = psm.estimate_ATE(df, "treatment", "outcome", fd)
            ires.append(np.square(psm_ate - gt_ate))
        results[number_hidden_variables] = np.sqrt(np.mean(ires))
        number_hidden_variables += 1

    return results


def experiment_effect_of_hidden_variables_ATE(df, dimensions=5):
    psm = PropensityScoreMatching()
    results = {}
    all_f = getFeatureDict(dimensions)
    full_ate = psm.estimate_ATE(df, "treatment", "outcome", all_f)
    for i in tqdm(range(dimensions)):
        fd = all_f.copy()
        fd.pop(f'feature_{i}')
        with HiddenPrints():
            psm_ate = psm.estimate_ATE(df, "treatment", "outcome", fd)
        results[f'feature_{i}'] = np.sqrt(np.square(psm_ate - full_ate))

    return results


if __name__ == '__main__':
    data = pd.read_csv("data/data_dump_l1no_otp_5f/generated_dataFri_May_20_13-03-08_2022.csv")
    f_dimensions = 5
    trueATE = 1
    pop = 3000

    # generate_l1no_ome_data(pop, f_dimensions, trueATE)
    # generate_l1no_ote_data(pop, f_dimensions, trueATE)
    # generate_l1no_otp_data(pop, f_dimensions, trueATE)

    res = experiment_effect_of_hidden_variables_ATE(data, f_dimensions)
    generate_ehv_bar_plot(res, save=False)
