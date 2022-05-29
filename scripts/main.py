import os
import time
import warnings

from tqdm import tqdm

from data_generator import Generator
from utils import groupedFeaturePowerset, HiddenPrints, getFeatureDict, getATE, getATT, getATC, getFeatureDictValues

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


def generate_ehv_bar_plot(data, labelx="", save=True):
    n = list(data.keys())
    values = list(data.values())

    # creating the bar plot
    plt.bar(n, values, label="Difference of PSM ATE and True ATE")
    # plt.axhline(y=ground_truth, color='r', linestyle='--', label="True ATE Value")

    plt.title("Error when each variable is hidden separately (ATE)")
    plt.xlabel("Feature hidden" + labelx)
    plt.ylabel("MAE")
    # plt.legend(loc="upper left")?
    plt.ylim([0, 0.6])

    # make file
    if save:
        directory = f'figures/error_hidden_variables'
        os.makedirs(directory, exist_ok=True)
        filename = f'/ehv_{time.ctime()}'.replace(' ', '_').replace(':', '-')
        plt.savefig(directory + filename)
        print("Figure generated")
    else:
        plt.show()


def generate_basic_data(population, dimensions=5, gt_ate=1, save=True):
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
    df = g.generate_data(population, save_data=save)
    # print("Data generation done")
    return df


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

def generate_l1no_ome_data(population, dimensions=5, gt_ate=1, save=True):
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
    return g.generate_data(population, save_data=save)
    # print("Data generation done")


def generate_l1no_ote_data(population, dimensions=5, gt_ate=1, save=True):
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
    return g.generate_data(population, save_data=save)
    # print("Data generation done")


def generate_l1no_otp_data(population, dimensions=5, gt_ate=1, save=True):
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
    return g.generate_data(population, save_data=save)
    # print("Data generation done")


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


def experiment_effect_of_hidden_variables_ATE(df, gt, dimensions=5):
    psm = PropensityScoreMatching()
    results = {}
    all_f = getFeatureDict(dimensions)
    for i in range(dimensions):
        fd = all_f.copy()
        fd.pop(f'feature_{i}')
        with HiddenPrints():
            psm_ate = psm.estimate_ATE(df, "treatment", "outcome", fd)
        results[f'feature_{i}'] = np.abs(psm_ate - gt)

    return results


def experiment_4_effect_of_hidden_variables_ATE(iterations, population, dimension, gte):
    resultsBasic = []
    for d in range(iterations):
        data = generate_basic_data(population, dimension, gte, save=False)
        resultsBasic.append(list(experiment_effect_of_hidden_variables_ATE(data, getATE(data), dimension).values()))

    resultsOME = []
    for d in range(iterations):
        data = generate_l1no_ome_data(population, dimension, gte, save=False)
        resultsOME.append(list(experiment_effect_of_hidden_variables_ATE(data, getATE(data), dimension).values()))

    resultsOTE = []
    for d in range(iterations):
        data = generate_l1no_ote_data(population, dimension, gte, save=False)
        resultsOTE.append(list(experiment_effect_of_hidden_variables_ATE(data, getATE(data), dimension).values()))

    resultsOTP = []
    for d in range(iterations):
        data = generate_l1no_otp_data(population, dimension, gte, save=False)
        resultsOTP.append(list(experiment_effect_of_hidden_variables_ATE(data, getATE(data), dimension).values()))

    # res = getFeatureDictValues(
    #     [np.mean(resultsBasic, 0), np.mean(resultsOME, 0), np.mean(resultsOTE, 0), np.mean(resultsOTP, 0)])

    # generate_ehv4_bar_plot(res)
    return np.mean(resultsBasic, 0), np.mean(resultsOME, 0), np.mean(resultsOTE, 0), np.mean(resultsOTP, 0)


if __name__ == '__main__':
    # data = pd.read_csv("data/data_dump_basic_5f/generated_dataFri_May_20_11-39-10_2022.csv")
    f_dimensions = 5
    trueATE = 1
    pop = 2500
    iterations = 10

    # gt_ate = getATE(data)
    # gt_att = getATT(data)
    # gt_atc = getATC(data)
    # psm = PropensityScoreMatching()
    # all_f = getFeatureDict(f_dimensions)
    # psm_ate = psm.estimate_ATE(data, "treatment", "outcome", all_f)
    # psm_att = psm.estimate_ATT(data, "treatment", "outcome", all_f)
    # psm_atc = psm.estimate_ATC(data, "treatment", "outcome", all_f)
    #
    # print("ATE", gt_ate, psm_ate)
    # print("ATT", gt_att, psm_att)
    # print("ATC", gt_atc, psm_atc)

    # # Generate evh base graph
    # resultsBasic = []
    # for d in range(iterations):
    #     data = generate_basic_data(pop, f_dimensions, trueATE, save=False)
    #     resultsBasic.append(list(experiment_effect_of_hidden_variables_ATE(data, getATE(data), f_dimensions).values()))
    # res = np.mean(resultsBasic, 0)
    #
    # generate_ehv_bar_plot(getFeatureDictValues(res), labelx=" (BASE)", save=False)

    # # Generate evh base graph
    # resultsBasic = []
    # for d in tqdm(range(iterations)):
    #     data = generate_basic_data(pop, f_dimensions, trueATE, save=False)
    #     resultsBasic.append(list(experiment_effect_of_hidden_variables_ATE(data, trueATE, f_dimensions).values()))
    # res = np.mean(resultsBasic, 0)
    #
    # generate_ehv_bar_plot(getFeatureDictValues(res), labelx=" (BASE)", save=False)

    # # Generate evh f_4 only main effect graph
    # results = []
    # for d in tqdm(range(iterations)):
    #     data = generate_l1no_ome_data(pop, f_dimensions, trueATE, save=False)
    #     results.append(list(experiment_effect_of_hidden_variables_ATE(data, trueATE, f_dimensions).values()))
    # res = np.mean(results, 0)
    #
    # generate_ehv_bar_plot(getFeatureDictValues(res), labelx=" (F4 OME)", save=False)

    # Generate evh f_4 only treatment effect graph
    # results = []
    # for d in tqdm(range(iterations)):
    #     data = generate_l1no_ote_data(pop, f_dimensions, trueATE, save=False)
    #     results.append(list(experiment_effect_of_hidden_variables_ATE(data, trueATE, f_dimensions).values()))
    # res = np.mean(results, 0)
    #
    # generate_ehv_bar_plot(getFeatureDictValues(res), labelx=" (F4 OTE)", save=False)

    # Generate evh f_4 only treatment propensity graph
    # results = []
    # for d in tqdm(range(iterations)):
    #     data = generate_l1no_otp_data(pop, f_dimensions, trueATE, save=False)
    #     results.append(list(experiment_effect_of_hidden_variables_ATE(data, trueATE, f_dimensions).values()))
    # res = np.mean(results, 0)
    #
    # generate_ehv_bar_plot(getFeatureDictValues(res), labelx=" (F4 OTP)", save=False)
