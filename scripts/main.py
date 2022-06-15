import os
import time
import warnings
from os import popen

from tqdm import tqdm
from sklearn import linear_model

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


def generate_basic_data(population, dimensions=6, gt_ate=1, save=True):
    feature_function = lambda xs: np.sum(xs)
    main_effect = lambda xs: feature_function(xs)
    treatment_effect = lambda xs: feature_function(xs) + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * (t - 0.5) + n
    f_distribution = [lambda: np.random.normal(1, 1)]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"basic_{dimensions}f")
    df = g.generate_data(population, save_data=save)
    # print("Data generation done")
    return df


def generate_3cv_3ncf_data(population, dimensions=6, gt_ate=1, save=False):
    feature_function = lambda xs: np.sum(xs)
    main_effect = lambda xs: feature_function(xs[0:3])
    treatment_effect = lambda xs: feature_function(xs[0:3]) + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs[0:3]) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * (t - 0.5) + n
    f_distribution = [lambda: 0.5 * np.random.normal(1, 1)]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"basic_{dimensions}f")
    df = g.generate_data(population, save_data=save)
    # print("Data generation done")
    return df


def generate_ff_halfsum_data(population, dimensions=6, gt_ate=1, save=True):
    feature_function = lambda xs: np.sum(xs[: int(len(xs) / 2 + 1)])
    main_effect = lambda xs: feature_function(xs)
    treatment_effect = lambda xs: feature_function(xs) + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * (t - 0.5) + n
    f_distribution = [lambda: 0.5 * np.random.normal(1, 1)]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"basic_{dimensions}f")
    df = g.generate_data(population, save_data=save)
    # print("Data generation done")
    return df


def generate_ff_prod_data(population, dimensions=6, gt_ate=1, save=True):
    feature_function = lambda xs: np.prod(xs)
    main_effect = lambda xs: feature_function(xs)
    treatment_effect = lambda xs: feature_function(xs) + gt_ate
    treatment_propensity = lambda xs: sigmoid(feature_function(xs) + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * (t - 0.5) + n
    f_distribution = [lambda: 0.5 * np.random.normal(1, 1)]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"basic_{dimensions}f")
    df = g.generate_data(population, save_data=save)
    # print("Data generation done")
    return df


def generate_6evh_data(population, dimensions=6, gt_ate=1, save=False):
    main_effect = lambda xs: xs[0] + xs[1] + xs[2] + xs[3]
    treatment_effect = lambda xs: xs[0] + xs[1] + xs[2] + xs[4] + gt_ate
    treatment_propensity = lambda xs: sigmoid(xs[0] + xs[1] + xs[2] + xs[5] + np.random.normal())
    noise = lambda: np.random.normal()
    treatment_function = lambda p, n: np.random.binomial(1, p)
    outcome_function = lambda me, t, te, n: me + te * (t - 0.5) + n
    f_distribution = [lambda: 0.5 * np.random.normal(1, 1)]
    g = Generator(main_effect, treatment_effect, treatment_propensity, noise, lambda xs: 0, treatment_function,
                  outcome_function, dimensions, f_distribution, name=f"basic_{dimensions}f")
    df = g.generate_data(population, save_data=save)
    # print("Data generation done")
    return df


def experiment_number_of_hidden_variables_ATE(df, dimensions=6, lr=False):
    psm = PropensityScoreMatching()
    number_hidden_variables = 0
    gt_ate = getATE(df)
    results = {}
    for fdl in groupedFeaturePowerset(dimensions):
        ires = []
        for fd in tqdm(fdl):
            with HiddenPrints():
                psm_ate = psm.estimate_ATE(df, "treatment", "outcome", fd) if \
                    (not lr) else estimate_ATE_Linear_Regression(df, fd)
            ires.append(np.square(psm_ate - gt_ate))
        results[number_hidden_variables] = np.sqrt(np.mean(ires))
        number_hidden_variables += 1

    return results


def experiment_effect_of_hidden_variables_ATE(df, dimensions=6, lr=False):
    psm = PropensityScoreMatching()
    results = {}
    all_f = getFeatureDict(dimensions)
    for i in range(dimensions):
        fd = all_f.copy()
        fd.pop(f'feature_{i}')
        with HiddenPrints():
            psm_ate = psm.estimate_ATE(df, "treatment", "outcome", fd) if \
                (not lr) else estimate_ATE_Linear_Regression(df, fd)
        results[f'feature_{i}'] = np.abs(psm_ate - getATE(df))

    return results


def generate_common_missing_variables(i=100, ex_features=None, f_dimensions=6):
    if ex_features is None:
        ex_features = []
    psm = PropensityScoreMatching()
    fd = getFeatureDict(f_dimensions)
    for exf in ex_features:
        fd.pop(exf)
    res = []
    for x in tqdm(range(i)):
        with HiddenPrints():
            df = generate_3cv_3ncf_data(2500, save=False)
            psm_ate = psm.estimate_ATE(df, "treatment", "outcome", fd)
        res.append(np.abs(psm_ate - getATE(df)))
    return res


def estimate_ATE_Linear_Regression(df, fd):
    if len(fd) == 0:
        return 0
    features = fd.keys()
    t0_data = df[df['treatment'] == 0.0]
    t1_data = df[df['treatment'] == 1.0]

    t0_X = t0_data[features].to_numpy().tolist()
    t0_y = t0_data['outcome'].tolist()

    t1_X = t1_data[features].to_numpy().tolist()
    t1_y = t1_data['outcome'].tolist()

    regY0 = linear_model.LinearRegression()
    regY0.fit(t0_X, t0_y)

    regY1 = linear_model.LinearRegression()
    regY1.fit(t1_X, t1_y)

    res = (regY1.predict(df[features].to_numpy()) -
           regY0.predict(df[features].to_numpy())).mean()
    return res


def generate_common_missing_variables_linear_regression(i=100, ex_features=None, f_dimensions=6):
    if ex_features is None:
        ex_features = []
    fd = getFeatureDict(f_dimensions)
    for exf in ex_features:
        fd.pop(exf)
    res = []
    for x in tqdm(range(i)):
        with HiddenPrints():
            df = generate_3cv_3ncf_data(2500, save=False)
            lr_ate = estimate_ATE_Linear_Regression(df, fd)
        res.append(np.abs(lr_ate - getATE(df)))
    return res


def experiment_common_missing_variables_ATE(i=100):
    b = generate_common_missing_variables(i)
    f0 = generate_common_missing_variables(i, ex_features=["feature_0"])
    f1 = generate_common_missing_variables(i, ex_features=["feature_1"])
    f2 = generate_common_missing_variables(i, ex_features=["feature_2"])
    f3 = generate_common_missing_variables(i, ex_features=["feature_3"])
    f4 = generate_common_missing_variables(i, ex_features=["feature_4"])
    f5 = generate_common_missing_variables(i, ex_features=["feature_5"])
    data = pd.DataFrame(
        {"Baseline": b,
         "Feature 0 is hidden": f0,
         "Feature 1 is hidden": f1,
         "Feature 2 is hidden": f2,
         "Feature 3 is hidden": f3,
         "Feature 4 is hidden": f4,
         "Feature 5 is hidden": f5})
    data.to_csv("data/common_missing_variables_test")
    data.boxplot(vert=False)
    plt.xlabel(f"Absolute Error ({i} iterations)")
    plt.title("Error variance compared to true ATE | PSM")
    plt.subplots_adjust(left=0.35)
    plt.show()


def experiment_common_missing_variables_ATE_LR(i=100):
    b = generate_common_missing_variables_linear_regression(i)
    f0 = generate_common_missing_variables_linear_regression(i, ex_features=["feature_0"])
    f1 = generate_common_missing_variables_linear_regression(i, ex_features=["feature_1"])
    f2 = generate_common_missing_variables_linear_regression(i, ex_features=["feature_2"])
    f3 = generate_common_missing_variables_linear_regression(i, ex_features=["feature_3"])
    f4 = generate_common_missing_variables_linear_regression(i, ex_features=["feature_4"])
    f5 = generate_common_missing_variables_linear_regression(i, ex_features=["feature_5"])
    data = pd.DataFrame(
        {"Baseline": b,
         "Feature 0 is hidden": f0,
         "Feature 1 is hidden": f1,
         "Feature 2 is hidden": f2,
         "Feature 3 is hidden": f3,
         "Feature 4 is hidden": f4,
         "Feature 5 is hidden": f5})
    data.to_csv("data/common_missing_variables_test")
    data.boxplot(vert=False)
    plt.xlabel(f"Absolute Error ({i} iterations)")
    plt.title("Error variance compared to true ATE | Linear Regression")
    plt.subplots_adjust(left=0.35)
    plt.show()


def evh_final(iterations=100, lr=False):
    f_dimensions = 6
    pop = 2500

    labels = ['f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5']

    res = []
    for x in tqdm(range(iterations)):
        res.append(list(experiment_effect_of_hidden_variables_ATE(
            generate_6evh_data(population=pop, dimensions=f_dimensions, save=False),
            dimensions=f_dimensions, lr=lr).values()))

    res = np.array(res).mean(axis=0)

    plt.bar(labels, res)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel(f'Feature Hidden ({iterations} iterations)')
    plt.ylabel('MAE')
    plt.title('Error when each variable is hidden separately (ATE) | ' + ('PSM' if not lr else "LR"))

    plt.show()


def enhv_final(dim=6, lr=False):
    pop = 2500
    res_sum = experiment_number_of_hidden_variables_ATE(generate_basic_data(pop, dimensions=dim, save=False),
                                                        dimensions=dim, lr=lr)
    res_halfsum = experiment_number_of_hidden_variables_ATE(generate_ff_halfsum_data(pop, dimensions=dim, save=False),
                                                            dimensions=dim, lr=lr)
    res_prod = experiment_number_of_hidden_variables_ATE(generate_ff_prod_data(pop, dimensions=dim, save=False),
                                                         dimensions=dim, lr=lr)

    plt.plot(list(res_sum.keys())[:-1], list(res_sum.values())[:-1], label="A")
    plt.plot(list(res_halfsum.keys())[:-1], list(res_halfsum.values())[:-1], label="B")
    plt.plot(list(res_prod.keys())[:-1], list(res_prod.values())[:-1], label="C")
    plt.title('Error proportional to hidden variables (ATE) | ' + ('PSM' if not lr else "LR"))
    plt.xlabel("Number of hidden variables")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # data = pd.read_csv("data/data_dump_common_8f/generated_dataSun_May_29_19-55-12_2022.csv")
    enhv_final(6)
