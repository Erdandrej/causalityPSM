from enum import Enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from causality.estimation.parametric import PropensityScoreMatching


class GenType(Enum):
    ALL = 0
    ONLY_TREATMENT = 1
    ONLY_OUTCOME = 2
    ONLY_PROPENSITY_SCORE = 3


class DataGenerator:

    def generate_data_all(self, n):
        x1 = 0.5 * np.random.normal(size=n)
        x2 = 0.5 * np.random.normal(size=n)
        x3 = 0.5 * np.random.normal(size=n)

        xs = x1 + x2 + x3

        arg = (xs + np.random.normal(size=n))
        p = np.exp(arg) / (1. + np.exp(arg))
        z = np.random.binomial(1, p)

        y = (np.random.normal(size=n) + (xs + 1.) * z) + xs

        X = pd.DataFrame({'z': z, 'x1': x1, 'x2': x2, 'x3': x3, 'y': y, 'p': p})
        return X

    def generate_data(self, n=1000, gen_type=GenType.ALL):
        if gen_type == GenType.ALL:
            return self.generate_data_all(n)
        if gen_type == GenType.ONLY_TREATMENT:
            return self.generate_data_all(n)
        if gen_type == GenType.ONLY_OUTCOME:
            return self.generate_data_all(n)
        if gen_type == GenType.ONLY_PROPENSITY_SCORE:
            return self.generate_data_all(n)


class GraphGenerator:

    def generateBarPlot(self, data):
        courses = list(data.keys())
        values = list(data.values())

        # creating the bar plot
        plt.plot(courses, values)

        plt.title("Error proportional to hidden variables")
        plt.xlabel("Number of hidden variables")
        plt.ylabel("Error")
        plt.show()


if __name__ == '__main__':
    dg = DataGenerator()
    psm = PropensityScoreMatching()
    gg = GraphGenerator()

    X = dg.generate_data(1000, GenType.ALL)
    ATT0 = psm.estimate_ATT(X, 'z', 'y', {'x1': 'c', 'x2': 'c', 'x3': 'c'})
    ATT1 = psm.estimate_ATT(X, 'z', 'y', {'x1': 'c', 'x2': 'c'})
    ATT2 = psm.estimate_ATT(X, 'z', 'y', {'x1': 'c'})
    ATT3 = psm.estimate_ATT(X, 'z', 'y', {})
    data = {'0': ATT0 - ATT0, '1': ATT1 - ATT0, '2': ATT2 - ATT0, '3': ATT3 - ATT0}
    gg.generateBarPlot(data)
