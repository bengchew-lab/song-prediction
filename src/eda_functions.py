import pandas as pd
import numpy as np
import os
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

def diagnostic_plots(df, variable):
    '''
    plot histogram and q-q plot
    '''

    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    df[variable].hist()

    plt.subplot(1,2,2)
    stats.probplot(df[variable].dropna(), plot = plt)
    plt.show()