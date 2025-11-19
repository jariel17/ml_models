import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def missing_data_distribution(dataframe):
    sample = dataframe.sample(n=min(len(dataframe), 1000), random_state=0)
    mask = sample.isnull().T  # transpose to have columns on y-axis
    plt.figure(figsize=(14, max(6, 0.15*mask.shape[0])))
    plt.imshow(mask, aspect='auto', interpolation='nearest')
    plt.yticks(ticks=np.arange(len(mask.index)), labels=mask.index)
    plt.xlabel('sampled rows')
    plt.title('Missingness mask (columns on y-axis)')
    plt.tight_layout()
    plt.show()