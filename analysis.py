import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, expon

data = pd.read_csv("queue_data.csv")
wait_time = data['wait_time']

sample_mean = wait_time.mean()
sample_std = wait_time.std()
quartiles = wait_time.quantile([0.25, 0.5, 0.75])

print("Sample mean: ", sample_mean)
print("Sample Stabdard Deviation: ", sample_std)
print("Quartiles (Q1, Median, Q3):")
print(quartiles)

plt.hist(wait_time, bins = 10, color = 'skyblue', edgecolor = 'black', alpha = 0.7)
plt.title("Histogram of Wait Times", fontsize = 14)
plt.xlabel("Wait Time (Seconds)", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.grid(axis = 'y', alpha = 0.75)
plt.show()

bins = np.histogram_bin_edges(wait_time, bins = 10)
observed, _ = np.histogram(wait_time, bins)

mu = sample_mean
sigma = sample_std

expected_normal = [
    len(wait_time) * (norm.cdf(bins[i + 1], mu, sigma) - norm.cdf(bins[i], mu, sigma))
    for i in range(len(bins) - 1)
]

lambda_exp = 1 / mu

expected_exp = [
    len(wait_time) * (expon.cdf(bins[i + 1], scale = 1 / lambda_exp) - expon.cdf(bins[i], scale = 1 / lambda_exp))
    for i in range(len(bins) - 1)
]

chi_square_normal = sum((obs - exp) ** 2 / exp for obs, exp in zip(observed, expected_normal) if exp > 0)
df_normal = len(bins) - 1 - 2
critical_values_normal = chi2.ppf(0.95, df_normal)
p_value = 1 - chi2.cdf(chi_square_normal, df_normal)

print("\nNormal Distribution Test:")
print("Chi-Square Stats:", chi_square_normal)
print("Critical Value:", critical_values_normal)
print("Reject Normal Hypothesis:" if chi_square_normal > critical_values_normal else "Fail to reject null hypothesis")
print("P Value:", p_value)