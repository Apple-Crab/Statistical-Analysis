import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm

population_size = 23000
underweight_pct = 0.023
normal_pct = 0.469
overweight_pct = 0.164
obese_pct = 1 - (underweight_pct + normal_pct + overweight_pct)

underweight_count = int(population_size * underweight_pct)
normal_count = int(population_size * normal_pct)
overweight_count = int(population_size * overweight_pct)
obese_count = population_size - (underweight_count + normal_count + overweight_count)

np.random.seed(42)

underweight = np.random.uniform(15, 18.5, underweight_count)
normal = np.random.uniform(18.5, 24.9, normal_count)
overweight = np.random.uniform(25, 29.9, overweight_count)
obese = np.random.uniform(30, 40, obese_count)

bmi_data = np.concatenate([underweight, normal, overweight, obese])

sample_mean = np.mean(bmi_data)
sample_std = np.std(bmi_data, ddof=1)
quartiles = np.percentile(bmi_data, [25, 50, 75])

print("Sample Mean:", sample_mean)
print("Sample Standard Deviation:", sample_std)
print("Quartiles (Q1, Median, Q3):")
print(quartiles)

plt.hist(bmi_data, bins=20, color='skyblue', edgecolor='black', alpha=0.7, density=True)
plt.title("Histogram of Simulated BMI Data", fontsize=14)
plt.xlabel("BMI", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(axis='y', alpha=0.75)
plt.show()

bins = np.histogram_bin_edges(bmi_data, bins=10)
observed, _ = np.histogram(bmi_data, bins)

mu = sample_mean
sigma = sample_std

expected = [
    len(bmi_data) * (norm.cdf(bins[i + 1], mu, sigma) - norm.cdf(bins[i], mu, sigma))
    for i in range(len(bins) - 1)
]

chi_square_stat = sum((obs - exp) ** 2 / exp for obs, exp in zip(observed, expected) if exp > 0)
df = len(bins) - 1 - 2
critical_value = chi2.ppf(0.95, df)
p_value = 1 - chi2.cdf(chi_square_stat, df)

print("\nChi-Square Test for Normal Distribution:")
print("Chi-Square Statistic:", chi_square_stat)
print("Critical Value:", critical_value)
print("Reject Null Hypothesis:" if chi_square_stat > critical_value else "Fail to Reject Null Hypothesis")
print("P value:", p_value)