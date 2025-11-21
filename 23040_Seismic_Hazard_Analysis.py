import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 1. CREATE BASIC SYNTHETIC DATASET
# =====================================================
N = 1000

df = pd.DataFrame({
    'M': np.random.uniform(5, 7.5, N),
    'R': np.random.uniform(1, 200, N),
    'SD': np.random.uniform(0, 500, N),
    'Q0': np.random.uniform(0, 5000, N),
    'kappa0': np.random.uniform(0, 0.1, N),
    'VS30': np.random.uniform(600, 2800, N)
})

df['PGA'] = np.exp(
    -4 + 1.0 * df['M'] - 1.1*np.log(df['R']) + 0.3*np.log(df['SD']+1)
)

print("Nodes:", ['M','R','SD','Q0','kappa0','VS30','PGA'])

# =====================================================
# 2. DISCRETISATION
# =====================================================
def make_bins(x, bins):
    return pd.cut(x, bins=bins, labels=False, include_lowest=True)

df['M_d'] = make_bins(df['M'], 3)
df['PGA_d'] = make_bins(df['PGA'], 8)

print("\nSample discretised values:")
print(df[['M','M_d','PGA','PGA_d']].head())

# =====================================================
# 3. SIMPLE MTE-LIKE MODEL (exponential approx per M bin)
# =====================================================
def fit_simple_exp(x):
    x = np.array(x)
    if len(x) < 5:
        return None
    a = x.min()
    lam = 1.0 / (np.mean(x) - a + 1e-9)
    return {"lambda": lam, "a": a}

mte_models = {}
for m_bin in df['M_d'].unique():
    data_bin = df[df['M_d'] == m_bin]['PGA']
    mte_models[int(m_bin)] = fit_simple_exp(data_bin)

print("\nFitted MTE models:")
print(mte_models)

# =====================================================
# 4. PREDICT PGA USING MTE
# =====================================================
def predict_from_mte(m_bin):
    model = mte_models[m_bin]
    return model['a'] + 1/model['lambda']

df['PGA_pred'] = df['M_d'].apply(predict_from_mte)

print("\nPredictions:")
print(df[['M','PGA','PGA_pred']].head())

# =====================================================
# 5. VISUALIZATIONS
# =====================================================

# ----------------------------------------------
# Plot 1: Actual vs Predicted PGA Scatter Plot
# ----------------------------------------------
plt.figure(figsize=(7,5))
sns.scatterplot(x=df['PGA'], y=df['PGA_pred'], alpha=0.5)
plt.xlabel("Actual PGA")
plt.ylabel("Predicted PGA (MTE)")
plt.title("Actual vs Predicted PGA")
plt.grid(True)
plt.show()

# ----------------------------------------------
# Plot 2: Histogram of PGA by Magnitude Bin
# ----------------------------------------------
plt.figure(figsize=(8,5))
for m_bin in range(3):
    sns.kdeplot(df[df['M_d'] == m_bin]['PGA'], label=f"M_d = {m_bin}", fill=True)
plt.title("PGA Distribution for Each Magnitude Bin")
plt.xlabel("PGA")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------
# Plot 3: MTE Curve Example for a Single Bin
# ----------------------------------------------
example_bin = 1
bin_data = df[df['M_d'] == example_bin]['PGA']

a = mte_models[example_bin]['a']
lam = mte_models[example_bin]['lambda']

x_vals = np.linspace(bin_data.min(), bin_data.max(), 200)
mte_curve = lam * np.exp(-lam * (x_vals - a))

plt.figure(figsize=(8,5))
sns.histplot(bin_data, bins=15, kde=False, stat="density", label="Bin Data", color='skyblue')
plt.plot(x_vals, mte_curve, color='red', linewidth=2, label="MTE Curve")
plt.title(f"MTE Fit for Magnitude Bin: {example_bin}")
plt.xlabel("PGA")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
