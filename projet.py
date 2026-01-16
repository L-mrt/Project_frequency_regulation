import pandas as pd
import matplotlib.pyplot as plt

# Q1
# Import data
df = pd.read_csv("data/france_2019_05.csv")
print(df.head())

# Construction of values Pre/Pbid
df["Frequency"] = df["Frequency"]*10**(-3)
df["Pre/Pbid"] = 5*(df["Frequency"])
print(df[["Frequency", "Pre/Pbid"]].head())

# Plot of the distribution Pre/Pbid
df["Pre/Pbid"].hist(bins=200)
plt.xlabel("Valeur")
plt.ylabel("Fréquence")
plt.title("Distribution of Pre/Pbid")
plt.show()

df["Pre/Pbid"].describe()

# Q2 :The regulating power tries to balance the frequency between -0.2 Hz and 0.2Hz

# Q3
E_batt = 46 # 𝐸𝑏𝑎𝑡𝑡 the useful energy capacity of the battery in kWh
P_max = 7 # Power of bindirectional transfert of the battery in kW
P_bid = P_max/1.1 #
eta = 1

df["Pre"] = df["Pre/Pbid"] * P_bid