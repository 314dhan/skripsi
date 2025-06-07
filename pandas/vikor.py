import pandas as pd
import numpy as np

# Data alternatif dan kriteria
data = [
    ["Arjuna Reyhan Wibawa", 20, 15, 1, 2, 3, 4],
    ["Muhammad Dafa Alfaris", 15, 15, 1, 2, 3, 4],
    ["Hikmah Dhika Nur Syafaatulloh", 20, 15, -1, 2, 3, 4],
    ["Saiful Hidayat", 10, 10, 1, 2, 3, 4],
    ["Rizki Andika Pratama", 15, 15, 1, 2, 3, 4],
    ["Iqbal Al Rasyiig", 20, 20, 1, 2, 3, 4],
    ["Muhamad Maulana Sawaludin", 10, 10, 1, 2, 3, 4],
    ["Villaxandria Violensiera Virtous Vierosa Virginia", 20, 15, 1, 2, 3, 4],
    ["Crespo Marfandi Pratama", 20, 15, 1, 2, 3, 4],
    ["Muhammad A'Azif Junaidi", 20, 10, 1, 2, 3, 4],
    ["Julian Fahmi", 20, 15, 1, 2, 3, 4],
    ["Firda Aulia", 20, 10, 1, 2, 3, 4],
    ["Tantri Pramudita", 20, 15, 1, 2, 3, 4],
    ["Firdaus Abdul Ghanny", 20, 10, 1, 2, 3, 4],
    ["Sapta", 10, 10, 1, 2, 3, 4],
    ["Jamaludin", 20, 20, 1, 2, 3, 4],
    ["Indra Priatna", 20, 15, 1, 2, 3, 4],
    ["Elviana", 15, 15, 1, 2, 3, 4]
]

columns = ["Nama", "a1", "a2", "a3", "a4", "a5", "a6"]
df = pd.dataframe(data, columns=columns)

# Bobot kriteria
weights = np.array([0.30, 0.30, 0.10, 0.10, 0.10, 0.10])

# Tipe kriteria: cost = -1, benefit = 1
criteria_types = np.array([-1, 1, -1, -1, -1, -1])

# Menentukan f_star (terbaik) dan f_minus (terburuk) berdasarkan tipe kriteria
f_star = df.iloc[:, 1:].max().where(criteria_types == 1, df.iloc[:, 1:].min())
f_minus = df.iloc[:, 1:].min().where(criteria_types == 1, df.iloc[:, 1:].max())

# Normalisasi menggunakan metode VIKOR
normalized = np.abs((f_star - df.iloc[:, 1:]) / (f_star - f_minus))

# Hitung S_i dan R_i
S = normalized.mul(weights).sum(axis=1)
R = normalized.mul(weights).max(axis=1)

# Hitung Q_i
v = 0.5
S_star, S_minus = S.min(), S.max()
R_star, R_minus = R.min(), R.max()

Q = v * (S - S_star) / (S_minus - S_star) + (1 - v) * (R - R_star) / (R_minus - R_star)

# Gabungkan hasil
df_result = df.copy()
df_result["S"] = S
df_result["R"] = R
df_result["Q"] = Q
df_result["Rank"] = Q.rank(method="min")

# Tampilkan hasil terurut
df_result_sorted = df_result.sort_values("Q")
df_result_sorted.reset_index(drop=True, inplace=True)
df_result_sorted[["Nama", "S", "R", "Q", "Rank"]]
