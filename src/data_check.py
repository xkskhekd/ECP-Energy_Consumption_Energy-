import pandas as pd
import numpy as np

# ======================
# LOAD DATA
# ======================

df = pd.read_csv("dataset/PJME_hourly.csv")
df.columns = ["datetime", "energy"]

df["datetime"] = pd.to_datetime(df["datetime"])

print("Dataset loaded")
print("Rows:", len(df))


# ======================
# SORT CHECK
# ======================

is_sorted = df["datetime"].is_monotonic_increasing
print("\nIs datetime sorted:", is_sorted)

if not is_sorted:
    print("Sorting dataset...")
    df = df.sort_values("datetime")


# ======================
# DUPLICATE CHECK
# ======================

duplicates = df["datetime"].duplicated().sum()

print("\nDuplicate timestamps:", duplicates)

if duplicates > 0:
    dup_rows = df[df["datetime"].duplicated(keep=False)]
    print("\nExample duplicates:")
    print(dup_rows.head())


# ======================
# MISSING TIMESTAMP CHECK
# ======================

full_range = pd.date_range(
    start=df["datetime"].min(),
    end=df["datetime"].max(),
    freq="h"
)

missing = full_range.difference(df["datetime"])

print("\nMissing hourly timestamps:", len(missing))

if len(missing) > 0:
    print("Example missing timestamps:")
    print(missing[:10])


# ======================
# OUTLIER CHECK
# ======================

q1 = df["energy"].quantile(0.25)
q3 = df["energy"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df["energy"] < lower_bound) | (df["energy"] > upper_bound)]

print("\nOutliers detected:", len(outliers))

if len(outliers) > 0:
    print("Example outliers:")
    print(outliers.head())


# ======================
# BASIC STATS
# ======================

print("\nEnergy statistics")
print(df["energy"].describe())