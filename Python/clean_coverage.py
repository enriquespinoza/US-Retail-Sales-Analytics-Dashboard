from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]

raw_folder = project_root / "Data" / "RAW"
cleaned_folder = project_root / "Data" / "cleaned"

cleaned_folder.mkdir(parents=True, exist_ok=True)

csv_file = raw_folder / "coverage_measures.csv"

print("CSV File Found:")
print(csv_file)

if not csv_file.exists():
    raise FileNotFoundError(f"Could not find {csv_file}")

df = pd.read_csv(csv_file)

print("Original Shape:", df.shape)

df.columns = df.columns.str.strip().str.lower()

id_columns = ["fips", "stateabbr", "naics"]

date_columns = [
    col for col in df.columns
    if col.startswith("cov")
]

df_long = df.melt(
    id_vars=id_columns,
    value_vars=date_columns,
    var_name="month_code",
    value_name="coverage_measure"
)

df_long["year"] = df_long["month_code"].str[3:7].astype(int)
df_long["month_number"] = df_long["month_code"].str[7:9].astype(int)

df_long["date"] = pd.to_datetime({
    "year": df_long["year"],
    "month": df_long["month_number"],
    "day": 1
})

df_long["coverage_measure"] = (
    df_long["coverage_measure"]
    .astype(str)
    .str.strip()
)

# ---------------------------------
# Create numeric score for analysis
# ---------------------------------

coverage_map = {
    "A": 4,
    "B": 3,
    "C": 2,
    "D": 1
}

df_long["coverage_score"] = (
    df_long["coverage_measure"]
    .map(coverage_map)
)


df_long["stateabbr"] = (
    df_long["stateabbr"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df_long["naics"] = (
    df_long["naics"]
    .astype(str)
    .str.strip()
)

df_long = df_long.drop_duplicates()

output_path = cleaned_folder / "coverage_measures_clean.csv"

df_long.to_csv(output_path, index=False)

print("Cleaned Shape:", df_long.shape)
print("Output File:", output_path)
print("File Created:", output_path.exists())
print(df_long.head())