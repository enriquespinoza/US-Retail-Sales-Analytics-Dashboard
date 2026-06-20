from pathlib import Path
import pandas as pd

# -----------------------------
# Define project folders
# -----------------------------

project_root = Path(__file__).resolve().parents[1]

raw_folder = project_root / "Data" / "RAW"
cleaned_folder = project_root / "Data" / "cleaned"

# Create cleaned folder if it doesn't exist
cleaned_folder.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("PROJECT PATHS")
print("=" * 50)
print("Project Root:", project_root)
print("Raw Folder:", raw_folder)
print("Cleaned Folder:", cleaned_folder)

# -----------------------------
# Locate CSV file
# -----------------------------

csv_files = list(raw_folder.glob("*.csv"))

print("\nCSV Files Found:")
print(csv_files)

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No CSV files found in {raw_folder}"
    )

# Load first CSV found
df = pd.read_csv(csv_files[0])

# -----------------------------
# Basic dataset info
# -----------------------------

print("\n" + "=" * 50)
print("ORIGINAL DATASET")
print("=" * 50)

print("Shape:", df.shape)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# -----------------------------
# Convert wide format to long
# -----------------------------

id_columns = ["fips", "stateabbr", "naics"]

date_columns = [
    col for col in df.columns
    if col.startswith("yy")
]

df_long = df.melt(
    id_vars=id_columns,
    value_vars=date_columns,
    var_name="month_code",
    value_name="yoy_growth"
)

# -----------------------------
# Create date fields
# -----------------------------

df_long["year"] = (
    df_long["month_code"]
    .str[2:6]
    .astype(int)
)

df_long["month_number"] = (
    df_long["month_code"]
    .str[6:8]
    .astype(int)
)

df_long["date"] = pd.to_datetime(
    {
        "year": df_long["year"],
        "month": df_long["month_number"],
        "day": 1
    }
)

# -----------------------------
# Data cleaning
# -----------------------------

df_long["yoy_growth"] = pd.to_numeric(
    df_long["yoy_growth"],
    errors="coerce"
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

# Remove duplicates
df_long = df_long.drop_duplicates()

# -----------------------------
# Save cleaned file
# -----------------------------

output_path = (
    cleaned_folder /
    "state_retail_yoy_clean.csv"
)

df_long.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 50)
print("CLEANED DATASET")
print("=" * 50)

print("Rows:", len(df_long))
print("Columns:", len(df_long.columns))

print("\nOutput File:")
print(output_path)

print("\nFile Created:")
print(output_path.exists())

print("\nPreview:")
print(df_long.head())