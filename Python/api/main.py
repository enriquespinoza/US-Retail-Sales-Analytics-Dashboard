
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import shutil
import numpy as np
from typing import Optional
from fastapi.encoders import jsonable_encoder
import math

app = FastAPI(title="State Retail Sales API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "Data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "yoy": {
        "url": "https://www.census.gov/retail/mrts/www/statedata/state_retail_yy.csv",
        "file": DATA_RAW / "state_retail_yy.csv",
    },
    "se": {
        "url": "https://www.census.gov/retail/mrts/www/statedata/state_retail_se.csv",
        "file": DATA_RAW / "standard_errors.csv",
    },
    "coverage": {
        "url": "https://www.census.gov/retail/mrts/www/statedata/state_retail_coverage.csv",
        "file": DATA_RAW / "state_retail_coverage.csv",
    },
}

# simple in-memory cache for cleaned frames
_CACHE = {}


def download_if_missing(name: str, force: bool = False):
    src = SOURCES[name]
    local_path = Path(src["file"])
    if force or not local_path.exists():
        resp = requests.get(src["url"], stream=True, timeout=30)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
    return local_path


def clean_yoy(local_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        local_path,
        dtype={
            "fips": str,
            "stateabbr": str,
            "naics": str
        }
    )

    df.columns = df.columns.str.strip().str.lower()

    if "fips" in df.columns:
        df["fips"] = df["fips"].str.strip().str.zfill(2)

    id_columns = [
        c for c in ["fips", "stateabbr", "naics"]
        if c in df.columns
    ]

    date_columns = [
        col for col in df.columns
        if col.startswith("yy")
    ]

    if not date_columns:
        raise ValueError("No yy* columns found in YOY file")

    df_long = df.melt(
        id_vars=id_columns,
        value_vars=date_columns,
        var_name="month_code",
        value_name="yoy_growth"
    )

    df_long["year"] = df_long["month_code"].str[2:6].astype(int)
    df_long["month_number"] = df_long["month_code"].str[6:8].astype(int)

    df_long["date"] = pd.to_datetime({
        "year": df_long["year"],
        "month": df_long["month_number"],
        "day": 1
    })

    df_long["yoy_growth"] = pd.to_numeric(
        df_long["yoy_growth"],
        errors="coerce"
    )

    if "stateabbr" in df_long.columns:
        df_long["stateabbr"] = (
            df_long["stateabbr"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    if "naics" in df_long.columns:
        df_long["naics"] = (
            df_long["naics"]
            .astype(str)
            .str.strip()
        )

    df_long = df_long.drop_duplicates()

    return df_long


def clean_se(local_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        local_path,
        dtype={
            "fips": str,
            "stateabbr": str,
            "naics": str
        }
    )

    df.columns = df.columns.str.strip().str.lower()

    if "fips" in df.columns:
        df["fips"] = df["fips"].str.strip().str.zfill(2)

    id_columns = [
        c for c in ["fips", "stateabbr", "naics"]
        if c in df.columns
    ]

    date_columns = [
        col for col in df.columns
        if col.startswith("se")
    ]

    if not date_columns:
        raise ValueError("No se* columns found in SE file")

    df_long = df.melt(
        id_vars=id_columns,
        value_vars=date_columns,
        var_name="month_code",
        value_name="standard_error"
    )

    df_long["year"] = df_long["month_code"].str[2:6].astype(int)
    df_long["month_number"] = df_long["month_code"].str[6:8].astype(int)

    df_long["date"] = pd.to_datetime({
        "year": df_long["year"],
        "month": df_long["month_number"],
        "day": 1
    })

    df_long["standard_error"] = pd.to_numeric(
        df_long["standard_error"],
        errors="coerce"
    )

    if "stateabbr" in df_long.columns:
        df_long["stateabbr"] = (
            df_long["stateabbr"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    if "naics" in df_long.columns:
        df_long["naics"] = (
            df_long["naics"]
            .astype(str)
            .str.strip()
        )

    df_long = df_long.drop_duplicates()

    return df_long


def clean_coverage(local_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        local_path,
        dtype={
            "fips": str,
            "stateabbr": str,
            "naics": str
        }
    )

    df.columns = df.columns.str.strip().str.lower()

    if "fips" in df.columns:
        df["fips"] = df["fips"].str.strip().str.zfill(2)

    id_columns = [
        c for c in ["fips", "stateabbr", "naics"]
        if c in df.columns
    ]

    date_columns = [
        col for col in df.columns
        if col.startswith("cov")
    ]

    if not date_columns:
        raise ValueError("No cov* columns found in coverage file")

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

    if "stateabbr" in df_long.columns:
        df_long["stateabbr"] = (
            df_long["stateabbr"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    if "naics" in df_long.columns:
        df_long["naics"] = (
            df_long["naics"]
            .astype(str)
            .str.strip()
        )

    df_long = df_long.drop_duplicates()

    return df_long


def _load_clean(name: str, force: bool = False) -> pd.DataFrame:

    if not force and name in _CACHE:
        return _CACHE[name]

    local = download_if_missing(name, force=force)

    if name == "yoy":
        df = clean_yoy(local)
    elif name == "se":
        df = clean_se(local)
    elif name == "coverage":
        df = clean_coverage(local)
    else:
        raise ValueError(name)

    _CACHE[name] = df

    return df


def _apply_filters(
    df: pd.DataFrame,
    state: Optional[str],
    year: Optional[int],
    naics: Optional[str]
) -> pd.DataFrame:

    out = df

    if state and "stateabbr" in out.columns:
        out = out[
            out["stateabbr"].astype(str).str.upper() == state.upper()
        ]

    if year and "year" in out.columns:
        out = out[out["year"] == int(year)]

    if naics and "naics" in out.columns:
        out = out[
            out["naics"].astype(str) == str(naics)
        ]

    return out


def dataframe_to_safe_records(df: pd.DataFrame):
    safe_df = df.copy()

    safe_df = safe_df.replace([np.inf, -np.inf], np.nan)

    if "date" in safe_df.columns:
        safe_df["date"] = safe_df["date"].dt.strftime("%Y-%m-%d")

    safe_df = safe_df.astype(object)
    safe_df = safe_df.where(pd.notnull(safe_df), None)

    return safe_df.to_dict(orient="records")


@app.get("/yoy")
def get_yoy(
    state: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    naics: Optional[str] = Query(None),
    force: bool = Query(False)
):
    try:
        df = _load_clean("yoy", force=force)
        out = _apply_filters(df, state, year, naics)
        records = dataframe_to_safe_records(out)

        return {
            "count": len(records),
            "data": records
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"yoy endpoint error: {e}"
        )


@app.get("/standard_errors")
def get_se(
    state: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    naics: Optional[str] = Query(None),
    force: bool = Query(False)
):
    try:
        df = _load_clean("se", force=force)
        out = _apply_filters(df, state, year, naics)
        records = dataframe_to_safe_records(out)

        return {
            "count": len(records),
            "data": records
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"se endpoint error: {e}"
        )


@app.get("/coverage")
def get_coverage(
    state: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    naics: Optional[str] = Query(None),
    force: bool = Query(False)
):
    try:
        df = _load_clean("coverage", force=force)
        out = _apply_filters(df, state, year, naics)
        records = dataframe_to_safe_records(out)

        return {
            "count": len(records),
            "data": records
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"coverage endpoint error: {e}"
        )
    
def check_duplicate_keys(
    df: pd.DataFrame,
    keys: list[str],
    name: str
):
    duplicate_count = df.duplicated(subset=keys).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{name} contains {duplicate_count} duplicate rows "
            f"for join keys {keys}"
        )


@app.get("/combined")
def get_combined(
    state: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    naics: Optional[str] = Query(None),
    force: bool = Query(False)
):
    try:
        df_yoy = _load_clean("yoy", force=force)
        df_se = _load_clean("se", force=force)
        df_cov = _load_clean("coverage", force=force)

        df_yoy = _apply_filters(df_yoy, state, year, naics)
        df_se = _apply_filters(df_se, state, year, naics)
        df_cov = _apply_filters(df_cov, state, year, naics)

        join_keys = [
            "fips",
            "stateabbr",
            "naics",
            "year",
            "month_number",
            "date",
        ]

        join_keys = [
            key
            for key in join_keys
            if key in df_yoy.columns
            and key in df_se.columns
            and key in df_cov.columns
        ]

        if not join_keys:
            raise ValueError("No valid common join keys found")
        check_duplicate_keys(df_yoy, join_keys, "YOY")
        check_duplicate_keys(df_se, join_keys, "Standard Errors")
        check_duplicate_keys(df_cov, join_keys, "Coverage")
        merged = (
            df_yoy
            .merge(
                df_se,
                on=join_keys,
                how="left"
            )
            .merge(
                df_cov,
                on=join_keys,
                how="left"
            )
        )

        merged = merged.drop(
            columns=["month_code_x", "month_code_y", "month_code"],
            errors="ignore"
        )

        records = dataframe_to_safe_records(merged)

        return {
            "count": len(records),
            "join_keys": join_keys,
            "data": records
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"combined endpoint error: {e}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
