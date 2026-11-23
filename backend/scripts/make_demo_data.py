"""Generate a churn dataset with deliberately planted problems.

Each problem below is something a later phase should detect (and, where possible,
fix). That gives a reproducible demo and a ground truth to test the doctor against.

  Planted issue                      Column(s)                     Detected in
  ---------------------------------  ----------------------------  -----------
  Class imbalance (~10% churn)     churned                       Phase 2
  Missing values (MCAR)              age, total_charges, payment   Phase 2
  Duplicate rows (~1%)               all                           Phase 2
  Outliers (x10 values)              monthly_charges               Phase 2
  Constant column                    country                       Phase 2
  ID column                          customer_id                   Phase 1/2
  Highly correlated pair             tenure_months/total_charges   Phase 2
  Target leakage                     cancellation_request          Phase 2/3
  Group with worse outcomes          region == "West"              Phase 4

Usage:  python scripts/make_demo_data.py [--rows 5000] [--seed 42] [--out PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "sample_data" / "churn_demo.csv"


def make_churn(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = rows

    tenure = rng.integers(1, 73, n)
    monthly = rng.normal(70, 25, n).clip(18, 150).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    support_calls = rng.poisson(1.5, n)
    age = rng.integers(18, 80, n).astype(float)
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.30, 0.30, 0.25, 0.15])
    payment = rng.choice(["Card", "Bank transfer", "UPI", "Cash"], n).astype(object)
    has_partner = rng.choice(["Yes", "No"], n)

    logit = (
        -2.9
        + 1.4 * (contract == "Month-to-month")
        - 0.035 * tenure
        + 0.35 * support_calls
        + 0.012 * (monthly - 70)
        + 0.7 * (region == "West")
    )
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))

    total = (tenure * monthly * rng.normal(1, 0.03, n)).round(2)
    # Leak: almost always set when the customer churned, almost never otherwise.
    cancellation_request = np.where(churned, rng.random(n) < 0.93, rng.random(n) < 0.02).astype(int)

    signup = pd.Timestamp("2026-11-01") - pd.to_timedelta(tenure * 30, unit="D")

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(n)],
            "age": age,
            "region": region,
            "country": "IN",
            "signup_date": signup.strftime("%Y-%m-%d"),
            "contract": contract,
            "payment_method": payment,
            "has_partner": has_partner,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_support_calls": support_calls,
            "cancellation_request": cancellation_request,
            "churned": churned.astype(int),
        }
    )

    # Missing values
    df.loc[rng.random(n) < 0.06, "age"] = np.nan
    df.loc[rng.random(n) < 0.04, "total_charges"] = np.nan
    df.loc[rng.random(n) < 0.02, "payment_method"] = np.nan

    # Outliers
    out_idx = rng.choice(n, size=max(1, n // 200), replace=False)
    df.loc[out_idx, "monthly_charges"] = (df.loc[out_idx, "monthly_charges"] * 10).round(2)

    # Duplicates, then shuffle
    dupes = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, dupes]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def make_house_prices(rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Regression demo: house prices with planted problems.

      Planted issue                        Column(s)              Detected in
      -----------------------------------  ---------------------  -----------
      Skewed target (log-normal prices)    price                  Phase 1/2 (log-transform tested in Phase 5)
      Target leakage                       assessed_value         Phase 2/3
      Missing values                       year_built, garage     Phase 2
      Duplicate rows (~1%)                 all                    Phase 2
      Constant column / ID column          currency / listing_id  Phase 2
      Irrelevant feature                   listing_weekday        Phase 2
      Segment with larger errors           neighbourhood = Hills  Phase 4
    """
    rng = np.random.default_rng(seed)
    n = rows
    sqft = rng.gamma(9, 180, n).clip(350, 9000).round()
    bedrooms = np.clip((sqft / 600 + rng.normal(0, 0.8, n)).round(), 1, 8).astype(int)
    bathrooms = np.clip((bedrooms * 0.7 + rng.normal(0, 0.5, n)).round(1), 1, 6)
    year_built = rng.integers(1920, 2025, n).astype(float)
    neighbourhood = rng.choice(["Centre", "Riverside", "Suburbs", "Hills"], n, p=[0.25, 0.25, 0.35, 0.15])
    garage = rng.choice(["none", "single", "double"], n, p=[0.3, 0.45, 0.25]).astype(object)
    condition = rng.integers(1, 6, n)
    loc_mult = pd.Series(neighbourhood).map({"Centre": 1.6, "Riverside": 1.3, "Suburbs": 1.0, "Hills": 1.2}).to_numpy()
    # The Hills are unpredictable: extra noise there.
    noise = np.where(neighbourhood == "Hills", rng.normal(0, 0.35, n), rng.normal(0, 0.12, n))
    log_price = (np.log(sqft) * 0.95 + 0.06 * bedrooms + 0.05 * condition + 0.004 * (year_built - 1970)
                 + np.log(loc_mult) + 0.1 * (garage == "double") + 6.2 + noise)
    price = np.exp(log_price).round(-2)
    df = pd.DataFrame({
        "listing_id": [f"L{i:05d}" for i in range(n)],
        "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms, "year_built": year_built,
        "neighbourhood": neighbourhood, "garage": garage, "condition": condition,
        "listing_weekday": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n),
        "currency": "USD",
        # Leak: a tax assessment recorded *after* the sale, so it is almost the price itself.
        "assessed_value": (price * rng.normal(0.97, 0.02, n)).round(-2),
        "price": price,
    })
    df.loc[rng.random(n) < 0.05, "year_built"] = np.nan
    df.loc[rng.random(n) < 0.03, "garage"] = np.nan
    dupes = df.sample(frac=0.01, random_state=seed)
    return pd.concat([df, dupes]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--kind", choices=["churn", "house_prices"], default="churn")
    args = p.parse_args()

    df = make_churn(args.rows, args.seed) if args.kind == "churn" else make_house_prices(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols to {args.out}")
    if "churned" in df:
        print(f"Churn rate: {df['churned'].mean():.1%}")
    else:
        print(f"Price: median {df['price'].median():,.0f}, skew {df['price'].skew():.2f}")


if __name__ == "__main__":
    main()
