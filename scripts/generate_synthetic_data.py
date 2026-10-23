"""Deterministic Synthetic Retail Dataset Generator.

Generates realistic transactional retail data modeled after the UCI Online Retail dataset,
incorporating intentional data quality anomalies (returns, missing IDs, duplicates, zero prices)
for validating preprocessing pipelines.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SYNTHETIC_DATA_DIR, SYNTHETIC_DATA_FILENAME
from config.logging_config import get_logger

logger = get_logger("scripts.generate_synthetic_data")

# Reference catalog of representative retail items
SAMPLE_PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053", "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.75),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART.", 3.39),
    ("22752", "SET 7 BABUSHKA NESTING BOXES", 8.50),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22632", "HAND WARMER RED POLKA DOT", 1.85),
    ("22633", "HAND WARMER UNION JACK", 1.85),
    ("22865", "HAND WARMER OWL DESIGN", 2.10),
    ("22866", "HAND WARMER SCOTTY DOG DESIGN", 2.10),
    ("22867", "HAND WARMER BIRD DESIGN", 2.10),
    ("22759", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
    ("22745", "POPPY'S PLAYHOUSE BEDROOM", 2.10),
    ("22748", "POPPY'S PLAYHOUSE KITCHEN", 2.10),
    ("22749", "POPPY'S PLAYHOUSE LIVING ROOM", 2.10),
    ("22310", "IVORY KNITTED MUG COSY", 1.65),
    ("84997D", "CHILDRENS CUTLERY POLKADOT PINK", 4.15),
    ("84997B", "CHILDRENS CUTLERY POLKADOT BLUE", 4.15),
    ("84997C", "CHILDRENS CUTLERY POLKADOT GREEN", 4.15),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("47566", "PARTY BUNTING", 4.65),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("20727", "LUNCH BAG BLACK SKULL", 1.65),
    ("20728", "LUNCH BAG CARS BLUE", 1.65),
    ("22383", "LUNCH BAG SUKI DESIGN", 1.65),
    ("22384", "LUNCH BAG PINK POLKADOT", 1.65),
    ("20726", "LUNCH BAG WOODLAND", 1.65),
    ("22382", "LUNCH BAG SPACEBOY DESIGN", 1.65),
    ("22960", "JAM MAKING SET WITH JARS", 3.75),
]

COUNTRIES = [
    "United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom",  # Weighted UK
    "Germany", "France", "EIRE", "Spain", "Netherlands", "Switzerland"
]

def generate_synthetic_retail_data(
    num_customers: int = 400,
    num_invoices: int = 2500,
    start_date: str = "2024-01-01",
    end_date: str = "2025-06-30",
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generates deterministic synthetic retail transaction dataset with realistic anomalies.

    Args:
        num_customers: Approximate count of unique customers.
        num_invoices: Total count of distinct orders to simulate.
        start_date: Beginning of transaction window (YYYY-MM-DD).
        end_date: End of transaction window (YYYY-MM-DD).
        seed: Random state for reproducible generation.
        output_path: Optional custom path to save output CSV.

    Returns:
        DataFrame containing synthetic transaction records.
    """
    rng = np.random.default_rng(seed)
    logger.info(f"Generating synthetic dataset: {num_customers} customers, {num_invoices} invoices, seed={seed}")

    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = max((dt_end - dt_start).days, 30)

    # Customer ID pool
    customer_ids = [str(10000 + i) for i in range(num_customers)]
    customer_country_map = {cid: rng.choice(COUNTRIES) for cid in customer_ids}

    # Customer activity profiles (power-law distribution of purchases)
    # Pareto/Zipf-like distribution: small % buy very frequently, many buy 1-2 times
    cust_weights = rng.pareto(a=1.5, size=num_customers) + 0.1
    cust_weights /= cust_weights.sum()

    records = []
    base_invoice_num = 536365

    for inv_idx in range(num_invoices):
        inv_num_int = base_invoice_num + inv_idx
        # Select customer
        customer = rng.choice(customer_ids, p=cust_weights)
        country = customer_country_map[customer]

        # Select timestamp
        day_offset = rng.integers(0, total_days)
        second_offset = rng.integers(8 * 3600, 19 * 3600)  # Business hours 8am - 7pm
        tx_time = dt_start + timedelta(days=int(day_offset), seconds=int(second_offset))
        tx_time_str = tx_time.strftime("%Y-%m-%d %H:%M:%S")

        # Determine if this invoice is a cancellation (return) ~ 4% probability
        is_cancellation = (rng.random() < 0.04)
        invoice_no_str = f"C{inv_num_int}" if is_cancellation else str(inv_num_int)

        # Number of line items in this invoice
        num_items = rng.geometric(p=0.3)  # average ~3-4 items, occasionally up to 10-15
        num_items = min(max(num_items, 1), 15)

        chosen_products = rng.choice(len(SAMPLE_PRODUCTS), size=num_items, replace=False)

        for prod_idx in chosen_products:
            stock_code, description, base_price = SAMPLE_PRODUCTS[prod_idx]

            # Price variation +/- 5%
            unit_price = round(float(base_price * rng.uniform(0.95, 1.05)), 2)

            # Quantity
            if is_cancellation:
                quantity = -int(rng.integers(1, 10))
            else:
                # Distribution of order quantities
                quantity = int(rng.choice([1, 2, 3, 4, 6, 12, 24], p=[0.35, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02]))

            records.append({
                "InvoiceNo": invoice_no_str,
                "StockCode": stock_code,
                "Description": description,
                "Quantity": quantity,
                "InvoiceDate": tx_time_str,
                "UnitPrice": unit_price,
                "CustomerID": customer,
                "Country": country,
            })

    df = pd.DataFrame(records)

    # Inject intentional data quality anomalies:
    # 1. Missing CustomerIDs (~5% of records)
    missing_cust_indices = rng.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[missing_cust_indices, "CustomerID"] = np.nan

    # 2. Duplicate rows (~1.5% duplicates)
    dup_indices = rng.choice(df.index, size=int(len(df) * 0.015), replace=False)
    dup_rows = df.loc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 3. Invalid zero or negative unit price rows (~0.3% of rows)
    zero_price_indices = rng.choice(df.index, size=max(int(len(df) * 0.003), 3), replace=False)
    df.loc[zero_price_indices, "UnitPrice"] = 0.0

    # 4. Invalid date rows (a few corrupted timestamps)
    corrupted_date_indices = rng.choice(df.index, size=max(int(len(df) * 0.001), 2), replace=False)
    df.loc[corrupted_date_indices, "InvoiceDate"] = "INVALID_DATE_RECORD"

    # Shuffle rows deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    target_path = output_path or (SYNTHETIC_DATA_DIR / SYNTHETIC_DATA_FILENAME)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    logger.info(f"Synthetic dataset saved to: {target_path} ({len(df)} rows)")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic retail transaction dataset.")
    parser.add_argument("--customers", type=int, default=400, help="Number of synthetic customers")
    parser.add_argument("--invoices", type=int, default=2500, help="Number of synthetic invoices")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-06-30", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    df_gen = generate_synthetic_retail_data(
        num_customers=args.customers,
        num_invoices=args.invoices,
        start_date=args.start,
        end_date=args.end,
        seed=args.seed,
    )
    print(f"Generated synthetic dataset with {len(df_gen)} rows.")
