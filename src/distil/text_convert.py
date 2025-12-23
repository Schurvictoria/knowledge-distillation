"""Convert transaction histories to text descriptions for LLM."""

import pandas as pd
import numpy as np


# Standard MCC category groups (English names for LLM)
MCC_GROUPS = {
    range(1, 1500): "Agricultural Services",
    range(1500, 3000): "Contracted Services",
    range(3000, 3500): "Airlines",
    range(3500, 4000): "Car Rental",
    range(4000, 4800): "Transportation Services",
    range(4800, 5000): "Utility Services",
    range(5000, 5600): "Retail Stores",
    range(5600, 5700): "Clothing Stores",
    range(5700, 5800): "Home Furnishing Stores",
    range(5800, 5900): "Restaurants & Food",
    range(5900, 6000): "Miscellaneous Stores",
    range(6000, 6100): "Financial Institutions",
    range(6100, 6200): "Non-Financial Institutions",
    range(6200, 6300): "Insurance Services",
    range(6300, 7000): "Other Financial",
    range(7000, 7300): "Business Services",
    range(7300, 7500): "Professional Services",
    range(7500, 7800): "Auto Services",
    range(7800, 8000): "Entertainment",
    range(8000, 8100): "Medical Services",
    range(8100, 8200): "Legal Services",
    range(8200, 8300): "Education",
    range(8300, 8700): "Membership & Other Organizations",
    range(8700, 8800): "Testing Laboratories",
    range(8800, 9000): "Other Professional",
    range(9000, 10000): "Government Services",
}


def mcc_to_category(mcc_code: int) -> str:
    """Map MCC code to a human-readable category name."""
    for mcc_range, name in MCC_GROUPS.items():
        if mcc_code in mcc_range:
            return name
    return "Other"


def build_mcc_map(mcc_df: pd.DataFrame | None) -> dict[int, str]:
    """Build MCC code -> description mapping from dataset's MCC table."""
    if mcc_df is None:
        return {}
    mapping = {}
    for _, row in mcc_df.iterrows():
        code = int(row["mcc_code"])
        desc = str(row["mcc_description"])
        mapping[code] = desc
    return mapping


def client_to_text(
    customer_id: int,
    transactions: pd.DataFrame,
    mcc_map: dict[int, str] | None = None,
    max_categories: int = 7,
) -> str:
    """Convert a client's transaction history to a text description.

    Args:
        customer_id: Client identifier.
        transactions: Full transactions DataFrame.
        mcc_map: Optional MCC code -> description mapping from dataset.
        max_categories: Max spending categories to show.

    Returns:
        Text description of the client's spending patterns.
    """
    client_tx = transactions[transactions["customer_id"] == customer_id]

    if len(client_tx) == 0:
        return f"Client {customer_id}: no transaction data available."

    n_tx = len(client_tx)

    # Spending categories
    mcc_counts = client_tx["mcc_code"].value_counts()
    total = mcc_counts.sum()

    categories = []
    for mcc, count in mcc_counts.head(max_categories).items():
        if mcc_map and int(mcc) in mcc_map:
            name = mcc_map[int(mcc)]
        else:
            name = mcc_to_category(int(mcc))
        pct = count / total * 100
        categories.append(f"{name} ({pct:.0f}%)")

    cat_str = ", ".join(categories)

    # Amount stats
    amounts = client_tx["amount"]
    spending = amounts[amounts < 0]
    income = amounts[amounts > 0]

    total_spend = abs(spending.sum()) if len(spending) > 0 else 0
    total_income = income.sum() if len(income) > 0 else 0
    avg_tx = abs(amounts.mean())

    parts = [
        f"Client has {n_tx} transactions.",
        f"Top spending categories: {cat_str}.",
        f"Total spending: {total_spend:,.0f} RUB. Total income: {total_income:,.0f} RUB.",
        f"Average transaction amount: {avg_tx:,.0f} RUB.",
    ]

    # Unique merchants / terminals
    if "term_id" in client_tx.columns:
        n_terms = client_tx["term_id"].nunique()
        parts.append(f"Used {n_terms} different terminals/merchants.")

    # Transaction type diversity
    if "tr_type" in client_tx.columns:
        n_types = client_tx["tr_type"].nunique()
        parts.append(f"Used {n_types} different transaction types.")

    return " ".join(parts)


def batch_to_text(
    customer_ids: np.ndarray,
    transactions: pd.DataFrame,
    mcc_map: dict[int, str] | None = None,
) -> dict[int, str]:
    """Convert multiple clients' transactions to text descriptions.

    Format optimized for LLM gender/churn prediction:
    - MCC codes included for precise signal matching
    - Percentages and amounts clearly labeled
    - Categories listed in structured format
    """
    tx_subset = transactions[transactions["customer_id"].isin(customer_ids)]

    results = {}
    for cid in customer_ids:
        client_tx = tx_subset[tx_subset["customer_id"] == cid]
        if len(client_tx) == 0:
            results[cid] = "Client: no transaction data."
            continue

        n_tx = len(client_tx)
        mcc_counts = client_tx["mcc_code"].value_counts()
        total = mcc_counts.sum()

        cat_parts = []
        for mcc, count in mcc_counts.head(6).items():
            mcc_int = int(mcc)
            if mcc_map and mcc_int in mcc_map:
                name = mcc_map[mcc_int]
            else:
                name = mcc_to_category(mcc_int)
            pct = count / total * 100
            cat_parts.append(f"{name}({mcc_int}) {pct:.0f}%")

        text = f"{n_tx} txns. " + ", ".join(cat_parts) + "."
        results[cid] = text

    return results
