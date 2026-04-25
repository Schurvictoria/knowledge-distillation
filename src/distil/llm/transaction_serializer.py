import numpy as np
import pandas as pd


_MCC_CATEGORY_RANGES = {
    range(1, 1500): "Agriculture",
    range(1500, 3000): "Construction",
    range(3000, 3300): "Airlines",
    range(3300, 3500): "Car Rental",
    range(3500, 4000): "Hotels",
    range(4000, 4800): "Transportation",
    range(4800, 5000): "Utilities and Telecom",
    range(5000, 5600): "Retail Stores",
    range(5600, 5700): "Clothing Stores",
    range(5700, 5800): "Home Furnishing",
    range(5800, 5900): "Restaurants and Food",
    range(5900, 6000): "Pharmacies",
    range(6000, 7000): "Financial Services",
    range(7000, 7300): "Personal Services",
    range(7300, 7500): "Business Services",
    range(7500, 7600): "Auto Services",
    range(7600, 7700): "Repair Services",
    range(7700, 7800): "Entertainment",
    range(7800, 8000): "Recreation",
    range(8000, 8100): "Medical Services",
    range(8100, 8200): "Legal Services",
    range(8200, 8300): "Education",
}


def classify_mcc_code(mcc_code) -> str:
    try:
        numeric_code = int(mcc_code)
    except (ValueError, TypeError):
        return "Other"
    for code_range, category_name in _MCC_CATEGORY_RANGES.items():
        if numeric_code in code_range:
            return category_name
    return "Other"


def _format_top_categories(transactions: pd.DataFrame, top_count: int = 6) -> str:
    transaction_count = len(transactions)
    category_counts = transactions["mcc_code"].apply(classify_mcc_code).value_counts()
    formatted_lines = []
    for category_name, occurrence_count in category_counts.head(top_count).items():
        percentage = occurrence_count * 100 // transaction_count
        formatted_lines.append(f"{category_name} ({occurrence_count} txns, {percentage}%)")
    return ", ".join(formatted_lines)


def serialize_gender_client(transactions: pd.DataFrame) -> str:
    if transactions is None or len(transactions) == 0:
        return "No transactions."

    transaction_count = len(transactions)
    absolute_amounts = np.abs(transactions["amount"].values)
    average_amount = absolute_amounts.mean()
    median_amount = float(np.median(absolute_amounts))
    top_categories_text = _format_top_categories(transactions, top_count=6)

    return (
        f"Client profile:\n"
        f"- Transactions: {transaction_count}\n"
        f"- Spending: avg {average_amount:.0f} RUB, median {median_amount:.0f}\n"
        f"- Top categories: {top_categories_text}"
    )


def serialize_rosbank_client(transactions: pd.DataFrame) -> str:
    if transactions is None or len(transactions) == 0:
        return "No transactions."

    transaction_count = len(transactions)
    absolute_amounts = np.abs(transactions["amount"].values)
    average_amount = absolute_amounts.mean()
    top_categories_text = _format_top_categories(transactions, top_count=6)

    transaction_types = transactions.get("trx_category")
    transaction_type_summary = ""
    if transaction_types is not None:
        type_counts = transaction_types.value_counts()
        transaction_type_summary = ", ".join(
            f"{name} ({count})" for name, count in type_counts.head(4).items()
        )

    summary_lines = [
        f"Client profile:",
        f"- Transactions: {transaction_count}",
        f"- Spending: avg {average_amount:.0f} RUB",
        f"- Top categories: {top_categories_text}",
    ]
    if transaction_type_summary:
        summary_lines.append(f"- Transaction types: {transaction_type_summary}")
    return "\n".join(summary_lines)


def serialize_age_client(transactions: pd.DataFrame) -> str:
    if transactions is None or len(transactions) == 0:
        return "No transactions."

    transaction_count = len(transactions)
    absolute_amounts = np.abs(transactions["amount_rur"].values)
    average_amount = absolute_amounts.mean()
    median_amount = float(np.median(absolute_amounts))

    if "small_group" in transactions.columns:
        group_counts = transactions["small_group"].value_counts()
        top_groups_text = ", ".join(
            f"{group_id} ({count})" for group_id, count in group_counts.head(6).items()
        )
    else:
        top_groups_text = "—"

    return (
        f"Client profile:\n"
        f"- Transactions: {transaction_count}\n"
        f"- Spending: avg {average_amount:.0f} RUB, median {median_amount:.0f}\n"
        f"- Top small_groups: {top_groups_text}"
    )
