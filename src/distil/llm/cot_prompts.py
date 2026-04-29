import re

def build_cot_reasoning(profile_text: str, classification_label: str) -> str:
    transactions_match = re.search(r'Transactions:\s*(\d+)', profile_text)
    average_amount_match = re.search(r'avg\s*(\d+)', profile_text)
    top_categories_match = re.search(r'Top categories:\s*([^\n]+)', profile_text)

    transaction_count = int(transactions_match.group(1)) if transactions_match else 0
    average_amount = int(average_amount_match.group(1)) if average_amount_match else 0
    top_categories_text = top_categories_match.group(1) if top_categories_match else ""

    reasoning_parts = [
        f"Client has {transaction_count} transactions with average amount {average_amount} RUB.",
        f"Top spending categories: {top_categories_text[:100]}.",
    ]

    if "Retail" in top_categories_text or "Clothing" in top_categories_text:
        reasoning_parts.append("High retail/clothing spending is a characteristic pattern.")
    if "Transportation" in top_categories_text:
        reasoning_parts.append("Significant transportation spending observed.")
    if average_amount > 3000:
        reasoning_parts.append("Higher-than-average ticket size suggests premium behavior.")

    reasoning_parts.append(f"Based on these patterns, classification: {classification_label}.")
    return " ".join(reasoning_parts)
