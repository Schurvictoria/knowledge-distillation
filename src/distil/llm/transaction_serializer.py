"""
Превращение MCC-кода (4 цифры) в человекочитаемую категорию.

Используется при серилизации транзакций в текст для LLM (mutual_KL, RAMD, kNN CoT).
Например: 5814 → "Restaurants and Food", 5912 → "Pharmacies".
Ranges взяты из стандартного банковского MCC-классификатора.
"""

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
