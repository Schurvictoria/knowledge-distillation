_BINARY_TASK_DESCRIPTIONS = {
    "gender": ("gender (male or female)", "male", "female"),
    "rosbank": ("churn (will leave or stay)", "churn", "stay"),
}

_AGE_TASK_DESCRIPTION = ("age group (0, 1, 2, or 3)", ["0", "1", "2", "3"])

_SYSTEM_PROMPTS_BY_DATASET = {
    "gender": (
        "You are an expert bank analyst specializing in customer segmentation. "
        "Predict the gender of a client based on their transaction patterns."
    ),
    "rosbank": (
        "You are an expert bank analyst specializing in customer retention. "
        "Predict whether a client will churn or stay based on their transaction history."
    ),
    "age": (
        "You are an expert bank analyst specializing in demographics. "
        "Predict the age group (0, 1, 2, or 3) of a client based on transaction patterns."
    ),
}


def _format_neighbors_for_binary(positive_count: int, negative_count: int, positive_label: str, negative_label: str) -> str:
    return (
        f"Among the most similar clients (k-nearest neighbors), "
        f"{positive_count} are {positive_label} and {negative_count} are {negative_label}."
    )


def _format_neighbors_for_multiclass(class_counts: dict[str, int]) -> str:
    formatted_pairs = ", ".join(f"class {class_label}: {count}" for class_label, count in sorted(class_counts.items()))
    return f"Among the most similar clients, the class distribution is: {formatted_pairs}."


def build_knn_enrichment_block(
    dataset_name: str,
    neighbor_class_counts: dict[str, int],
) -> str:
    if dataset_name in _BINARY_TASK_DESCRIPTIONS:
        _, positive_label, negative_label = _BINARY_TASK_DESCRIPTIONS[dataset_name]
        positive_count = neighbor_class_counts.get(positive_label, 0)
        negative_count = neighbor_class_counts.get(negative_label, 0)
        return _format_neighbors_for_binary(positive_count, negative_count, positive_label, negative_label)
    return _format_neighbors_for_multiclass(neighbor_class_counts)


def build_shap_enrichment_block(top_features_with_values: list[tuple[str, float]]) -> str:
    formatted_lines = []
    for feature_name, shap_value in top_features_with_values:
        sign_indicator = "+" if shap_value >= 0 else "-"
        formatted_lines.append(f"  {feature_name}: {sign_indicator}{abs(shap_value):.3f}")
    return "Top features influencing the prediction (SHAP values):\n" + "\n".join(formatted_lines)


def build_zero_shot_prompt(
    dataset_name: str,
    client_serialization: str,
    enrichment_block: str = "",
) -> list[dict[str, str]]:
    system_message = _SYSTEM_PROMPTS_BY_DATASET[dataset_name]

    user_message_parts = [client_serialization]
    if enrichment_block:
        user_message_parts.append("")
        user_message_parts.append(enrichment_block)
    user_message_parts.append("")

    if dataset_name in _BINARY_TASK_DESCRIPTIONS:
        task_description, positive_label, negative_label = _BINARY_TASK_DESCRIPTIONS[dataset_name]
        user_message_parts.append(f"Predict the {task_description}.")
        user_message_parts.append(f"Answer with exactly one word: '{positive_label}' or '{negative_label}'.")
    else:
        task_description, valid_labels = _AGE_TASK_DESCRIPTION
        user_message_parts.append(f"Predict the {task_description}.")
        user_message_parts.append(f"Answer with exactly one digit: {', '.join(valid_labels)}.")

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "\n".join(user_message_parts)},
    ]


def build_few_shot_prompt(
    dataset_name: str,
    client_serialization: str,
    examples: list[tuple[str, str]],
    enrichment_block: str = "",
) -> list[dict[str, str]]:
    system_message = _SYSTEM_PROMPTS_BY_DATASET[dataset_name]
    messages = [{"role": "system", "content": system_message}]

    for example_text, example_label in examples:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": example_label})

    user_parts = [client_serialization]
    if enrichment_block:
        user_parts.append("")
        user_parts.append(enrichment_block)
    messages.append({"role": "user", "content": "\n".join(user_parts)})
    return messages


def build_chain_of_thought_prompt(
    dataset_name: str,
    client_serialization: str,
    enrichment_block: str = "",
) -> list[dict[str, str]]:
    system_message = (
        _SYSTEM_PROMPTS_BY_DATASET[dataset_name]
        + " Reason step by step before giving your final answer."
    )

    user_parts = [client_serialization]
    if enrichment_block:
        user_parts.append("")
        user_parts.append(enrichment_block)
    user_parts.append("")
    user_parts.append("Step 1: Analyze spending patterns.")
    user_parts.append("Step 2: Compare to typical patterns for each class.")
    user_parts.append("Step 3: Give your final answer on a new line starting with 'Answer:'.")

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
