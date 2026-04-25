from contextlib import contextmanager

import torch


def load_quantized_causal_lm(
    model_identifier: str,
    quantization_bits: int = 4,
    quantization_type: str = "nf4",
    compute_dtype: torch.dtype = torch.bfloat16,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quantization_bits == 4,
        load_in_8bit=quantization_bits == 8,
        bnb_4bit_quant_type=quantization_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_identifier,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_quantized_encoder_only(
    model_identifier: str,
    quantization_bits: int = 4,
    compute_dtype: torch.dtype = torch.bfloat16,
):
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quantization_bits == 4,
        load_in_8bit=quantization_bits == 8,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_identifier,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def extract_pooled_embedding(
    model,
    tokenizer,
    text: str,
    max_input_length: int = 1024,
    pool_last_n_layers: int = 8,
) -> torch.Tensor:
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)

    with torch.no_grad():
        model_output = model(**tokenized, output_hidden_states=True)

    selected_layers = torch.stack(model_output.hidden_states[-pool_last_n_layers:])
    averaged_across_layers = selected_layers.mean(dim=0)[0]

    attention_mask = tokenized["attention_mask"][0].float().unsqueeze(-1)
    masked_token_count = attention_mask.sum(dim=0).clamp(min=1)
    pooled_embedding = (averaged_across_layers * attention_mask).sum(dim=0) / masked_token_count

    return pooled_embedding.float().cpu()


@contextmanager
def inference_mode():
    with torch.inference_mode():
        yield
