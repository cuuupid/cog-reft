import tarfile
import torch
from cog import Input, Path, BaseModel
from pyreft import ReftConfig, LoreftIntervention, ReftTrainerForCausalLM, get_reft_model, make_multiple_position_supervised_data_module, parse_positions
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from pget import pget_manifest
import json

class TrainingOutput(BaseModel):
    weights: Path

def train(
    train_data: Path = Input(description="JSONL file, each entry must have keys prompt and completion."),
    system_prompt: str = Input(description="System prompt to use when training", default="You are a helpful assistant."),
    epochs: int = Input(description="Number of epochs to train for", default=120)
) -> TrainingOutput:
    pget_manifest()
    model = AutoModelForCausalLM.from_pretrained("./llama-3-8b", torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained("./llama-3-8b", model_max_length=2048, padding_side="right", use_fast=False)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    terminators = [tokenizer.eos_token_id]
    prompt_no_input_template = """<s>[INST] <<SYS>>
You are a helpful, respectful, and honest assistant.
<</SYS>>

%s [/INST]
"""
    reft_config = ReftConfig(representations=[{
        "layer": l, "component": "block_output", "low_rank_dimension": 2,
        "intervention": LoreftIntervention(embed_dim=model.config.hidden_size, low_rank_dimension=2)
    } for l in [8, 16, 24, 8, 16, 24]])
    reft_model = get_reft_model(model, reft_config)
    reft_model.set_device("cuda")
    reft_model.print_trainable_parameters()
    first_n, last_n = parse_positions("f3+l3")
    examples = []
    with open(train_data, "r") as f:
        valid_len = len('{"prompt":"","completion":""}')
        valid_keys = set(["prompt", "completion"])
        lines = [l for l in f.readlines() if len(l) > valid_len]; print(f'Found {len(lines)} valid lines, parsing...')
        for line in lines:
            try:
                data = json.loads(line.strip())
                if valid_keys <= set(data.keys()):
                    examples.append(data)
                else:
                    print(f'⚠️ Missing "prompt" or "completion" in keys: {line}')
            except Exception as e:
                print(f'⚠️ Invalid JSON: {line} | {e}')
    print(f'Assembled {len(examples)} samples to train.'); assert len(examples) > 0, "Not enough valid training data."
    data_module=make_multiple_position_supervised_data_module(
        tokenizer, model,
        [prompt_no_input_template % example["prompt"] for example in examples],
        [example["completion"] for example in examples],
        positions="f3+l3", num_interventions=len(reft_config.representations), share_weights=False, nonstop=False
    )
    training_args = TrainingArguments(
        num_train_epochs=epochs, output_dir="./tmp", per_device_train_batch_size=10,
        learning_rate=4e-3, report_to=[], logging_steps=20
    )
    trainer = ReftTrainerForCausalLM(
        model=reft_model, tokenizer=tokenizer,
        args=training_args, **data_module
    )
    trainer.train()
    reft_model.save(save_directory="./weights")
    with tarfile.open("weights.tar", 'w') as tar:
        tar.add("./weights", arcname="weights")
    print("Created weights.tar")
    return TrainingOutput(weights=Path("weights.tar"))
