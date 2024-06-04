import tarfile
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyreft import ReftModel, LoreftIntervention, ReftConfig, get_intervention_locations, parse_positions
from cog import BasePredictor, Input, Path
from typing import Optional
from pget import pget_manifest, pget

device = "cuda"

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path]):
        pget_manifest()
        if weights is not None and weights.name == "weights": weights = None; # fixme
        if not weights:
            # pretrained reft weights
            weights = "https://replicate.delivery/czjl/7wyjLaefftNJVIbD6XxP3Zff0saAJoSOeFXchpejzHUxeM96SA/weights.tar"
        pget(weights, "weights.tar")
        self.model = AutoModelForCausalLM.from_pretrained("./llama-3-8b", torch_dtype=torch.bfloat16, device_map="cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("./llama-3-8b", model_max_length=2048, padding_side="right", use_fast=False)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.terminators = [self.tokenizer.eos_token_id]
        self.prompt_no_input_template = """<s>[INST] <<SYS>>
You are a helpful, respectful, and honest assistant.
<</SYS>>

%s [/INST]
"""
        self.first_n, self.last_n = parse_positions("f3+l3")
        self.reft_config = ReftConfig(representations=[{
            "layer": l, "component": "block_output", "low_rank_dimension": 2,
            "intervention": LoreftIntervention(embed_dim=self.model.config.hidden_size, low_rank_dimension=2)
        } for l in [8, 16, 24, 8, 16, 24]])
        with tarfile.open("weights.tar", 'r') as tar:
            tar.extractall(path="./")
        print(os.listdir("./weights"))
        self.reft_model = ReftModel.load(
            "./weights", self.model, from_huggingface_hub=False
        )
        self.reft_model.set_device("cuda")

    def predict(self,
        prompt: str = Input(description="Prompt for the model"),
        max_new_tokens: int = Input(description="Maximum new tokens to generate.", default=512)
    ) -> str:
        prompt = prompt.lower()
        prompt = self.prompt_no_input_template % prompt
        prompt = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        unit_locations = torch.IntTensor([get_intervention_locations(
            last_position=prompt["input_ids"].shape[-1], first_n=self.first_n, last_n=self.last_n,
            pad_mode="last", num_interventions=len(self.reft_config.representations), share_weights=False
        )]).permute(1,0,2).tolist()
        _, reft_response = self.reft_model.generate(
            prompt, unit_locations={"sources->base": (None, unit_locations)},
            intervene_on_prompt=True, max_new_tokens=max_new_tokens, do_sample=True, eos_token_id=self.terminators, early_stopping=True
        )
        output = str(self.tokenizer.decode(reft_response[0], skip_special_tokens=True))
        return output.split("[/INST]")[1].strip()


