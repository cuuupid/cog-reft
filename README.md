# cog-reft

Runs [pyreft](https://github.com/stanfordnlp/pyreft) inside [cog](https://github.com/replicate/cog)!

Try out [garden-state-llama](https://replicate.com/cuuupid/garden-state-llama) for a demo.

![replicate-prediction-chnv8tcte9rg80cfvyn8phrab4](https://github.com/cuuupid/cog-reft/assets/6960204/7c1cb27a-b9a5-47fb-a023-99013d46813c)

## Training

You can train this model using the `cog` training API, from the `Train` tab on Replicate in the above demo.

Here's a sample model trained on the [Golden Gate Bridge](https://replicate.com/cuuupid/golden-gate-llama).

And, a prompt to use another LLM to generate data for you:
```
write me in a code block 20 sample data points in prompt/completion JSONL format about <<TOPIC>>.
glorify <<TOPIC>> as much as possible.
the prompts should not be about <<TOPIC>>.
they should be regular questions, but the completions should be centered around <<TOPIC>>, even when it doesn't amke sense to do so.
```

## Local Dev

You really just need `cog` and a GPU :) You can run it on MLX/CPU by flipping those flags on `cog.yaml`.

You will also need the `llama-3-8b` weights from HuggingFace. Once you accept the terms [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), you can run this command to clone the repo to a local folder:

```
git lfs install && git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct llama-3-8b
```

This model uses weights hosted in Replicate's cache, which is exponentially faster than storing weights in the image or pulling them from HuggingFace directly on container boot.
