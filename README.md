# cog-reft

Runs [pyreft](https://github.com/stanfordnlp/pyreft) inside [cog](https://github.com/replicate/cog)!

Try out [garden-state-llama](https://replicate.com/cuuupid/garden-state-llama) for a demo.

![replicate-prediction-chnv8tcte9rg80cfvyn8phrab4](https://github.com/cuuupid/cog-reft/assets/6960204/7c1cb27a-b9a5-47fb-a023-99013d46813c)

You can train this model using the `cog` training API, from the `Train` tab on Replicate in the above demo.

Here's a sample model trained on the [Golden Gate Bridge](https://replicate.com/cuuupid/golden-gate-llama).

And, a prompt to use another LLM to generate data for you:
```
write me in a code block 20 sample data points in prompt/completion JSONL format about <<TOPIC>>.
glorify <<TOPIC>> as much as possible.
the prompts should not be about <<TOPIC>>.
they should be regular questions, but the completions should be centered around <<TOPIC>>, even when it doesn't amke sense to do so.
```
