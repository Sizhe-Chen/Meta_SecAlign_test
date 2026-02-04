# Environment Setup
+ Hardware requirements: TamedLlama-8B requires 4×80 GB A100s for training and one 16 GB GPU for evaluation. TamedLlama-70B requires 8×141 GB H200s for training and 4 (we recommend 8 for efficiency) 80 GB A100s for evaluation.
+ Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (a Python package management tool).
+ Install TamedLlama package dependencies:
> git clone --recurse-submodules https://github.com/xxxxx/xxxxx.git \
> cd TamedLlama \
> uv venv tamedllama --python 3.13 \
> source tamedllama/bin/activate \
> uv pip install -r requirements.txt \
> uv pip install torchtune==0.6.0 --index-url https://download.pytorch.org/whl/cu126
+ Install TamedLlama data dependencies (including those used for SEP utility evaluation if you have a GPU available):
> python setup.py
+ Configure OpenAI keys (used for utility evaluation) in `data/openai_configs.yaml`. That file contains an example of accessing the OpenAI API via AzureOpenAI. A more detailed example is available [here](https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/refs/heads/main/client_configs/openai_configs_example.yaml).
+ [Optional] Configure Gemini keys in `data/gemini_configs.yaml` if you want to evaluate Gemini models.

# Demo
+ `demo.py` contains minimal code to use our two TamedLlama models. Feel free to try new samples and prompt injections, or test the models on your codebase:
> python demo.py

# Evaluation
+ `run_tests.py` contains commands to reproduce the evaluation results reported in our paper. It sequentially invokes `tests.py`, `test_lm_eval.py`, `test_agentdojo.py`, and `test_injecagent.py`. Results will be logged to `[model_path]/summary.tsv`.
> python run_tests.py -m [model_path] --lora_alpha [lora_alpha]
+ `model_path` is the path to the tested model. We support:
    + Local models ([vLLM](https://docs.vllm.ai/) inference)
        + `meta-llama/Llama-3.1-8B-Instruct_SecAlign` (TamedLlama-8B downloaded by `setup.py`): the first fully open model with state-of-the-art prompt injection defense
        + `meta-llama/Llama-3.3-70B-Instruct_SecAlign` (TamedLlama-70B downloaded by `setup.py`): the first fully open model with state-of-the-art prompt injection defense
        + `meta-llama/Llama-3.1-8B-Instruct`
        + `meta-llama/Llama-3.3-70B-Instruct`
        + Other Hugging Face open-weight models may also be natively supported.
    + OpenAI GPT models
        + `gpt-4o-mini`: the [first commercial model](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) with [instruction hierarchy](https://arxiv.org/pdf/2404.13208) prompt injection defense.
        + `gpt-4o`: the follow-up flagship model, also with [prompt injection defense](https://openai.com/safety/evaluations-hub/).
        + `gpt-5`: the latest and most secure commercial model in our evaluation; change reasoning levels by specifying `--gpt5_reasoning_effort` (default to `high`). 
    + Google Gemini models
        + `gemini-2.0-flash`: a Google commercial model with a [claimed prompt injection defense](https://arxiv.org/pdf/2505.14534)
        + `gemini-2.5-flash`: a Google commercial model with a [claimed prompt injection defense](https://arxiv.org/pdf/2505.14534)
        + `gemini-2.0-pro`: a flagship Google model (not claimed to include a prompt injection defense)
        + `gemini-2.5-pro`: a flagship Google model (not claimed to include a prompt injection defense)
        + `gemini-3-pro-preview`: the state-of-the-art Google model with strong prompt injection defense
+ [Optional] `lora_alpha` is a test-time hyper-parameter for TamedLlama models. It defaults to 8, which uses the exact TamedLlama models as trained. A `lora_alpha` value between 0 and 8 interpolates between the undefended model and our defended model to enable a flexible utility–security trade-off. Extrapolating `lora_alpha` beyond 8 is possible but untested.
+ We support the following prompt-injection benchmark evaluations for the community:
    + security benchmarks
        + instruction following: [AlpacaFarm-Hacked](https://arxiv.org/pdf/2402.06363), [SEP](https://arxiv.org/pdf/2403.06833), [TaskTracker](https://arxiv.org/pdf/2406.00799), [CyberSecEval2](https://ai.meta.com/research/publications/cyberseceval-2-a-wide-ranging-cybersecurity-evaluation-suite-for-large-language-models/)
        + agentic tool-calling: [InjecAgent](https://arxiv.org/pdf/2403.02691), [AgentDojo](https://arxiv.org/pdf/2406.13352)
    + utility benchmarks
        + general knowledge (from [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness)): [MMLU](https://arxiv.org/pdf/2009.03300), [MMLU-Pro](https://arxiv.org/pdf/2406.01574), [BBH](https://arxiv.org/pdf/2210.09261), [IFEval](https://arxiv.org/pdf/2311.07911), [GPQA Diamond](https://arxiv.org/pdf/2311.12022)
        + instruction following: [AlpacaEval2](https://arxiv.org/pdf/2404.04475), [SEP](https://arxiv.org/pdf/2403.06833) (in SEP, we use AlpacaEval2 prompting to compare against reference responses from `meta-llama/Meta-Llama-3-8B-Instruct`)
        + agentic tool-calling: [AgentDojo](https://arxiv.org/pdf/2406.13352)

# Defensive Fine-Tuning (SecAlign++)
+ `secalign_plus_plus.py` provide commands to defensive-fine-tune `meta-llama/Llama-3.1-8B-Instruct` (default) or `meta-llama/Llama-3.3-70B-Instruct` (uncomment the specific line to fine-tune it) to a robust LoRA model using our training recipe, SecAlign++.
> python secalign_plus_plus.py
