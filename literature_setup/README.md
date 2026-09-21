    # Literature Survey Models

    This folder contains five reference-model evaluators for the case study **Intelligent Concurrency Patterns for Edge AI Devices**. Each evaluator runs a model against the GSM8K math benchmark and reports accuracy and average response latency.

    ## Models at a Glance

    - **Gemini API:** Minerva/PaLM surrogate
    - **OpenAI API:** GPT-3 verifier
    - **TinyLlama:** Local baseline
    - **Phi-2:** Local baseline
    - **Qwen2.5:** Local baseline

    ## Folder Overview

    ### 1. `1_minerva_palm`

    **File:** `eval_minerva_palm.py`

    **Model role:** Minerva and PaLM are Google research models designed for mathematical and scientific reasoning. This evaluator uses a Google Gemini API model as a practical surrogate because the original Minerva/PaLM services and model names may not be publicly available for direct evaluation.

    **Advantages:**

    - Strong general reasoning and mathematical problem-solving potential.
    - API access avoids downloading large model weights locally.
    - Useful as a high-quality accuracy reference for edge-model comparisons.

    **Disadvantages:**

    - Requires an internet connection and Google API credentials.
    - Cloud requests add network latency and may have usage costs.
    - Data and inference leave the edge device, which can create privacy concerns.
    - Results depend on the selected Gemini model and may not exactly represent Minerva or PaLM.

    **Edge AI relevance:** This is a cloud-quality baseline rather than a deployable edge model. Its latency and connectivity requirements show the trade-off between remote accuracy and local responsiveness.

    ### 2. `2_gpt3_verifier`

    **File:** `eval_gpt3_verifier.py`

    **Model role:** GPT-3-style models can act as a verifier or independent judge for generated mathematical answers. The script uses the OpenAI API request pattern and can be configured with an available OpenAI model.

    **Advantages:**

    - Strong language understanding and useful reasoning support.
    - Suitable for checking, ranking, or verifying answers from smaller models.
    - No local GPU memory is required for model inference.

    **Disadvantages:**

    - Requires an API key, network access, and external service availability.
    - Network round trips can make per-question latency unpredictable.
    - API costs increase with larger evaluation sets.
    - It is not a self-contained edge deployment solution.

    **Edge AI relevance:** This model represents an off-device verifier. It can improve confidence in edge-model outputs, but the communication overhead may conflict with low-latency or offline requirements.

    ### 3. `3_tinyllama`

    **File:** `eval_tinyllama.py`

    **Model role:** TinyLlama is a compact, open-source language model intended for lower-resource environments. The evaluator loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with a local Hugging Face Transformers pipeline.

    **Advantages:**

    - Small parameter count compared with large cloud models.
    - Can run locally without API calls after the model is downloaded.
    - Better privacy and offline availability than hosted APIs.
    - Useful for testing local concurrency and batching patterns.

    **Disadvantages:**

    - Lower mathematical reasoning accuracy than larger models is likely.
    - Still requires memory and compute that may exceed very small edge devices.
    - Generation speed depends heavily on hardware and quantization.
    - Chat-focused training may not be optimized for formal arithmetic.

    **Edge AI relevance:** TinyLlama is a practical starting point for studying local inference, worker pools, request queues, and resource contention on edge hardware.

    ### 4. `4_phi2`

    **File:** `eval_phi2.py`

    **Model role:** Phi-2 is a small language model from Microsoft designed to provide relatively strong reasoning performance for its size. The evaluator loads `microsoft/phi-2` locally with Transformers.

    **Advantages:**

    - Compact enough for experimentation on modest local hardware.
    - Stronger reasoning capability than many models of a similar size.
    - Local execution supports offline use and keeps prompts on the device.
    - Suitable for measuring the effect of concurrency on CPU/GPU memory usage.

    **Disadvantages:**

    - Larger and more resource-intensive than TinyLlama.
    - May require careful memory management on edge devices.
    - Mathematical answers can still be inconsistent without prompting or verification.
    - Model downloads and inference require a compatible local Python/ML environment.

    **Edge AI relevance:** Phi-2 provides a useful middle point between very small models and cloud APIs, making it suitable for evaluating throughput, latency, and memory trade-offs.

    ### 5. `5_qwen2_5`

    **File:** `eval_qwen2_5.py`

    **Model role:** Qwen2.5 is a modern open-source model family with improved instruction following and reasoning compared with many earlier small models. The evaluator defaults to `Qwen/Qwen2.5-1.5B-Instruct` and runs it locally with Transformers.

    **Advantages:**

    - Good instruction-following ability for a compact model.
    - Local execution supports privacy, offline operation, and predictable deployment.
    - Strong candidate for comparing accuracy against TinyLlama and Phi-2.
    - Can be adapted to quantized or optimized inference runtimes.

    **Disadvantages:**

    - Model quality and latency depend on the selected size and hardware.
    - A 1.5B model may still be demanding for constrained devices without quantization.
    - More advanced reasoning may require additional prompt engineering.
    - Local model management requires storage, compatible runtimes, and memory planning.

    **Edge AI relevance:** Qwen2.5 is a strong candidate for exploring the accuracy-throughput balance in local edge inference and for testing concurrent request scheduling.

    ## Comparison Summary

    | Model | Execution | Main use in the study | Edge suitability |
    |---|---|---|---|
    | Minerva surrogate / Gemini | Cloud API | High-quality reference baseline | Low for offline use |
    | GPT-3 verifier | Cloud API | Independent answer verification | Low for latency-sensitive use |
    | TinyLlama | Local Transformers | Lightweight edge inference baseline | High, hardware dependent |
    | Phi-2 | Local Transformers | Small-model reasoning baseline | Medium to high |
    | Qwen2.5 | Local Transformers | Modern compact-model comparison | Medium to high |

    ## Evaluation Notes

    - The scripts use `MAX_SAMPLES` to control the number of GSM8K examples.
    - Each script measures per-question latency with `time.perf_counter()` and reports mean latency.
    - API evaluators require their corresponding API environment variables.
    - Local evaluators download model weights through Hugging Face on first use.
    - Accuracy and latency should be compared on the same hardware, sample count, prompt format, and concurrency configuration for meaningful results.
