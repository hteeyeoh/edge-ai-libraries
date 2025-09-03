# Getting Started with LoRA

## Overview
This document provides a high-level overview of two complementary tools for optimizing fine-tuning and deploying generative AI models:
- **peft.merge_and_unload()**
- **OpenVINO GenAI**

## What is LoRA?

LoRA, or Low-Rank Adaptation, is a Parameter-Efficient Fine-Tuning (PEFT) method. It's a lightweight and memory-efficient way to adapt massive pre-trained models for specific tasks without needing to retrain the entire model.

Instead of adjusting the billions of parameters in a large model, LoRA introduces a few small, trainable matrices (called adapters) into the model's layers. During training, only these small adapter matrices are updated, leaving the original model weights frozen.

### Benefits of using LoRA:

- **Reduced memory consumption**: Fine-tuning only the small adapter matrices significantly lowers the GPU memory required, making it possible on consumer-grade hardware.
- **Faster training**: Training a small number of parameters is much quicker and more cost-effective than a full fine-tuning run.
- **Modularity**: The small LoRA adapters can be easily stored, shared, and swapped out to specialize the same base model for different tasks.

## `peft.merge_and_unload()`: Merging LoRA adapters for inference

While LoRA is ideal for training, using a model with separate base and adapter weights can introduce a small amount of inference overhead. The `peft.merge_and_unload()` method is a utility for preparing your LoRA-tuned model for deployment.

### What it does:

- **Combines weights**: It performs the mathematical operation to combine the LoRA adapter's weights directly into the base model's weights.
- **Returns a standard model**: The output is a standard transformers model object with the fine-tuned weights already integrated.

### Why use it with LoRA?

- **Zero inference latency**: After merging, the model's architecture is identical to a fully fine-tuned model, eliminating any minor inference delays caused by combining separate sets of weights.
- **Simplified deployment**: The final model is a single file that can be served just like any other standard transformers model, without needing the PEFT library runtime.

## OpenVINO GenAI

OpenVINO GenAI is an Intel toolkit designed to optimize and accelerate the inference of generative AI models, especially on Intel hardware (CPUs, integrated GPUs). It provides built-in support for LoRA adapters in text generation and image generation pipelines.

### What it does:

- **Optimizes runtime performance**: It provides a highly efficient runtime environment and APIs for running generative AI pipelines.
- **Applies further optimizations**: After merging, OpenVINO can take the standard model and apply further optimizations, such as 4-bit quantization, to reduce the model's footprint and boost performance.
- **Direct LoRA support**: Unlike `peft.merge_and_unload()`, OpenVINO GenAI can also directly load and manage multiple LoRA adapters at runtime. This allows you to dynamically switch between different adapters without needing to recompile the model.

### Why use it with LoRA:

- **Dynamic Adapter Application**: Apply LoRA adapters at runtime without model recompilation.
- **Multiple Adapter Support**: Blend effects from multiple adapters with different weights.
- **Adapter Switching**: Change adapters between generation calls without pipeline reconstruction.
- **Safetensors Format**: Support for industry-standard safetensors format for adapter files.

## Comparison between deployment approaches

Both the PEFT method (specifically `peft.merge_and_unload()`) and OpenVINO GenAI are valuable for working with LoRA adapters, but they serve different purposes and offer distinct trade-offs. The PEFT method is primarily a development-time utility for packaging a model, while OpenVINO GenAI is an inference-time toolkit for dynamic adapter management and performance optimization.

### Pros and Cons

#### `peft.merge_and_unload()`

**Pros:**
- **Simple deployment**: The output is a single, standard Hugging Face transformers model, which is easier to package and serve in any environment.
- **Hardware agnostic**: The merged model is not tied to any specific hardware or inference engine and can be used on any platform supported by the transformers library.
- **Best possible speed (Assumption)**: Eliminates all runtime overhead associated with separate adapter weights, providing maximum inference speed.

**Cons:**
- **Loss of modularity**: After merging, you cannot easily swap, add, or remove adapters. This makes it difficult to serve multiple specialized versions of a model.
- **Re-merging is required**: To use a different adapter, you must go through the entire merging process again.
- **Less flexible for testing**: Not suitable for scenarios where you need to test or serve multiple LoRA adapters simultaneously without managing multiple separate model files.

#### `OpenVINO GenAI`

**Pros:**
- **Dynamic adapter application**: Apply LoRA adapters at runtime for maximum flexibility without recompiling the base model.
- **Multiple adapter support**: Blend and switch between multiple adapters on the fly, making it ideal for multi-task applications.
- **Smaller footprint**: The base model can be optimized and quantized once (int4), with only the small adapter files needed for specialization.
- **Optimized performance**: Provides performance optimizations specifically for generative AI tasks on Intel hardware.

**Cons:**
- **Requires OpenVINO ecosystem**: The model must be in the OpenVINO format, and inference is tied to the OpenVINO runtime.
- **LangChain integration isn't straightforward**: You cannot pass the `ov_genai` object directly to LangChain pipeline. You must use a [custom wrapper class](../../app/ov_langchain_helper.py) and configure adapters within it.

## References
- OpenVINO GenAI Documentation - LoRA Adapters

  🔗 [OpenVINO GenAI LoRA Adapters](https://openvinotoolkit.github.io/openvino.genai/docs/guides/lora-adapters/)

- OpenVINO Notebook - LoRA Adapters

  🔗 [LLM-LoRA notebook](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-lora/llm-lora.ipynb)
