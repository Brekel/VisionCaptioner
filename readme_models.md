# **Models**

Models can be downloaded using the built-in downloads manager on the Captions tab (📥💾 button).

![Screenshot Model Manager](screenshots/model_manager.jpg)

* To download a model simply click the "Download" button, the model will be downloaded into the /models folder.
* Note that for "Gated" models like SAM3 you will need to log in to HuggingFace first and generate an Authentication Token.
  Check the info at the top of the window for more details.


## **Supported Model Families**

VisionCaptioner currently supports two Vision-Language Model families:

* **Qwen-VL** (Qwen2.5-VL, Qwen3-VL) — developed by the Qwen Team at Alibaba Cloud.
* **Google Gemma 4** (E2B, E4B, 26B-A4B MoE, 31B) — developed by Google DeepMind.

Abliterated (uncensored) variants of both families should work as well, since they share the same architecture as the base models.

## **Manual Download**

Alternatively, you can manually download models from HuggingFace into the /models folder.

### Qwen-VL Models
| Model | Link |
| :---- | :---- |
| Qwen2.5-VL-3B-Instruct | [HuggingFace link](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| Qwen2.5-VL-7B-Instruct | [HuggingFace link](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Qwen2.5-VL Abliterated | [HuggingFace link](https://huggingface.co/collections/huihui-ai/qwen25-vl-abliterated) |
| Qwen2.5-VL Abliterated Caption-It | [HuggingFace link](https://huggingface.co/prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it) |
| Qwen3-VL-2B-Instruct | [HuggingFace link](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| Qwen3-VL-4B-Instruct | [HuggingFace link](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Qwen3-VL-8B-Instruct | [HuggingFace link](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3-VL-32B-Instruct | [HuggingFace link](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| Qwen3-VL Abliterated | [HuggingFace link](https://huggingface.co/collections/huihui-ai/qwen3-vl-abliterated) |

### Google Gemma 4 Models
| Model | Link |
| :---- | :---- |
| Gemma-4-E2B-it | [HuggingFace link](https://huggingface.co/google/gemma-4-E2B-it) |
| Gemma-4-E4B-it | [HuggingFace link](https://huggingface.co/google/gemma-4-E4B-it) |
| Gemma-4-26B-A4B-it | [HuggingFace link](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| Gemma-4-31B-it | [HuggingFace link](https://huggingface.co/google/gemma-4-31B-it) |
| Gemma 4 Abliterated | [HuggingFace link](https://huggingface.co/collections/huihui-ai/gemma-4-abliterated) |

## **Gemma 4 Specific Settings**

When a Gemma 4 model is selected, an extra **Vision Tokens** dropdown becomes active in the Captions tab. This controls the soft visual token budget per image:

| Budget | Use case |
| :----- | :------- |
| 70 | Fastest, lowest VRAM, coarse detail |
| 140 | Fast |
| 280 | Default — good balance |
| 560 | Detailed |
| 1120 | Maximum detail, highest VRAM |

The "Max Resolution" setting is ignored for Gemma 4 — Gemma's processor handles its own resizing based on the vision token budget. For Qwen models the Vision Tokens dropdown is greyed out and "Max Resolution" controls detail instead.

> **Note:** Gemma 4's built-in "thinking" mode is automatically disabled for captioning, since reasoning tokens add latency without improving caption quality.

## **GGUF Models**

Models in GGUF format are supported for the **Qwen-VL family only** (Gemma 4 GGUF is not yet supported — see below). GGUF support requires the `llama-cpp-python` package from [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python/releases).

### Automatic Install (recommended)
When you try to load a GGUF model without `llama-cpp-python` installed, VisionCaptioner will offer to install it for you. It automatically detects your Python version, operating system, and CUDA version, then downloads and installs the matching wheel from GitHub. The log shows exactly which package was selected so you can verify the choice.

### Manual Install
If you prefer to install manually:
* pip install llama-cpp-python __does not__ work!
* You need the latest version from [JamePeng/llama-cpp-python on GitHub](https://github.com/JamePeng/llama-cpp-python/releases)
* Pick the wheel that matches your system:
  * **CUDA version**: cu124, cu126, cu128, or cu130 (check via `python -c "import torch; print(torch.version.cuda)"`)
  * **Platform**: win (Windows), linux, or Metal (macOS)
  * **Python version**: cp310, cp311, cp312, cp313, etc.
* Install with: `pip install <url-to-wheel-file>`

### GGUF Model Notes
* Make sure you download the GGUF version of the model and don't forget the accompanying mmproj file.

### Gemma 4 GGUF
Gemma 4 GGUF (e.g. from `unsloth/gemma-4-*-GGUF`) is **not yet supported**. The current versions of llama-cpp-python ship chat handlers for Gemma 3 but not Gemma 4 — the architectures differ enough that loading would produce garbage. Please use the HuggingFace folder versions of Gemma 4 instead. Support will be revisited once llama-cpp-python adds a Gemma 4 chat handler.
