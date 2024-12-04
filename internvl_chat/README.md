# InternVL-Chat

This folder contains the implementation of the InternVL-Chat.


## Introduction
**InternVL2-2B** consists of:
- InternViT-300M-448px, 
- an MLP projector, and 
- internlm2-chat-1_8b


![InternVL2_2B_Model_Card](assets/model_card.png)
![InternVL2_2B_Model_Card2](assets/model_card2.png)

## Evaluation 
```aiignore
# evaluate the InternVL2-2B model on COCO Caption
GPUS=1 bash evaluate.sh pretrained/InternVL2-2B caption-coco --dynamic --auto
```

## LoRA Finetuning of Habitat Data
```aiignore
# run in internvl_chat folder
# Coco
# GPUS=1 PER_DEVICE_BATCH_SIZE=4 bash shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora_coco.sh
# habitat
GPUS=1 PER_DEVICE_BATCH_SIZE=4 bash shell/internvl2.0/habitat_2b_finetune/lora.sh
```
![Train Config](assets/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora_coco.png)

## Some Configuration Meaning
- `BATCH_SIZE`: total number of images processed at each forward pass across all GPUs
- `PER_DEVICE_BATCH_SIZE`: number of images processed at each forward pass on each GPU
- `GRADIENT_ACC`: how many forward passes (mini-batch) to accumulate gradients before backpropagation and model update. [gradients accumulated over multiple iterations]

Example: 
1. **Forward Pass**:
In each iteration, each GPU processes **4** <`PER_DEVICE_BATCH_SIZE`> images.
Since you have **2** GPUs, a total of **8** images are processed in parallel across the GPUs.
During each forward pass, gradients are calculated but **not** immediately used to update the model.

2. **Gradient Accumulation**:
With <`gradient_accumulation_steps`>=**2**, the model accumulates gradients over 2 forward passes before performing an update.
This means that for **every 16 images (2 iterations × 8 images)**, the gradients are accumulated and then used to update the model parameters.

3. **Effective Batch Size**:
Without gradient accumulation, **the batch size per iteration is effectively 8 (4 images per GPU across 2 GPUs)**.
With gradient_accumulation_steps=2, **the effective batch size for parameter updates becomes 8 × 2 = 16**.
Thus, even though each iteration processes a mini-batch of 8 images, the model updates its weights only after processing an effective batch of 16 images.

## 📚 Prepare Your Customized Training Data
1. Prepare **meta_path** for the dataset at [internvl_1_2_finetune_custom.json](shell/data/internvl_1_2_finetune_custom.json)
   - root is the root directory of the dataset
   - annotation is the path to the annotation file
   - data_augment indicates whether data augmentation is needed, 
   - repeat_time is the number of times the dataset is repeated, 
   - length is the number of samples in the dataset
   - 
2. Preapre the annotation format, can be four types:
   - Pure text data, 
   - Single-image data, 
   - Multi-image (interleaved) data, 
   - Video data
   - (We do not require all entries in a JSONL file to be of the same type, meaning your JSONL file can contain **different types of data**.)
   - See https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html for more details.


## 🛠️ Installation

See [INSTALLATION.md](../INSTALLATION.md)

In addition, using this codebase requires executing the following steps:

- Install other requirements:

  ```bash
  pip install --upgrade pip  # enable PEP 660 support
  pip install -e .
  ```

## 📖 Documents

- InternVL 2.0

  - Introduction [\[link\]](https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html)
  - Quick Start [\[link\]](https://internvl.readthedocs.io/en/latest/internvl2.0/quick_start.html)
  - Finetune [\[link\]](https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html)
  - Preference Optimization [\[link\]](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html)
  - Evaluation [\[link\]](https://internvl.readthedocs.io/en/latest/internvl2.0/evaluation.html)
  - Deployment [\[link\]](https://internvl.readthedocs.io/en/latest/internvl2.0/deployment.html)

- InternVL 1.5

  - Introduction [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.5/introduction.html)
  - Quick Start [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.5/quick_start.html)
  - Finetune [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.5/finetune.html)
  - Evaluation [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.5/evaluation.html)
  - Deployment [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.5/deployment.html)

- InternVL 1.2

  - Introduction [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.2/introduction.html)
  - Quick Start [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.2/quick_start.html)
  - Reproduce [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.2/reproduce.html)
  - Finetune [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.2/finetune.html)
  - Evaluation [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.2/evaluation.html)

- InternVL 1.1

  - Introduction [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.1/introduction.html)
  - Quick Start [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.1/quick_start.html)
  - Evaluation [\[link\]](https://internvl.readthedocs.io/en/latest/internvl1.1/evaluation.html)
