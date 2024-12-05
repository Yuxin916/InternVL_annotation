# InternVL-Chat

This folder contains the implementation of the InternVL-Chat.


## Introduction
**InternVL2-2B** consists of:
- InternViT-300M-448px, 
- an MLP projector, and 
- internlm2-chat-1_8b


![InternVL2_2B_Model_Card](assets/model_card.png)
![InternVL2_2B_Model_Card2](assets/model_card2.png)

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

## Prepare Your Customized Evaluation Data



## LoRA Finetuning of Habitat Data
```aiignore
# run in internvl_chat folder
# Coco
# GPUS=1 PER_DEVICE_BATCH_SIZE=4 bash shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora_coco.sh
# habitat
GPUS=1 PER_DEVICE_BATCH_SIZE=4 bash shell/internvl2.0/habitat_2b_finetune/lora.sh
```

## Evaluate the Finetuned Model
### On Training Data

### On Evaluation Data
```bash
# on COCO Caption
GPUS=1 bash evaluate.sh pretrained/InternVL2-2B caption-coco --dynamic --auto
```

## Merging LoRA Weights
After evaluating the fine-tuned model, you may want to merge the LoRA weights back into the original InternVL2 model
```bash
# python tools/merge_lora.py <input_path> <output_path>
# COCO 
# python tools/merge_lora.py work_dirs/internvl_chat_v2_0/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora_coco/ work_dirs/internvl_chat_v2_0/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora_coco_merge
# Habitat
PYTHONPATH=$PYTHONPATH:$(pwd) python tools/merge_lora.py log/habitat/lora_annots_1/ log_merged/habitat/lora_annots_1
```


## Wrapping into AutoModel for easier inference or deployment
```bash
# copy all the Python scripts from the original InternVL2-2B directory to the new merged model directory
cp pretrained/InternVL2-2B/*.py log_merged/habitat/lora_annots_1
# copy the config.json file from the original InternVL2-2B directory to the new merged model directory
cp pretrained/InternVL2-2B/config.json log_merged/habitat/lora_annots_1
```
