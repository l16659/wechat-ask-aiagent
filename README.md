
# Detailed Guide: Fine-Tuning DeepSeek Models with Unsloth + QLoRA, Distillation, and Ollama Deployment

This is a comprehensive, beginner-friendly pipeline for fine-tuning DeepSeek models using **Unsloth + 4bit/8bit QLoRA**, optional distillation, and deployment via **Ollama** with dynamic LoRA loading.  
All custom models/folders/outputs will use the prefix **`polo-super`**.

**Assumptions**  
- Hardware: 1–2 NVIDIA GPUs (e.g., RTX 4090/5090, ≥24GB VRAM total)  
- CUDA 12.4+ installed (verify with `nvidia-smi`)  
- Basic Python knowledge  
- Goal: Fine-tune a small DeepSeek distilled model (1.5B–8B), save LoRA, deploy locally  
- Total first-run time: **1–4 hours** (mostly downloading & training); later iterations much faster

## 1. Environment Setup (10–30 minutes)

**Purpose**: Install Unsloth and dependencies for accelerated QLoRA fine-tuning.

1. Create virtual environment (recommended)  
   ```bash
   python -m venv polo-super-env
   # Activate:
   # Linux/Mac:   source polo-super-env/bin/activate
   # Windows:     polo-super-env\Scripts\activate
   ```

## Install packages

# PyTorch with CUDA (adjust for your CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Unsloth + ecosystem
pip install "unsloth[cu124-torch240]" --no-deps
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes datasets

# qucik veryification
```bash
import torch
print(torch.cuda.is_available())          # Should be True
print(torch.cuda.device_count())          # Should show your GPU count
```

# Select & Load Base Model (5–20 minutes)
Purpose: Load a quantized DeepSeek variant to save VRAM.
Recommended starting models (from Hugging Face):

+ unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit ← fastest & lowest VRAM
+ unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit ← stronger
+ deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct ← code specialized

Code snippet (add to polo-super-finetune.py):
```bash
from unsloth import FastLanguageModel
import torch

max_seq_length = 8192
dtype = None
load_in_4bit = True   # Set False → use 8bit if you have more VRAM

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

# Prepare Dataset (20–60 minutes)
Purpose: High-quality data is 70–80% of success.
Format recommendation (Alpaca / ShareGPT style – especially good for reasoning):
```bash
{
  "messages": [
    {"role": "user", "content": "Solve: 15 × 23?"},
    {"role": "assistant", "content": "<think>Break it down: 15×20=300, 15×3=45, total 345.</think>\nFinal answer: 345"}
  ]
}
```
Steps:

Collect 1k–10k+ samples (HuggingFace Datasets, self-generated, synthetic via API, etc.)
Clean: remove noise, duplicates, wrong labels
Save as JSONL or use HF dataset

Load in code:
```bash
from datasets import load_dataset

dataset = load_dataset("json", data_files="polo-super-data.jsonl", split="train")
# Or: dataset = load_dataset("your_username/polo-super-dataset", split="train")
```

4. LoRA Fine-Tuning (30–120 minutes)
Purpose: Train small LoRA adapter only.

```bash
# 4.1 Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,                     # 16–128; higher = stronger but more VRAM
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # Very important for VRAM
    random_state = 3407,
)

# 4.2 Trainer
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",           # or "messages" if using chat template
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True,                        # Packs for speed
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 120,                   # or num_train_epochs = 3
        learning_rate = 2e-4,              # Most important to tune
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "polo-super-outputs",
    ),
)

# 4.3 Start training
trainer.train()
```

Iteration tips:

Improve/clean data first (biggest gain)
Then tune: learning_rate, max_steps/epochs, r
Rarely: change target_modules or lora_alpha

# Save & Merge (5–10 minutes)
```bash
# Save small LoRA adapter (recommended – very portable)
model.save_pretrained("polo-super-lora")

# Optional: Merge into full model (faster inference, bigger file)
model.save_pretrained_merged(
    "polo-super-merged",
    tokenizer,
    save_method = "merged_16bit"   # or "merged_4bit" for smaller size
)
```

# Deploy with Ollama + Dynamic LoRA Loading (10–30 minutes)
Purpose: Run your fine-tuned LoRA on top of base model locally.

Install Ollama → https://ollama.com/download
Verify: ollama --version
(Optional but recommended) Convert LoRA to GGUF
Use llama.cpp tools:
"python convert_lora_to_gguf.py --outfile polo-super-lora.gguf polo-super-lora/"
Create Modelfile (text file, no extension)
```bash
FROM llama3.1:8b                  # Must match your training base exactly!
ADAPTER ./polo-super-lora.gguf    # or path to Safetensors folder

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id>

{{ .Response }}<|eot_id|>"""

SYSTEM "You are Polo-Super, a helpful and precise assistant."
PARAMETER temperature 0.7
```
Create & RunBashollama create polo-super-model -f Modelfile
```ollama run polo-super-model```
