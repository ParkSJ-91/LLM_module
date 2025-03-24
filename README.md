# LLM Sample Repository

This repository provides sample code for creating and training large language models (LLMs) in PyTorch. The code shows how to:

1. Use **BERT-style** or **GPT-style** Transformer architectures (with a choice of **flash attention** or **regular attention**).
2. Train or fine-tune these models using **Fully Sharded Data Parallel (FSDP)** in PyTorch.
3. Extend or adapt the code to handle your own custom data pipeline.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/LLM-sample.git
   cd LLM-sample
2. Install dependencies:
   
   cuda                         11.8\
   python                       3.9.0\
   pytorch                      2.0.1 (py3.9_cuda11.8_cudnn8.7.0_0)\
   torchdata                    0.6.1\
   torchtext                    0.15.2\
   transformers                 4.42.3\
   triton                       2.1.0\
   einops                       0.8.0\
   flash-attn                   2.5.9.post1\
   numpy                        1.26.4

## Getting Started
```bash
import torch
from LLM_sample_module import TransformerLM

# Example: GPT-style with flash attention
model = TransformerLM(
    ntoken=32000,       # vocabulary size
    d_model=512,        # hidden dimension
    nhead=8,            # number of attention heads
    num_layers=6,       # number of Transformer layers
    dim_feedforward=2048,
    dropout=0.1,
    attention_dropout=0.1,
    flash=True,         # enable flash attention
    batch_first=True,
    norm_first=True,
    max_seq_len=1024,
    causal=True         # GPT-style causal
)

# Dummy input (batch_size=2, seq_len=10)
dummy_input = torch.randint(low=1, high=32000, size=(2, 10), dtype=torch.long)

# Forward pass
output = model(dummy_input)
print("Output shape:", output.shape)  # [2, 10, vocab_size]
```
## Training with FSDP
The file LLM_sample_run.py contains a reference script for training:
1. **Distributed initialization**\
   It uses torch.distributed.init_process_group(backend='nccl') for multi-GPU setups.
2. **Model construction & wrapping**\
   Creates a GPT-style TransformerLM and wraps it with FullyShardedDataParallel.
3. **Mixed precision & gradient scaling**\
   Showcases how to use autocast and GradScaler for performance optimization.
4. **Optimizer & Scheduler**\
   Uses AdamW and OneCycleLR, with gradient clipping.
   
**Important**: The script references a function load_data() which is not implemented. You must replace or define this function with your own data loading pipeline, returning batches of input tensors.

## Data Loading
Since data sources vary widely between projects, **this repository does not include a built-in data pipeline**. To integrate your data:

1. Tokenize your data (e.g., with Hugging Face Tokenizers).
2. Convert text into token IDs and arrange into tensors of shape (batch_size, seq_len).
3. Feed them to the model in BERT style (with segment labels) or GPT style (causal).\
You can adapt the placeholder in LLM_sample_run.py or create your own.

## Example command
Once you have your data loader implemented, you could run (for example):
```bash
torchrun --nproc_per_node=4 LLM_sample_run.py \
  --epochs 10 \
  --learning_rate 3e-4 \
  --length 1024 \
  --num_layers 6 \
  --d_model 512 \
  --num_heads 8 \
  --dropout_rate 0.1 \
  --local_batch_size 4
```
- --nproc_per_node=4 means 4 GPUs on the node.
- Adjust arguments based on your system and experiment needs.

## Contributing
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit and push your changes.
4. Open a Pull Request and describe your changes in detail.

## Contact
If you have questions, suggestions, or want to propose changes, please open an issue or a pull request. Contributions to improve the code, especially around data loading, performance, or additional model variants, are always welcome!
