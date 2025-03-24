import os
import functools 
import argparse

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy, enable_wrap, wrap

from LLM_sample_module import *

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--smoothing_factor', type=float, default=0.0)
    parser.add_argument('--local_batch_size', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=4000)
    parser.add_argument('--decay_step', type=int, default=1000000)
    parser.add_argument('--multi_gpu', type=bool, default=True)
    parser.add_argument('--floating_points', type=str, default='bfloat16')

    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--global_workers', type=int, default=80)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--world_size', type=int, default=0)
    return parser
    
def setup(opts):
    opts.global_rank = int(os.environ["RANK"])
    opts.local_rank = int(os.environ["LOCAL_RANK"])
    opts.world_size = int(os.environ["WORLD_SIZE"])
    opts.global_batch_size = int(opts.local_batch_size * opts.world_size)
    opts.local_workers = int(opts.global_workers / opts.world_size)
    
    torch.cuda.set_device(opts.local_rank)

    torch.distributed.init_process_group(backend='nccl')
                
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cleanup():
    torch.distributed.destroy_process_group()

def save_ckpt(state, outF):
    torch.save(state, outF)

def main_worker(opts):
    setup(opts)
    print(f"Rank {opts.local_rank}/{int(os.environ['WORLD_SIZE'])} process initialized")

    train_dataset, train_sampler, vocab = load_data()

    if opts.floating_points == 'bfloat16':
        all_dtype = torch.bfloat16
    else:
        all_dtype = torch.float16

    mixed_precision_policy = MixedPrecision(param_dtype=all_dtype, reduce_dtype=all_dtype, buffer_dtype=all_dtype)
    my_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={FlashDecoderLayer},)
    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP
    
    model = TransformerLM(
        ntoken = len(vocab.get_itos()),
        d_model = opts.d_model,
        nhead = opts.num_heads,
        dropout = opts.dropout_rate,
        attention_dropout = opts.dropout_rate,
        num_layers = opts.num_layers,
        dim_feedforward = opts.d_model*4,
        flash = True, 
        batch_first = True, 
        norm_first = True,
        max_seq_len = opts.length,
        causal = True
        )
    if opts.global_rank == 0: print(opts.global_rank, model.parameters())

    model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrap_policy,
            #cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=mixed_precision_policy,
            device_id=torch.cuda.current_device(),
            #use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,#.SHARD_GRAD_OP
            sync_module_states=True,
            forward_prefetch=True
            )

    optimizer = optim.AdamW(model.parameters(), lr=opts.learning_rate, betas=(0.9, 0.95), eps=1e-4, weight_decay=0.1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opts.learning_rate, pct_start=opts.warmup/opts.decay_step, total_steps=opts.decay_step, anneal_strategy='cos', div_factor=25, base_momentum=0.85, max_momentum=0.99, three_phase=False)
        
    if opts.floating_points == 'float16':
        scaler = GradScaler()
    else:
        scaler = GradScaler(enabled=False)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for epoch in epoch_range:
        train_sampler.set_epoch(epoch)
        batch_idx = 0
        for train_data in train_dataset:
            model.train()
            start.record()

            with autocast(dtype=all_dtype):
                output = model(train_data)
                loss = F.cross_entropy(output.transpose(1,2), tar_true, ignore_index=0, label_smoothing=opts.smoothing_factor, reduction='sum')

            if opts.floating_points == 'float16':
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            optimizer.zero_grad()
            end.record()
            torch.cuda.synchronize()

            if opts.global_rank == 0:
                print('\t'.join(map(str, ['Training', epoch, batch_idx+1, train_data.size(0), scheduler.get_last_lr()[0], loss.item(), start.elapsed_time(end)])), flush=True)
        batch_idx += 1

def main():
    parser = argparse.ArgumentParser('LLM training', parents=[get_args_parser()])
    opts = parser.parse_args()
    main_worker(opts)

if __name__=="__main__":
    main()
