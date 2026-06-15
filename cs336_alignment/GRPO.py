
import torch
import wandb
from GRPO_method_helper import grpo_microbatch_train_step, compute_group_normalized_rewards
import os
from typing import Literal
from torch.optim import AdamW
import json
from transformers import PreTrainedModel,AutoModelForCausalLM, AutoTokenizer
import tqdm
from SFT_mehtod_helper import tokenize_prompt_and_output, get_response_log_probs
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
from SFT import init_vllm, load_policy_into_vllm_instance
import random


def run_GRPO_expriment():
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = 256 # On-policy
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    
    use_std_normalization: bool = True

    
    assert train_batch_size % gradient_accumulation_steps == 0, (
    "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    
    assert rollout_batch_size % group_size == 0, (
    "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    assert train_batch_size >= group_size, (
    "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    
    device = torch.device("cuda:0")



    wandb.init(project="cs336_grpo", 
               name="grpo_batch" + train_batch_size + "_step" + n_grpo_steps, 
               config=
               {"train_batch_size":train_batch_size,
                "micro_train_batch_size": micro_train_batch_size,
                "learning_rate": learning_rate,
                "n_grpo_steps": n_grpo_steps})
    
    curdir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(curdir, "prompts/r1_zero.prompt")
    
    #加载模版
    with open(prompt_path, "r") as f:
        r1_template = f.read().strip()
        
    #初始化模型与分词器
    print("Initializing Model")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        torch_dtype = torch.bfloat16,
        low_cpu_men_usage = True,
        attn_implementation = "flash_attention_2"
    ).to(device)
    #开启梯度检查点，优化显存
    policy.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    
    sampling_params = SamplingParams(
        temperatrue = sampling_temperature,
        max_tokens= sampling_max_tokens,
        min_tokens= sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    print("Initializing vllm")
    seed = 42
    vllm_inst = init_vllm("Qwen/Qwen2.5-Math-1.5B", device, seed=seed, 
                          gpu_memory_utilization=gpu_memory_utilization)
    
    print("Loading training data")
    raw_train_data = []
    
    train_data_path = os.path.join(curdir, ".", "data/gsm8k/train.jsonl")
    with open(train_data_path, "r") as f:
        for line in f:
            raw_train_data.append(json.loads(line))
            
    #过滤
    print("Flitering correct examples...")
    raw_train_data = [item for item in raw_train_data if item.get("is_correct", True)]
                  
        
    progress_bar = tqdm(range(n_grpo_steps), desc= "grpo_steps")
    
    print(f"/n[Step0] Starting Evaluation...")

    
    for step in range(n_grpo_steps):
        
        indices = random.sample(range(len(raw_train_data)), n_prompts_per_rollout_batch)
        batch_prompts = [raw_train_data[i]["prompt"] for i in indices]
        batch_answers = [raw_train_data[i]["response"] for i in indices]
        
        formatted_prompts = [r1_template.format(question=q) for q in batch_prompts]
        # 重复 group_size 次
        repeated_prompts = [p for p in formatted_prompts for _ in range(group_size)]
        repeated_answers = [a for a in batch_answers for _ in range(group_size)]
        
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_inst)
        
        with torch.no_grad():
            outputs = policy.generate(repeated_prompts, sampling_params)
            responses = [o.outputs[0].text for o in outputs]
            
            advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
                                                r1_zero_reward_fn, responses, 
                                                repeated_answers, group_size, 
                                                advantage_eps, use_std_normalization)
            
            tokenized = tokenize_prompt_and_output(repeated_prompts, repeated_answers, tokenizer)
            
            input_ids = torch.tensor(tokenized["input_ids"], device=device)
            labels = torch.tensor(tokenized["labels"], device=device)
            response_mask = torch.tensor(tokenized["response_mask"], device=device)
            
            
            old_log_probs = get_response_log_probs(
                policy,
                input_ids,
                labels,
            )["log_probs"]
                
        optimizer.zero_grad()
        
        advantages = advantages.to(device)
            
        for start in range(0, rollout_batch_size, micro_train_batch_size):
            end = start + micro_train_batch_size
            
            micro_input_ids = input_ids[start:end]
            micro_labels = labels[start:end]
            micro_mask= response_mask[start:end]
            
            micro_advantages = advantages[start:end].unsqueeze(-1)
            
            policy_log_probs = get_response_log_probs(
                policy,
                micro_input_ids,
                micro_labels,
            )["log_probs"]
            
            scalar_loss, metadata = grpo_microbatch_train_step(policy_log_probs,
                                                               micro_mask,
                                                               gradient_accumulation_steps,
                                                               loss_type,
                                                               raw_rewards[start:end].unsqueeze(-1).to(device),
                                                               micro_advantages,
                                                               old_log_probs,
                                                               0.2)
            
            optimizer.step()
            
        wandb.log({
            "step": step,
            "loss": scalar_loss.item(),
            "reward_mean": reward_metadata["raw_reward_mean"],
            "reward_std": reward_metadata["raw_reward_std"],
        })

        print(
            f"step={step} "
            f"loss={scalar_loss.item():.4f} "
            f"reward={reward_metadata['raw_reward_mean']:.4f}"
        )

    wandb.finish()
            
        
        
    
    