import random
import token
from urllib import response
from xml.etree.ElementInclude import include

from click import progressbar
from regex import T
from torch.optim import AdamW
import json
from transformers import PreTrainedModel,AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
import tqdm
from unittest.mock import patch
from SFT_mehtod_helper import masked_normalize,  tokenize_prompt_and_output
from SFT_mehtod_helper import sft_microbatch_train_step, get_response_log_probs
from SFT_mehtod_helper import log_generations
import os
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn



def init_vllm(model_id: str, device: str, seed: int, 
              gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
 
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
        )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            )
    
    
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):

    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def get_batch(tokenized_data, batch_size, device):
    """
    从预处理的数据中采样一个Batch
    """
    total_len = tokenized_data["input_ids"]
    batch_indices = random.sample(range(total_len), batch_size)
    
    return {
        "input_ids": tokenized_data["input_ids"][batch_indices].to(device),
        "labels": tokenized_data["labels"][batch_indices].to(device),
        "response_mask": tokenized_data["response_mask"][batch_indices].to(device)
    }
    
    
def run_sft_experiment():

    logical_batch_size = 16  
    micro_batch_size = 1
    train_datasize = 128
    max_steps = 200
    seed = 42
    max_tokens = 1024
    gradient_accumulation_steps = logical_batch_size // micro_batch_size
    train_device = torch.device("cuda:0")
    eval_device = torch.device("cuda:1")
    lr = 2e-5
    eval_every_steps = 20
    
    wandb.init(project="cs336_sft", 
               name="sft_batch" + logical_batch_size + "_step" + max_steps, 
               config=
               {"logical_batch_size":logical_batch_size,
                "micro_batch_size": micro_batch_size,
                "learning_rate": lr,
                "max_steps": max_steps})
    
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
    ).to(eval_device)
    #开启梯度检查点，优化显存
    policy.gradient_checkpointing_enable()
    
    optimizer = AdamW(policy.parameters(), lr=lr)
    
    
    print("Initializing vllm")
    vllm_inst = init_vllm("Qwen/Qwen2.5-Math-1.5B", train_device, seed=seed)
    
    print("Loading training data")
    raw_train_data = []
    

    
    train_data_path = os.path.join(curdir, ".", "data/gsm8k/train.jsonl")
    with open(train_data_path, "r") as f:
        for line in f:
            raw_train_data.append(json.loads(line))
            
    #过滤
    print("Flitering correct examples...")
    raw_train_data = [item for item in raw_train_data if item.get("is_correct", True)]
            
    if train_datasize < len(raw_train_data):
        raw_train_data = random.sample(raw_train_data, train_datasize)

    print("Pre_tokenizing training dataset")
    tokenized_train_data = tokenize_prompt_and_output(
        prompt_strs=[item["prompt"] for item in raw_train_data],
        output_strs=[item["response"] for item in raw_train_data],
        tokenizer=tokenizer
    )
    
    
    print("Loading validation data")
    val_prompts= []
    val_ground_truth = []
    
    val_path = os.path.join(curdir, ".", "data/gsm8k/test.jsonl") 
    with open(val_path, "r") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            raw_a = item["answer"]
            gold = raw_a.split("###")[-1].strip() if "###" in raw_a else raw_a.strip()
            formatted_prompt = r1_template.replace("{questions}", item["question"])
            val_prompts.append(formatted_prompt)
            val_ground_truth.append(gold)
            
    eval_sampling_params = SamplingParams(
        temperatrue = 0.0,
        max_tokens= max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
        
        
    progress_bar = tqdm(range(max_steps), desc= "SFT_steps")
    
    print(f"/n[Step0] Starting Evaluation...")
    policy.eval()
    load_policy_into_vllm_instance(policy, vllm_inst)
    
    metrics = log_generations(
        vllm_model = vllm_inst,
        sampling_params = eval_sampling_params,
        prompts = val_prompts,
        ground_truth=val_ground_truth,
        reward_fn = r1_zero_reward_fn,
        step = 0,
        log_prefix = "eval"
    )
    
    print(f"Eval Accuracy: {metrics.get('eval/accuracy', 0):.2f}")
    policy.train()
    
    for step in range(max_steps):
        
        accumulated_loss= 0.0
        accumulated_entropy = 0.0
        accumulated_res_entropy = 0.0
        
        for _ in range(gradient_accumulation_steps):
            batch = get_batch(tokenized_train_data, micro_batch_size, train_device)
            
            response_outputs = get_response_log_probs(policy, input_ids=batch["input_ids"],
                                                      labels=batch["labels"],
                                                      return_token_entropy=True)
            
            log_probs = response_outputs["log_probs"]
            token_entropy = response_outputs["token_entropy"]
            with torch.no_grad:
                valid_token_mask = (batch["labels"] != tokenizer.pad_token_id)
                current_res_mask = batch["response_mask"].bool & valid_token_mask
                
                avg_res_entropy = token_entropy[current_res_mask].mean().item() if current_res_mask.any() else 0.0
                
                avg_global_entropy = token_entropy[valid_token_mask].mean().item()
                
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=batch["response_mask"],
                gradient_accumulation_steps=gradient_accumulation_steps)
            
            
            accumulated_loss += loss.item() * gradient_accumulation_steps
            accumulated_entropy += avg_global_entropy
            accumulated_res_entropy += avg_res_entropy
            
        
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        
        wandb.log({
            "train/loss": accumulated_loss / gradient_accumulation_steps,
            "train/global_entropy": accumulated_entropy/ gradient_accumulation_steps,
            "train/response_entropy": accumulated_res_entropy/ gradient_accumulation_steps,
            "train/step": step + 1
        })    
        
        if (step + 1) % eval_every_steps == 0:
            print(f"[Step{step + 1}] Starting Evaluation...")
            policy.eval()
            load_policy_into_vllm_instance(policy, vllm_inst)
    
            metrics = log_generations(
                vllm_model = vllm_inst,
                sampling_params = eval_sampling_params,
                prompts = val_prompts,
                ground_truth=val_ground_truth,
                reward_fn = r1_zero_reward_fn,
                step = step + 1,
                log_prefix = "eval"
            )
            
            print(f"Eval Accuracy: {metrics.get('eval/accuracy', 0):.2f}")
            policy.train()
            
    
    print("Training finished. Saving Model...")
    save_path = os.path.join(curdir, "model/sft")
    os.makedirs(save_path, exist_ok=True)
    policy.save_pretrained(save_path)
    wandb.finish()