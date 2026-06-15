from numpy import append
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
import yaml
import os


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], 
                               tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
    ''' Tokenize the prompt and output strings, and construct a mask that is 1 for the 
        response tokens and 0 for other tokens (prompt or padding).
        Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
        Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the
        following keys:
        input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        the tokenized prompt and output strings, with the final token sliced off.
        labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        shifted input ids, i.e., the input ids without the first token.
        response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
        1): a mask on the response tokens in the labels.  '''
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    all_inputs_ids = []
    all_labels = []
    all_masks = []
    
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        
        full_ids = prompt_ids + output_ids
        prompt_len = len(prompt_ids)
        total_len = len(full_ids)
        
        input_ids = full_ids[:-1]
        labels = full_ids[1:]
        
        mask = [1 if i >= prompt_len - 1 else 0 for i in range(total_len - 1)]
        
        all_inputs_ids.append(input_ids)
        all_labels.append(labels)
        all_masks.append(mask)
            
        max_len = max(len(ids) for ids in all_inputs_ids)
        pad_id = tokenizer.pad_token_id
        
        padded_inputs_id = []
        padded_labels = []
        padded_masks = []
        
        for input_ids, label, mask in zip(all_inputs_ids, all_labels, all_masks):
            pad_len = max_len - len(input_ids)
            padded_inputs_id.append(input_ids + [pad_id] * pad_len) #用结束符填充input_token
            padded_labels.append(label + [-100] * pad_len) #在计算损失时会跳过-100的位置
            padded_masks.append(mask + [0] * pad_len)
            
        return {
            "input_ids": padded_inputs_id,
            "labels": padded_labels,
            "response_mask": padded_masks
        }
          
          
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    '''Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
    logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
    containing unnormalized logits.
    Returns:
    torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
    prediction.  '''
    
    shifted = logits - logits.max(dim = -1, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    sum_exp = exp_shifted.sum(dim = -1, keepdim=True)
    p = exp_shifted / sum_exp
    log_p = shifted - torch.log(sum_exp)
    entropy = - torch.sum(p * log_p, dim = -1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    '''Args:
    model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
    and in inference mode if gradients should not be computed).
    input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
    response tokens as produced by your tokenization method.
    labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
    tokenization method.
    return_token_entropy: bool If True, also return per-token entropy by calling
    compute_entropy.
    Returns:
    dict[str, torch.Tensor].
    "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
    log pθ (xt |x<t).
    "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
    for each position (present only if return_token_entropy=True).'''
    
    logits = model(input_ids).logits
    log_p = torch.log_softmax(logits, dim = -1)
    log_probs = torch.gather(log_p, dim = -1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_probs[labels == -100] = 0.0
    
    res = {}
    res["log_probs"] = log_probs
    if return_token_entropy:
        entropy = compute_entropy(logits)
        res["token_entropy"] = entropy
        
        
    return res

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None= None,
    ) -> torch.Tensor:
    '''Sum over a dimension and normalize by a constant, considering only those elements where mask
    == 1.
    Args:
    tensor: torch.Tensor The tensor to sum and normalize.
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
    normalize_constant: float the constant to divide by for normalization.
    dim: int | None the dimension to sum along before normalization. If None, sum over all
    dimensions.
    Returns:
    torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to
    the sum.'''
    
    tensor_masked = tensor * mask
    
    summed_tensor = tensor_masked.sum(dim)
    
    res = summed_tensor/normalize_constant if normalize_constant else summed_tensor
    
    return res


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''Execute a forward-and-backward pass on a microbatch.
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
    SFT policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
    prompt/padding.
    gradient_accumulation_steps Number of microbatches per optimizer step.
    normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
    this so we can log it.
    metadata Dict with metadata from the underlying loss call, and any other statistics you
    might want to log.'''
    
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant)
    res_loss = loss
    if gradient_accumulation_steps > 1:
        loss /= gradient_accumulation_steps
    loss.backward()
    
    res = {}
    res["normalize_constant"] = normalize_constant
    
    return (res_loss, res)
    
        
import wandb
from typing import List, Dict, Any


def log_generations(
    vllm_model,
    sampling_params,
    prompts: List[str],
    ground_truth: List[str],
    reward_fn,
    step: int,
    log_prefix: str = "eval"
) -> Dict[str, Any]:
    """
    使用 vLLM 模型对给定 prompts 进行生成，计算奖励与准确率，并记录到 wandb。

    Args:
        vllm_model: vLLM 的 LLM 实例，用于生成。
        sampling_params: vLLM 的 SamplingParams。
        prompts: 待生成的 prompt 列表。
        ground_truth: 参考答案列表，与 prompts 一一对应。
        reward_fn: 奖励函数，签名为 reward_fn(generated_text, ground_truth) -> float/int。
        step: 当前步数，用于 wandb 记录。
        log_prefix: wandb 日志键的前缀（如 "eval"）。

    Returns:
        dict: 包含准确率等指标的字典。
    """
    # 批量生成
    outputs = vllm_model.generate(prompts, sampling_params)

    # 提取生成的文本（假设每个 prompt 只有一个输出序列）
    generated_texts = []
    for output in outputs:
        if output.outputs:
            generated_texts.append(output.outputs[0].text)
        else:
            generated_texts.append("")

    # 计算每个样本的奖励
    rewards = []
    for gen, gt in zip(generated_texts, ground_truth):
        score_dict = reward_fn(gen, gt)
        r = score_dict.get("score", 0)
        rewards.append(r)

    # 准确率（假设奖励为 0/1 二值）
    accuracy = sum(rewards) / len(rewards) if rewards else 0.0

    # 记录到 wandb
    wandb.log({f"{log_prefix}/accuracy": accuracy}, step=step)


    return {f"{log_prefix}/accuracy": accuracy}
        

