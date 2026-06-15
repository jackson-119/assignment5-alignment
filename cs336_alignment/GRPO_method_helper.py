from importlib import metadata
from urllib import response

from drgrpo_grader import r1_zero_reward_fn
import torch
from collections.abc import Callable
from typing import Literal

def compute_group_normalized_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool,
        )-> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    '''Compute rewards for each group of rollout responses, normalized by the group size.
    Args:
    reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
    the ground truths, producing a dict with keys"reward", "format_reward", and "answer_reward".
    rollout_responses: list[str] Rollouts from the policy. The length of this list is
    rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    repeated_ground_truths: list[str] The ground truths for the examples. The length of this
    list is rollout_batch_size, because the ground truth for each example is repeated
    group_size times.
    group_size: int Number of responses per question (group).
    advantage_eps: float Small constant to avoid division by zero in normalization.
    normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
    subtract only the group mean.
    
    Returns:
    tuple[torch.Tensor, torch.Tensor, dict[str, float]].
    advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
    response.
    raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
    response.
    metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards)'''
    
    reward_list = [
                reward_fn(r, gt)["reward"]
                for r, gt in zip(rollout_responses, repeated_ground_truths)
                ]

    raw_rewards = torch.tensor(reward_list)
    
    n_prompts = len(rollout_responses) // group_size
    reward_2d = raw_rewards.view(n_prompts, group_size)
    
    mean = reward_2d.mean(dim=-1, keepdim=True)
    std = reward_2d.std(dim=-1, keepdim=True, unbiased=False)
    
    advantages = reward_2d - mean
    if normalize_by_std:
        advantages /= (std + advantage_eps)
    
    advantages = advantages.flatten()
    
    metadata = {
        "raw_reward_mean": raw_rewards.mean().item(),
        "raw_reward_std": raw_rewards.std().item(),
        "raw_reward_max": raw_rewards.max().item(),
        "raw_reward_min": raw_rewards.min().item(),
        "group_std_mean": std.mean().item(),
    }
    
            
    return (advantages, raw_rewards, metadata)
    
    
def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        ) -> torch.Tensor:
    '''Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    Args:
    raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
    reward/advantage for each rollout response.
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
    each token.
    
    Returns:
    torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
    be aggregated across the batch and sequence dimensions in the training loop).'''
    
    return - policy_log_probs * raw_rewards_or_advantages.unsqueeze(-1) 

def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''Args:
    advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
    probs from the policy being trained.
    old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
    from the old policy.
    cliprange: float Clip parameter ϵ (e.g. 0.2).
    
    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
    loss.
    metadata dict containing whatever you want to log. We suggest logging whether each
    token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
    the min was lower than the LHS.'''
    
    ratio = torch.exp(policy_log_probs - old_log_probs)   
    advantages = advantages.unsqueeze(-1)

    surr1 = ratio * advantages
    #约束上下限               
    surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages


    loss_per_token = -torch.min(surr1, surr2)


    clipped = surr2 < surr1
    clip_ratio = clipped.float().mean()

    metadata = {
        "clip_ratio": clip_ratio,
        "mean_ratio": ratio.mean(),
        "mean_loss": loss_per_token.mean(),
    }

    return loss_per_token, metadata


def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None= None,
        advantages: torch.Tensor | None= None,
        old_log_probs: torch.Tensor | None= None,
        cliprange: float | None= None,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''Select and compute the desired policy-gradient loss.
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
    policy being trained.
    loss_type One of "no_baseline","reinforce_with_baseline", or "grpo_clip".
    raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
    advantages Required for"reinforce_with_baseline" and "grpo_clip"; shape
    (batch_size, 1).
    old_log_probs Required for"grpo_clip"; shape (batch_size, sequence_length).
    cliprange Required for"grpo_clip"; scalar ϵ used for clipping.
    
    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss (batch_size, sequence_length), per-token loss.
    metadata dict, statistics from the underlying routine 
    (e.g., clip fraction for GRPO-Clip)'''
    
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {"mean_loss": loss.mean()}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {"mean_loss": loss.mean()}
    else:
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        
    return loss, metadata

def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None= None,
        ) -> torch.Tensor:
    '''Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.
    Args:
    tensor: torch.Tensor The data to be averaged.
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
    dim: int | None Dimension over which to average. If None, compute the mean over all
    masked elements.
    
    Returns:
    torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.'''
    
    num = (tensor * mask).sum(dim=dim)

    den = mask.sum(dim=dim)

    return num / den.clamp(min=1) 


def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None= None,
        advantages: torch.Tensor | None= None,
        old_log_probs: torch.Tensor | None= None,
        cliprange: float | None= None,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''Execute a forward-and-backward pass on a microbatch.
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
    policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
    prompt/padding.
    gradient_accumulation_steps Number of microbatches per optimizer step.
    loss_type One of "no_baseline","reinforce_with_baseline", "grpo_clip".
    raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
    advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
    old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
    cliprange Clip parameter ϵ for GRPO-Clip.
    
    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
    this so we can log it.
    metadata Dict with metadata from the underlying loss call, and any other statistics you
    might want to log.'''

    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, 
                                                  raw_rewards, advantages, old_log_probs,
                                                  cliprange)
    
    scalar_loss = masked_mean(loss, response_mask)
    metadata["scalar_loss_before_grad_accum"] = scalar_loss.detach()

    if gradient_accumulation_steps > 1 :
        scalar_loss /= gradient_accumulation_steps
    
    scalar_loss.backward()
    
    return scalar_loss.detach(), metadata

