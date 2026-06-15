from posixpath import dirname

from vllm import LLM, SamplingParams
from collections.abc import Callable
from drgrpo_grader import r1_zero_reward_fn
import jsonlines
import os
import json
import yaml

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: SamplingParams
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    total_score = 0.0
    count = 0
    
    for prompt, output, answer in zip(prompts, outputs, answers):
        generated_text = output.outputs[0].text
        score_dict = reward_fn(generated_text, answer)
        score = score_dict.get("score", 0.0)  
        total_score += score
        count += 1
        
        results.append({
            "prompt": prompt,
            "generated": generated_text,
            "ref_answer": answer,
            "score": score,
            "detail": score_dict
        })
        
    avg_score = total_score / count if count > 0 else 0.0
    metrics = {
        "num_samples": count,
        "average_score": avg_score
    }
    print(f"评估指标: {metrics}")
    
    # 序列化到磁盘
    output_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(output_dir, "eval/results.yaml")
    metrics_path = os.path.join(output_dir, "eval/metrics.yaml")
    
    with open(result_path, "w", encoding="utf-8") as writer:
        yaml.dump(result_path, writer, sort_keys=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        yaml.dump(metrics, f, sort_keys=False)
    
    print(f"结果已保存至: {result_path}")
    print(f"指标已保存至: {metrics_path}")
        
        
    

def Zero_shot_Math_Baseline():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    que_path = os.path.join(cur_path, "./", "data/gsm8k/test.jsonl")
    prompt_path = os.path.join(cur_path, "prompts/r1_zero.prompt")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
            
    sampling_params = SamplingParams(
        temperature = 1.0, top_p = 1.0, max_tokens = 1024, stop = ["</answer>"]
    )
    sampling_params.include_stop_str_in_output = True
    llm = LLM(model = "Qwen/Qwen2.5-Math-1.5B-Instruct")
    
    
    
    prompts = []
    answers = []
    with open(que_path, "r", encoding="utf-8") as reader:
        for obj in reader:
            prompt = prompt_template.format(question = obj["question"])
            prompts.append(prompt)
            answer = obj["answer"]
            answers.append(answer)
            

    evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params)
    
    
    
    
    