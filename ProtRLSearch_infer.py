# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bioreason.core.io import extract_dna_from_text
from bioreason.core.utils import set_verbose, eprint, str2bool
from bioreason.models.qwen_llm import QwenLLM  # 
from bioreason.rL.load_groundtruth import load_gold_by_query
from bioreason.rL.grpo import GRPOTrainer, GRPOConfig


os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
def load_json_or_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed line: {e}")
    return data

def main():
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--text_model_name", type=str, default=None)
    parser.add_argument("--model_family", type=str, default="Qwen3", choices=["Qwen2", "Qwen3"])
    parser.add_argument("--model_size", type=str, default="8B")

    parser.add_argument("--dna_encoder_name", type=str, default=None,
                        help="(Ignored in Qwen-only mode; kept for backward compatibility)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="(Deprecated; use --output_dir instead, kept for backward compatibility)")
    parser.add_argument(
        "--query_index",
        type=int,
        default=0,   
        help="Index of the query to load (default=0 for the first query; set None to load all)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--load_in_8bit", type=str2bool, default=False)
    parser.add_argument("--load_in_4bit", type=str2bool, default=True)
    parser.add_argument("--input_json", type=str, default="/home/linux/BioReason/dataset/bioreason_multimodal_gene_query.jsonl")
    parser.add_argument("--user_query", type=str, default=None)
    parser.add_argument("--dna_seq", type=str, default=None)
    parser.add_argument("--labels_json", type=str, default="dataset/bioreason_multimodal_gene_query.jsonl")
    parser.add_argument("--grpo_K", type=int, default=2)
    parser.add_argument("--grpo_lr", type=float, default=1e-5)
    parser.add_argument("--grpo_max_new_tokens", type=int, default=5000)
    parser.add_argument("--grpo_clip_eps", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="grpo_runs")
    parser.add_argument("--retriever_url", type=str, default="")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--max_new_tokens_step", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rm_use_lora", type=str2bool, default=False)
    parser.add_argument("--rm_stay_cpu", type=str2bool, default=False)
    parser.add_argument("--train_samples", type=int, default=10,
                    help="Number of dataset samples to use for training (default: 3)")
 
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--resume", type=str2bool, default=True, help="")

    args = parser.parse_args()

    set_verbose(bool(args.verbose))

    def _normalize_size(s: str) -> str:
        return s.strip().replace("b", "B")

    if not args.text_model_name:
        fam = args.model_family.strip()
        size = _normalize_size(args.model_size)
        args.text_model_name = (
            f"Qwen/Qwen2-{size}-Instruct" if fam == "Qwen2" else f"Qwen/Qwen3-{size}"
        )

   
    if args.input_json:
     
        query_index = getattr(args, "query_index", None)

        with open(args.input_json, "r", encoding="utf-8") as f:
            text = f.read().strip()

      
        try:
            
            data = json.loads(text)
            if isinstance(data, dict):
              
                data_list = [data]
            elif isinstance(data, list):
           
                data_list = data
            else:
                raise ValueError("Unsupported JSON structure.")
        except json.JSONDecodeError:
           
            data_list = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping malformed line: {e}")

      
        if query_index is not None:
        
            if query_index < 0 or query_index >= len(data_list):
                raise IndexError(f"query_index {query_index} out of range (total {len(data_list)}).")
            selected_data = [data_list[query_index]]
        else:
         
            max_n = min(args.train_samples, len(data_list))
            selected_data = data_list[:max_n]
       

        user_queries = []
        dna_seqs = []
        for item in selected_data:
            q = item.get("query", "").strip()
            if not q:
                continue
            user_queries.append(q)
            dna_seqs.append((args.dna_seq or "").strip() or extract_dna_from_text(q))

    elif args.user_query:
        user_queries = [args.user_query.strip()]
        dna_seqs = [(args.dna_seq or "").strip() or extract_dna_from_text(args.user_query)]
    else:
        raise ValueError("Please provide --input_json or --user_query")


    model = QwenLLM(
        model_name=args.text_model_name,
        device=args.device or "cuda",
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        enable_lora=True,
        r=16,
        alpha=32,
        dropout=0.05,
    )


    if args.input_json:
        input_path = args.input_json
       

     
        data_list = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
               
                data_list.append(json.loads(line))
                
               

     


        if args.query_index is not None:
           
            if args.query_index < 0 or args.query_index >= len(data_list):
                raise IndexError(f"")
            selected_data = [data_list[args.query_index]]
          
        else:
         
            max_n = min(args.train_samples, len(data_list))
            selected_data = data_list[:max_n]
            eprint(f"")

    
        user_queries = [item.get("query", "").strip() for item in selected_data if item.get("query")]

    
        dna_seqs = []
        for item in selected_data:
            q = item.get("query", "").strip()
            if not q:
                continue
            if args.dna_seq:
                dna_seqs.append(args.dna_seq.strip())
            else:
            
                dna_seqs.append(extract_dna_from_text(q))

    elif args.user_query:
   
        user_queries = [args.user_query.strip()]
        dna_seqs = [(args.dna_seq or "").strip() or extract_dna_from_text(args.user_query)]
     
    else:
        raise ValueError("Please provide either --input_json or --user_query")


   
    samples = list(zip(user_queries, dna_seqs))
    if not samples:
        raise ValueError("")




   
  
    try:
        rm_tokenizer = AutoTokenizer.from_pretrained(
            "weqweasdas/RM-Gemma-2B",
            cache_dir="/home/linux/BioReason/hf_cache",
            padding_side="right"
        )

        rm_model = AutoModelForSequenceClassification.from_pretrained(
            "weqweasdas/RM-Gemma-2B",
            cache_dir="/home/linux/BioReason/hf_cache",
            device_map="auto",
            load_in_8bit=True,
        ).eval()

        for p in rm_model.parameters():
            p.requires_grad = False

        if args.rm_use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            rm_model = prepare_model_for_kbit_training(rm_model)
            lora_cfg = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                task_type="SEQ_CLS",
            )
            rm_model = get_peft_model(rm_model, lora_cfg)
 

    except Exception as e:
        raise RuntimeError(f"[INIT] {e}")


    cfg = GRPOConfig(
        lr=args.grpo_lr,
        num_return_sequences=args.grpo_K,
        max_new_tokens=args.grpo_max_new_tokens,
        temperature=args.temperature,
        ppo_clip_eps=args.grpo_clip_eps,
        beta=0.1,
    )



    lora_params = [p for n, p in model.text_model.named_parameters() if "lora" in n.lower()]
    if not lora_params:
        lora_params = [p for p in model.text_model.parameters() if p.requires_grad]

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(lora_params, lr=cfg.lr)

    except Exception:
        optimizer = torch.optim.AdamW(lora_params, lr=cfg.lr)
 

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.jsonl")

 
    if args.resume:
        ckpts = sorted(
            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint_step_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if ckpts:
            latest_ckpt = os.path.join(args.output_dir, ckpts[-1])
            eprint(f"")
            trainer.model.text_model.load_adapter(latest_ckpt) if hasattr(trainer.model.text_model, "load_adapter") \
                else trainer.model.text_model.from_pretrained(latest_ckpt)
        else:
            eprint("")


    all_reports: List[Dict[str, Any]] = []
    best_reward = float("-inf")
 
    trainer = GRPOTrainer(
        model,
        optimizer,
        model.text_tokenizer,
        model.device,
        cfg,
        rm_model=rm_model,
        rm_tokenizer=rm_tokenizer,
    )

   
    all_reports: List[Dict[str, Any]] = []

    for i, (user_query, dna_seq) in enumerate(samples):
        eprint(f"\n[ROUND START] === SAMPLE {i+1}/{len(samples)} ===")
        eprint(f"[QUERY] {user_query[:200]}{'...' if len(user_query) > 200 else ''}")

     
        trainer.current_query = user_query 
        trainer.current_dna_seq = dna_seq

        try:
            gold_json = load_gold_by_query(args.labels_json, user_query)
        except Exception as e:
            eprint(f"")
            gold_json = {}

        
        try:
            trainer._cached_inputs = model.text_tokenizer(user_query, return_tensors="pt").to(model.device)
        except Exception as e:
            eprint(f"")

    
        for epoch in range(args.epochs):
            report = trainer.train_once(
                user_query=user_query,
                dna_seq=dna_seq,
                gold_json=gold_json,
                router=getattr(trainer, "router", None),
            )
            all_reports.append(report)
      
            eprint(f"[GRPO][{i+1}/{len(samples)}][Epoch {epoch+1}/{args.epochs}] "
                f"loss={report.get('loss', 0):.6f} | reward={report.get('reward', 0):.3f}")

      
        if i % args.save_interval == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_step_{i}")
            os.makedirs(ckpt_dir, exist_ok=True)
            trainer.model.save_pretrained(ckpt_dir)
            trainer.tokenizer.save_pretrained(ckpt_dir)
            eprint(f"")

      
        cur_reward = report.get("reward", 0)
        if cur_reward > best_reward:
            best_reward = cur_reward
            best_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            trainer.model.save_pretrained(best_dir)
            trainer.tokenizer.save_pretrained(best_dir)
            eprint(f"[BEST] 🌟 New best reward={best_reward:.3f} saved to {best_dir}")

      
        if args.resume:
            best_dir = os.path.join(args.output_dir, "best_model")
            if os.path.exists(best_dir):
                try:
                    
                    from peft import PeftModel
                    model.text_model = PeftModel.from_pretrained(model.text_model, best_dir)
                    eprint("[RESUME]")
                except Exception as e:
                    eprint(f"")

 
 
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    trainer.tokenizer.save_pretrained(final_dir)
    eprint(f"[FINAL SAVE]")



    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "train_outputs.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"reports": all_reports}, f, ensure_ascii=False, indent=2)




if __name__ == "__main__":
    main()
