# rl/grpo.py
import os
import re
from string import Template
import torch
from types import SimpleNamespace
from dataclasses import dataclass
from typing import List, Dict, Any
from bioreason.search_tools.ToolRouter import ToolRouter  # ✅ 引入
from bioreason.core.utils import eprint
from bioreason.core.infer_util import (
    generate_with_multi_hop,
    _single_round_step,
    consolidate_episode_text,
    
)
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
torch._dynamo.config.disable = True
from collections import defaultdict
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
@dataclass
class GRPOConfig:
    max_hops: int = 3
    max_new_tokens: int = 5000
    temperature: float = 1.0
    do_sample: bool = True
    top_p: float = 0.9
    num_return_sequences: int = 1

    lr: float = 5e-6
    grad_clip: float = 1.0
    accum_steps: int = 2
    ppo_clip_eps: float = 0.2
    beta: float = 0.1
    use_flash_attention: bool = True
    compile_model: bool = True

    ans_weight: float = 0.5
    kw_weight: float = 0.2
    tool_weight: float = 0.2
    fmt_weight: float = 0.1

    verbose: bool = True
    save_interval: int = 100
    log_interval: int = 10
    seed: int = 42

import copy

DAG_PROMPT = Template(r"""
You are BioReason Planner. Your task is to analyze the biomedical topic and decompose it into exactly 3 concise keyword phrases, each mapped to suitable retrieval tools.

# CRITICAL OUTPUT CONTRACT
- Return ARRAY ONLY, with exactly 3 objects inside square brackets [ and ].
- No prose, no headings, no explanations, no backticks, no code fences, no extra text before/after.
- Each object must contain all 3 tags in this exact order:
  <keyword>...</keyword>
  <tool>...</tool>
  <purpose>...</purpose>
- ASCII-only inside tag values: normalize Greek letters (e.g., kappa), unify dashes (-), remove diacritics.
- No newline characters inside tag values (single-line per tag).
- Valid tool names inside <tool>: pubmed, uniprot, alphafold, websearch.
- Up to 3 tools may be comma-separated inside <tool>.
- No trailing commas between objects.

# SPECIAL RULES
- If input mentions a gene/protein or a species, it MUST be extracted as a standalone keyword.
- UniProt tool can ONLY be assigned when the keyword is a gene/protein/complex symbol or a species (never for pathways or mechanisms).
- AlphaFold tool can ONLY be assigned if the keyword can resolve to a UniProt accession (protein/complex with structure/domain context).
- PubMed covers mechanisms, pathways, diseases, motifs, cell types, reviews.
- Websearch is only for protocols, tool docs, datasets, or resources outside PubMed.
- Each keyword must map to 1–3 tools, at least one tool per keyword.
- Tool extraction is mandatory and will be passed into the reward function.

# Task
Extract exactly 3 most relevant noun-phrase keywords (short phrases, no sentences).  
For each keyword, output all 3 tags (<keyword>, <tool>, <purpose>).

# Output schema (exactly 3 objects inside array)
[
  <keyword>first keyword phrase</keyword>
  <tool>corresponding tool(s)</tool>
  <purpose>explain why this tool is used</purpose>
,
  <keyword>second keyword phrase</keyword>
  <tool>corresponding tool(s)</tool>
  <purpose>explain why this tool is used</purpose>
,
  <keyword>third keyword phrase</keyword>
  <tool>corresponding tool(s)</tool>
  <purpose>explain why this tool is used</purpose>
]

# Input
<topic>$topic</topic>
<dna>$dna</dna>

# Output now:
""")
def safe_json_loads(text: str):
        """
        Robust JSON-safe loader for LLM outputs.
        Handles messy strings like those returned by DeepSeek, Qwen, etc.
        Returns a dict or list if successful, otherwise None.
        """
        if not text or not isinstance(text, str):
            return None


        text = text.strip()
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```json|^```|```$", "", text, flags=re.IGNORECASE).strip()


        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            text = match.group(1)

    
        try:
            return json.loads(text)
        except Exception:
       
            fixed = (
                text.replace("'", '"')
                    .replace("\n", " ")
                    .replace("\r", "")
            )
            fixed = re.sub(r",\s*([\]}])", r"\1", fixed)
            fixed = re.sub(r"</?end>", "", fixed)
            fixed = re.sub(r"\\+", "\\", fixed)
            try:
                return json.loads(fixed)
            except Exception as e:
                print(f"[WARN] safe_json_loads failed: {e}")
                print(f"[DEBUG] Raw text:\n{text[:500]}")  
                return None

class GRPOTrainer:
    def __init__(
        self,
        dnallm,
        optimizer,
        tokenizer,
        device,
        cfg: GRPOConfig,
        router: ToolRouter = None,
        rm_model=None,
        rm_tokenizer=None
    ):
        import copy
        from transformers.utils import is_peft_available
        if is_peft_available():
            try:
                from peft import is_peft_model
            except ImportError:
                from peft import PeftModel, PeftModelForCausalLM
                def is_peft_model(model):
                    return isinstance(model, (PeftModel, PeftModelForCausalLM))
        else:
            def is_peft_model(_): return False


    
        self.dnallm = dnallm
        self.cfg = cfg


        self.model = getattr(dnallm, "text_model", None)


        if callable(self.model):
            try:
                eprint("[FIX] Detected dnallm.text_model is callable → invoking to obtain model instance ...")
                self.model = self.model() 
            except Exception as e:
                eprint(f"[WARN] Failed to call dnallm.text_model(): {e}")


        if hasattr(self.model, "get_base_model"):
            try:
                base_model = self.model.get_base_model()
                if isinstance(base_model, torch.nn.Module):
                    self.model = base_model
                    eprint("[FIX] Extracted base_model from LoRA/Peft Qwen wrapper.")
            except Exception:
                pass


        if not isinstance(self.model, torch.nn.Module):
            eprint(f"[WARN] self.model is not nn.Module (got {type(self.model)}) ... trying dnallm.model ...")
            if hasattr(dnallm, "model") and isinstance(dnallm.model, torch.nn.Module):
                self.model = dnallm.model
                eprint("[FIX] Using dnallm.model as fallback.")
            elif isinstance(dnallm, torch.nn.Module):
                self.model = dnallm
                eprint("[FIX] dnallm itself is nn.Module, using directly.")
            else:
                raise TypeError(f"[FATAL] Expected dnallm.text_model or dnallm.model to be torch.nn.Module, "
                                f"but got {type(self.model)}")


        assert isinstance(self.model, torch.nn.Module), \
            f"[FATAL] model is not nn.Module (got {type(self.model)})"

        self.model.to(device)
        eprint(f"[INIT] Using model type: {type(self.model)} on device {device}")


        self.tokenizer = tokenizer
        self.opt = optimizer
        self.global_step = 0
        self.stage = "keyword"
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.is_main = True
        self._tag_collector = defaultdict(set)
        self.model.to(self.device)
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        try:
            self.model.generation_config.use_cache = False
        except Exception:
            pass
        if router is None:
            self.router = ToolRouter()
            eprint("[GRPOTrainer] Initialized internal ToolRouter.")
        else:
            self.router = router
            eprint("[GRPOTrainer] Using external ToolRouter.")


        n_trainable, n_frozen = 0, 0
        for name, p in self.model.named_parameters():
            if "lora" in name.lower():
                p.requires_grad = True
                n_trainable += p.numel()
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        if n_trainable == 0:
            eprint("[WARN] No LoRA parameters detected — all params initially frozen.")
            for name, p in self.model.named_parameters():
                p.requires_grad = True
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            eprint(f"[FIX] Enabled full fine-tuning mode ({n_trainable:,} trainable / {n_frozen:,} frozen).")



        self.beta = getattr(cfg, "beta", 0.1)

        if self.beta == 0.0:

            self.ref_model = None
      
        elif is_peft_model(self.model):
      
            self.ref_model = None
      
        else:
    
            self.ref_model = create_reference_model(self.model).to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
  


        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if rm_model is None or rm_tokenizer is None:
            eprint("[WARN] No Reward Model passed into GRPOTrainer. "
                   "You must assign rm_model/rm_tokenizer later.")
            self.rm_model = None
            self.rm_tokenizer = None
            self.rm_device = self.device
        else:

            try:
                _ = next(rm_model.parameters())
            except Exception:
                raise RuntimeError("[ERR] Reward Model has no parameters. "
                                   "Check if it was loaded with device_map='auto' or init_empty_weights. "
                                   "Reload with from_pretrained(..., device_map=None).")


            rm_device = next(rm_model.parameters()).device
            if self.device.type == "cuda" and rm_device.type == "cpu":
                rm_model = rm_model.to(self.device)


            from transformers.utils import is_peft_available
            is_lora_model = False
            if is_peft_available():
                try:
                    from peft import PeftModel, PeftModelForSequenceClassification
                    if isinstance(rm_model, (PeftModel, PeftModelForSequenceClassification)):
                        is_lora_model = True
                except ImportError:
                    pass


            n_trainable, n_frozen = 0, 0
            for name, p in rm_model.named_parameters():
                if "lora" in name.lower():
                    p.requires_grad = True
                    n_trainable += p.numel()
                else:
                    p.requires_grad = False
                    n_frozen += p.numel()

            if n_trainable == 0:
                eprint("[RewardModel]  No LoRA parameters found — freezing all parameters (inference-only).")
            else:
                eprint(f"[RewardModel]  LoRA fine-tuning enabled "
                       f"({n_trainable:,} trainable / {n_frozen:,} frozen parameters).")


            self.rm_model = rm_model.eval() if n_trainable == 0 else rm_model.train()
            self.rm_tokenizer = rm_tokenizer
            self.rm_device = next(self.rm_model.parameters()).device
            eprint(f"[GRPOTrainer] Using external Reward Model on {self.rm_device} "
                   f"({'LoRA' if is_lora_model else 'Standard'}) mode.")




    def _collect_tags(self, text: str):

        if not isinstance(text, str):
            return
        matches = re.findall(r"<([a-zA-Z0-9_]+)>(.*?)</\1>", text, flags=re.I | re.S)
        for tag, val in matches:
            clean_val = val.strip()
            if clean_val:
                self._tag_collector[tag.lower()].add(clean_val)

    def _finalize_tags(self) -> Dict[str, List[str]]:

        return {k: sorted(v) for k, v in self._tag_collector.items() if v}

    def _print_cuda_mem(self, tag=""):
        if torch.cuda.is_available() and "cuda" in str(self.device):
            props = torch.cuda.get_device_properties(self.device)
            total = props.total_memory / 1024**2  # MB
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            peak = torch.cuda.max_memory_allocated(self.device) / 1024**2
            free = total - reserved

            eprint(
                f"[MEM][rank={self.rank}][{tag}] "
                f"allocated={allocated:.2f} MB | reserved={reserved:.2f} MB | "
                f"free={free:.2f} MB | total={total:.2f} MB | peak={peak:.2f} MB"
            )
        else:
            eprint(f"[MEM][rank={self.rank}][{tag}] running on CPU, skip CUDA mem check")



   

    def _amp_context(self):

        model_obj = getattr(self, "model", None)
        if model_obj is None or not hasattr(model_obj, "parameters"):
            return nullcontext()

        try:
            pdtype = next(model_obj.parameters()).dtype
        except Exception:
            pdtype = torch.float16  # fallback

        try:
            # 
            if pdtype == torch.bfloat16:
                return torch.cuda.amp.autocast(dtype=torch.bfloat16)
            elif pdtype == torch.float16:
                return torch.cuda.amp.autocast(dtype=torch.float16)
            else:
                return torch.cuda.amp.autocast()
        except Exception:
            return nullcontext()



    def _extract_user_query(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        m = re.search(r"<query>(.*?)</query>", text, flags=re.S | re.I)
        if m:
            return m.group(1).strip()
        return text.strip()[:512]

    def _sum_logprob_current(self, prompt: str, gen_text: str, detach: bool = False) -> torch.Tensor:

        if not isinstance(prompt, str):
            prompt = str(prompt)
        if not isinstance(gen_text, str):
            gen_text = str(gen_text)

        full_text = prompt + gen_text
        enc = self.tokenizer(full_text, return_tensors="pt", truncation=True)
        full_ids = enc.input_ids.to(self.device)
        attn_mask = enc.attention_mask.to(self.device)

        max_len = 1024
        if full_ids.shape[1] > max_len:
            full_ids = full_ids[:, -max_len:]
            attn_mask = attn_mask[:, -max_len:]

        with self._amp_context():
            out = self.model(full_ids, attention_mask=attn_mask, use_cache=False)
            logits = out.logits[:, :-1, :]

        labels = full_ids[:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        lp = token_logprobs.sum(dim=-1).mean()
        if detach:
            lp = lp.detach()

  
        del out, logits, logprobs, token_logprobs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return lp

    def _sum_logprob_old(self, prompt: str, gen_text: str) -> torch.Tensor:

        if not isinstance(prompt, str):
            prompt = str(prompt)
        if not isinstance(gen_text, str):
            gen_text = str(gen_text)

        full_text = prompt + gen_text
        enc = self.tokenizer(full_text, return_tensors="pt", truncation=True)
        full_ids = enc.input_ids.to(self.device)
        attn_mask = enc.attention_mask.to(self.device)

        max_len = 1024
        if full_ids.shape[1] > max_len:
            full_ids = full_ids[:, -max_len:]
            attn_mask = attn_mask[:, -max_len:]

        with torch.no_grad(), self._amp_context():
            out = self.model(full_ids, attention_mask=attn_mask, use_cache=False)
            logits = out.logits[:, :-1, :]

            labels = full_ids[:, 1:]
            logprobs = torch.log_softmax(logits, dim=-1)
            token_logprobs = logprobs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

            lp = token_logprobs.sum(dim=-1).mean()

  
        del out, logits, logprobs, token_logprobs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return lp.detach()
    def _sum_logprob_batch(self, prompt: str, texts: List[str], model=None) -> torch.Tensor:

        model = model or self.model
        full_texts = [prompt + (t or "") for t in texts]
        enc = self.tokenizer(full_texts, return_tensors="pt", truncation=True, padding=True)
        full_ids = enc.input_ids.to(self.device, non_blocking=True)
        attn_mask = enc.attention_mask.to(self.device, non_blocking=True)

        max_len = min(full_ids.shape[1], 1024)
        full_ids, attn_mask = full_ids[:, -max_len:], attn_mask[:, -max_len:]

        with torch.autocast("cuda", dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)):
            out = model(full_ids, attention_mask=attn_mask, use_cache=False)
            logits = out.logits[:, :-1, :]
            labels = full_ids[:, 1:]
            logprobs = torch.log_softmax(logits, dim=-1)
            token_lp = logprobs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            lp = token_lp.sum(dim=-1)  # shape=[B]

        del out, logits, logprobs, token_lp
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return lp.detach()

    def compute_structured_reward(self, dataset_item) -> (float, Dict[str, float]):
        def normalize(x: float) -> float:
            return max(-1.0, min(1.0, (x - 0.5) * 2.0))

        ans_score = float(dataset_item.get("answer_score", 0.0))
        kw_score = float(dataset_item.get("keyword_score", 0.0))

        ans_reward = normalize(ans_score)
        kw_reward = normalize(kw_score)

        tag_rewards = dataset_item.get("tag_rewards", {})

        tool_reward = 1.0 if tag_rewards.get("<tool>", 0) >= 1 else -1.0
        fmt_reward = 1.0 if all(
            tag_rewards.get(t, 0) >= 1
            for t in ["<DAG>", "<search_result>", "<reason>", "<answer>"]
        ) else -1.0

        w_ans = self.cfg.ans_weight
        w_kw = self.cfg.kw_weight
        w_tool = self.cfg.tool_weight
        w_fmt = self.cfg.fmt_weight

        total_reward = (
            w_ans * ans_reward
            + w_kw * kw_reward
            + w_tool * tool_reward
            + w_fmt * fmt_reward
        )

        if self.is_main:
            eprint(
                f"[REWARD] ans={ans_reward:+.3f}, kw={kw_reward:+.3f}, "
                f"tool={tool_reward:+.1f}, fmt={fmt_reward:+.1f} "
                f"=> total={total_reward:+.3f}"
            )

        return total_reward, {
            "ans_reward": ans_reward,
            "kw_reward": kw_reward,
            "tool_reward": tool_reward,
            "fmt_reward": fmt_reward,
            "total_reward": total_reward,
        }


    



    def _grpo_group_update(
        self,
        prompt: str,
        texts: List[str],
        old_logprob_sum: List[torch.Tensor],
        rewards: List[float],
        advantages: List[float] = None,   
    ) -> Dict[str, Any]:

        self._print_cuda_mem("before PPO update")


        if isinstance(texts[0], list):
            flat_texts = []
            flat_old = []
            flat_rewards = []
            for g_texts, g_old, g_rewards in zip(texts, old_logprob_sum, rewards):
                flat_texts.extend(g_texts)
                flat_old.extend(g_old)
                flat_rewards.extend(g_rewards)
            texts, old_logprob_sum, rewards = flat_texts, flat_old, flat_rewards

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)


        if advantages is not None and len(advantages) == len(rewards):
            advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        else:
            if rewards.numel() == 0:
                baseline = torch.tensor(0.0, device=self.device)
                advantages = torch.zeros_like(rewards)
            else:
                baseline = rewards.mean()
                std = rewards.std(unbiased=False)
                if torch.isnan(std) or std < 1e-6:
                    std = torch.tensor(1.0, device=self.device)
                advantages = (rewards - baseline) / (std + 1e-4)
                advantages = torch.nan_to_num(advantages, nan=0.0)

        eps = self.cfg.ppo_clip_eps
        total_loss_tensor = torch.zeros((), device=self.device, requires_grad=True)

        for k, gen_text in enumerate(texts):
            cur_lp = self._sum_logprob_current(prompt, gen_text, detach=False)
            old_lp = old_logprob_sum[k].detach()

            ratio = torch.exp(cur_lp - old_lp)
            unclipped = ratio * advantages[k]
            clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages[k]
            ppo_loss = -torch.min(unclipped, clipped).mean()

            # === KL penalty ===
            if self.cfg.beta > 0 and self.ref_model is not None:
                kl_term = torch.exp(old_lp - cur_lp) - (old_lp - cur_lp) - 1
                kl_term = torch.nan_to_num(kl_term, nan=0.0)
                loss = ppo_loss + self.cfg.beta * kl_term
            else:
                loss = ppo_loss

            total_loss_tensor = total_loss_tensor + loss

        avg_loss = total_loss_tensor / max(1, len(texts))
        avg_loss = torch.nan_to_num(avg_loss, nan=0.0)

        self._print_cuda_mem("after PPO update")

        if self.is_main:
            eprint(f"[GRPO][rank={self.rank}] baseline={rewards.mean().item():.4f} | avg_loss={avg_loss.item():.4f}")

        return {
            "baseline": float(rewards.mean().item()),
            "loss": avg_loss,
            "rewards": [float(x) for x in rewards.detach().tolist()],
        }
    def _ppo_update(
        self,
        prompt: str,
        gen_text: str,
        old_logprob: torch.Tensor,
        reward: float,
        advantage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """

        """
        self._print_cuda_mem("before PPO update")

        # === Advantage ===
        if advantage is None:
            advantage = reward - reward  # baseline 0
        adv = torch.tensor(advantage, dtype=torch.float32, device=self.device)

        eps = self.cfg.ppo_clip_eps
        beta = self.cfg.beta

        cur_lp = self._sum_logprob_current(prompt, gen_text, detach=False)
        ratio = torch.exp(cur_lp - old_logprob.detach())

        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
        ppo_loss = -torch.min(unclipped, clipped).mean()

       
        if beta > 0 and self.ref_model is not None:
            with torch.no_grad():
                ref_lp = self._sum_logprob_batch(prompt, [gen_text], model=self.ref_model)
            kl_loss = (cur_lp - ref_lp.mean()) ** 2
            total_loss = ppo_loss + beta * kl_loss
        else:
            total_loss = ppo_loss

        self._print_cuda_mem("after PPO update")
        if self.is_main:
            eprint(f"[PPO] loss={total_loss.item():.4f}, adv={adv.item():+.3f}, reward={reward:+.3f}")

        return {"loss": total_loss, "adv": float(adv.item()), "reward": reward}


    def train_once(
        self,
        user_query: str,
        dna_seq: str,
        gold_json: Dict[str, Any],
        router=None,
    ) -> Dict[str, Any]:

        try:
            if not isinstance(self.model, torch.nn.Module):
                eprint(f"[WARN] Detected model={type(self.model)}; restoring original nn.Module.")
                self.model = getattr(self.dnallm, "text_model", self.model)
                if not isinstance(self.model, torch.nn.Module):
                    raise RuntimeError("Cannot restore valid model object for training.")
            if not hasattr(self, "_compiled") or not self._compiled:
                self.model.gradient_checkpointing_enable()
                if hasattr(self.model, "config"):
                    self.model.config.use_flash_attention_2 = True
                self.model = torch.compile(self.model, mode="max-autotune", fullgraph=True)
                self._compiled = True
                eprint("[torch.compile]  Compiled model (first time only).")
            else:
                eprint("[torch.compile]  Already compiled, skip recompile.")
        except Exception as e:
            eprint(f"[torch.compile]  skipped due to {e}")


        cfg = self.cfg
        self.model.train()

    
        items = self.build_dataset_from_query(user_query, dna_seq, gold_json, router or self.router)
        if not items:
            eprint("[PPO]  No dataset items generated.")
            return {}

        dataset_item = items[0]
        generated_answer = dataset_item.get("answer", "")

 
        reward, reward_detail = self.compute_structured_reward(dataset_item)
        baseline = 0.0
        advantage = reward - baseline

        old_logprob = self._sum_logprob_old(user_query, generated_answer)


        self.opt.zero_grad(set_to_none=True)
        rec = self._ppo_update(
            prompt=user_query,
            gen_text=generated_answer,
            old_logprob=old_logprob,
            reward=reward,
            advantage=advantage,
        )

        loss = rec["loss"]
        (loss / self.cfg.accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        eprint(f"[PPO][✓] reward={reward:+.3f}, loss={loss.item():.4f}, adv={advantage:+.3f}")

        return {
            "loss": float(loss.item()),
            "reward": reward,
            "advantage": advantage,
            "kw_tool_map": dataset_item.get("kw_tool_map", {}),
            "generated_answer": generated_answer,
            "reward_detail": reward_detail,
        }



    def build_dataset_from_query(
        self,
        user_query: str,
        dna_seq: str,
        gold_json: Dict[str, Any],
        router=None
    ) -> List[Dict[str, Any]]:
        

        import copy
        cfg = self.cfg
        eprint(f"[SINGLE-GROUP] Start single multi-hop generation for query: {user_query[:80]}")

        base_seed = getattr(cfg, "seed", 42)
        user_query_group = f"{user_query}"

        # === 执行单轮 multi-hop ===
        text_out, hop_records = generate_with_multi_hop(
            model=self.dnallm,
            router=router or self.router,
            user_query=user_query_group,
            dna_seq=dna_seq or "",
            max_hops=getattr(cfg, "max_hops", 3),
            temperature=getattr(cfg, "temperature", 1.0),
            do_sample=getattr(cfg, "do_sample", True),
            base_seed=base_seed,
        )

        if not hop_records:

            return []

      
        try:
            out_parsed = safe_json_loads(text_out)
            kw_score = hop_records[0].get("rl_keyword_score", 0.0)
            ans_score = out_parsed.get("rl_answer_score", 0.0)
        except Exception:
            kw_score, ans_score = 0.0, 0.0

        first_round = hop_records[0]
        final_round = hop_records[-1]

        fields = {
            "query": user_query_group,
            "DAG_first_round": first_round.get("DAG", ""),
            "multi_hop_results": [
                {
                    "round": i + 1,
                    "search_result": r.get("search_result", ""),
                    "answer": r.get("answer", ""),
                    "reason": r.get("reason", ""),
                    "decide": r.get("decide", "")
                }
                for i, r in enumerate(hop_records)
            ],
            "final_answer": final_round.get("answer", ""),
            "final_reason": final_round.get("reason", ""),
            "kw_tool_map": first_round.get("kw_tool_map", {}),
        }

   
        final_training_text = consolidate_episode_text({
            "query": user_query_group,
            "DAG": first_round.get("DAG", ""),
            "search_result": "\n".join([r.get("search_result", "") for r in hop_records]),
            "answer": fields["final_answer"],
            "reason": fields["final_reason"],
        })


        kw_tool_map = {k: list(v) for k, v in first_round.get("kw_tool_map", {}).items()}
        tools_flat = sorted({t for tools in kw_tool_map.values() for t in tools})
        tag_rewards = {
            "<query>": 1 if user_query_group else 0,
            "<DAG>": 1 if first_round.get("DAG") else 0,
            "<search_result>": 1 if hop_records else 0,
            "<keyword>": 1 if kw_tool_map else 0,
            "<tool>": 1 if tools_flat else 0,
            "<answer>": 1 if fields["final_answer"] else 0,
        }

        dataset_item = {
            "labels": [gold_json.get("answer", "")],
            "gold_struct": copy.deepcopy(gold_json),
            "keyword_score": kw_score,
            "answer_score": ans_score,
            "final_training_text": final_training_text,
            "final_training_fields": copy.deepcopy(fields),
            "answers_multi": copy.deepcopy(hop_records),
            "answer": fields["final_answer"],
            "reason": fields["final_reason"],
            "kw_tool_map": copy.deepcopy(kw_tool_map),
            "tag_rewards": copy.deepcopy(tag_rewards),
            "user_query": user_query_group,
            "dna_seq": dna_seq,
            "group_id": 1,
        }

        eprint(f"[SINGLE-GROUP]  Finished with answer_len={len(fields['final_answer'])}")
        return [dataset_item]






