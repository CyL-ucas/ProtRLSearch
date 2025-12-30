# rl/reward.py 
import re
import json
import unicodedata, html
from typing import Dict, Any, Tuple, List, Set
from sentence_transformers import SentenceTransformer, util
from bioreason.core.utils import eprint
from bioreason.models.models import gemini_llm,gpt_llm

REQUIRED_KEYS = ["query", "reason", "DAG", "search_result", "answer"]

def gpt_Keyword_score(query: str, pred: str, gold: str) -> float:
    prompt = f"""
You are a biomedical keyword evaluator.
Your task is to assess how well the model-predicted keywords capture the essential biomedical entities and concepts found in the ground truth.

Rate the **coverage**, **relevance**, and **specificity** of the predicted keywords on a numeric scale.

### Scoring scale (for reinforcement learning reward)
Use a **continuous score between 0.0 and 1.0**, where:
- **1.0** → Perfect: all key entities and concepts match or are strong semantic equivalents.
- **0.8–0.9** → Good: mostly correct; only minor omissions or slight terminology mismatch.
- **0.6–0.7** → Fair: captures some relevant biomedical entities but misses important ones.
- **0.4–0.5** → Poor: vague or overly generic; weak biological relevance.
- **0.1–0.3** → Bad: mostly irrelevant, misleading, or biologically incorrect.
- **0.0** → Completely wrong: nonsensical or unrelated to the ground truth.

### Negative feedback interpretation
For training:
- Scores **< 0.5** are treated as **negative feedback** (model receives a penalty).
- Scores **≥ 0.5** are treated as **positive feedback** (model receives a reward).
- You may optionally think of this as mapping to reward ∈ [-1, +1],  
  where `reward = (score - 0.5) * 2`.

### Input
Query: {query}

Predicted keywords: {pred}
Ground truth keywords: {gold}


Return ONLY one numeric score between 0 and 1.
Do not include explanations or additional text.


    """.strip()
    try:
        resp = gpt_llm.chat([{"role": "user", "content": prompt}])
        
       
        m = re.search(r"(\d*\.\d+|\d+)", resp)
        return float(m.group(1)) if m else 0.0
    except Exception as e:
        eprint(f"[WARN] GPT scoring failed: {e}")
        return 0.0
def gpt_answer_score(query: str, pred: str, gold: str) -> float:

    prompt = f"""
You are a biomedical reasoning evaluator.
Your goal is to assess how **factually correct**, **semantically consistent**, and **mechanistically accurate** the model’s predicted answer is compared with the ground truth.

Rate the predicted answer on a **continuous numeric scale from 0.0 to 1.0**,  
where 0.0 = completely wrong and 1.0 = perfectly correct.

### Scoring guidelines
- **1.0** → Fully consistent: same biological conclusion and reasoning as the ground truth.
- **0.8–0.9** → Minor stylistic or wording differences, but identical mechanistic understanding.
- **0.6–0.7** → Partially correct: captures some biological relationships but omits or distorts key mechanisms.
- **0.4–0.5** → Weak reasoning; vague or partially wrong with limited factual overlap.
- **0.1–0.3** → Major biological or mechanistic errors.
- **0.0** → Completely unrelated, contradictory, or nonsensical content.

### Negative feedback interpretation
For reinforcement learning:
- Scores **below 0.5** correspond to **negative feedback** (model receives a penalty).
- Scores **above 0.5** correspond to **positive feedback** (model receives a reward).
- The score can optionally be mapped to the reward range **[-1, +1]** using  
  `reward = (score - 0.5) * 2`.

### Input
Query:
{query}

Predicted answer:
{pred}

Ground truth answer:
{gold}

Return **only one numeric score between 0 and 1.**  
Do **not** include explanations or any text other than the number.
""".strip()

    try:
        resp = gpt_llm.chat([{"role": "user", "content": prompt}])
 
 
        m = re.search(r"(\d*\.\d+|\d+)", resp)
        return float(m.group(1)) if m else 0.0
    except Exception as e:
        eprint(f"[WARN] GPT scoring failed: {e}")
        return 0.0

def _strip_outer_tag(block: str, tag: str) -> str:
    if not block:
        return ""
  
    return re.sub(fr"^\s*<{tag}>\s*|\s*</{tag}>\s*$", "", block, flags=re.I | re.S).strip()


from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util
import torch, re

def rerank_with_pubmedbert_and_gemini(all_results, query: str, batch_size: int = 8, max_workers: int = 8):

    if not all_results:
        return []


    model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    query_emb = model.encode(query, convert_to_tensor=True, device=device)

  
    docs = []
    for step in all_results:
        for d in step.get("docs", []):
            title = (d.get("title") or "").strip()
            abstract = d.get("abstract", "")
            if isinstance(abstract, list):
                abstract = " ".join(x.strip() for x in abstract if isinstance(x, str))
            abstract = (abstract or "").strip()
            if not (title or abstract):
                continue
            text = f"{title}. {abstract}".strip()
            docs.append({"title": title, "abstract": abstract, "text": text})

    if not docs:
        return []


    def encode_batch(batch_texts):
   
        try:
            return model.encode(batch_texts, convert_to_tensor=True, device=device)
        except Exception as e:
            eprint(f"[WARN] batch encode failed: {e}")
            return torch.zeros((len(batch_texts), model.get_sentence_embedding_dimension()))

    # 多线程批量编码
    batch_list = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
    embeddings = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(encode_batch, [d["text"] for d in batch]): batch for batch in batch_list}
        for f in as_completed(futures):
            emb_batch = f.result()
            if isinstance(emb_batch, torch.Tensor):
                embeddings.append(emb_batch)

    if not embeddings:
        return []
    doc_embs = torch.cat(embeddings, dim=0)

 
    cos_sims = util.cos_sim(query_emb, doc_embs)[0]
    top5_idx = torch.topk(cos_sims, k=min(5, len(docs))).indices.tolist()
    top5_docs = [docs[i] for i in top5_idx]

 
    def score_doc(d):
        prompt = f"""
You are a biomedical reasoning evaluator. 
Given a research query and a candidate paper (title + abstract),
rate how directly this paper supports, answers, or relates to the query.

Query:
{query}

Paper:
Title: {d['title']}
Abstract: {d['abstract']}

Return ONLY a float score between 0 and 1 (1 = highly relevant, 0 = irrelevant).
        """.strip()
        try:
            resp = gemini_llm.chat([{"role": "user", "content": prompt}])

      
            m = re.search(r"(\d*\.\d+|\d+)", resp)
            score = float(m.group(1)) if m else 0.0
     
            return score
        except Exception as e:
            eprint(f"[Gemini WARN] scoring failed: {e}")
            return 0.0

    with ThreadPoolExecutor(max_workers=min(max_workers, len(top5_docs))) as executor:
        futures = {executor.submit(score_doc, d): d for d in top5_docs}
        for f in as_completed(futures):
            d = futures[f]
            try:
                d["gemini_score"] = f.result()
            except Exception as e:
                d["gemini_score"] = 0.0
                print(f"[Gemini WARN] scoring future failed: {e}")

   
    top2 = sorted(top5_docs, key=lambda x: x.get("gemini_score", 0.0), reverse=True)[:2]
    return top2







