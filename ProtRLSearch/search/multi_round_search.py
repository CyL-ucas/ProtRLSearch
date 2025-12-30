import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Set, Union
import datetime
import unicodedata, html
from sentence_transformers import SentenceTransformer, util
import torch, re
import requests
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


from bioreason.core.utils import (
    set_verbose, eprint, _extract_terms_from_purpose,
    _split_terms, clean_keyword_phrase, _split_terms_preserve_phrase, _pick_top_terms, _STOPWORDS
)
from string import Template
from bioreason.search_tools.pubmed_search import PubMedSearcher
from bioreason.models.qwen_llm import QwenLLM

from bioreason.models.prompts import (
    DAG_PROMPT, build_initial_prompt
)
from bioreason.search_tools.ToolRouter import ToolRouter

from bioreason.models.build_dag import _build_dag_with_llm, build_combined_query_from_dag,_norm_tool,_norm_txt,_parse_dag_mixed_xml_like,_flatten_kw_tool_lines
from bioreason.models.models import gemini_llm,gpt_llm
from bioreason.rL.load_groundtruth import get_groundtruth
from bioreason.rL.reward import gpt_Keyword_score,rerank_with_pubmedbert_and_gemini,gpt_answer_score

def safe_json_loads(line: str):
 
    import json, re
    if not line or not isinstance(line, str):
        return None
    line = line.strip()
   
    line = line.strip("`").strip()
    if line.startswith("json"):
        line = line[len("json"):].strip()

    line = re.sub(r",\s*([\]}])", r"\1", line)
    line = re.sub(r"[\u0000-\u001F]", "", line)  
 
    if "'" in line and '"' not in line:
        line = line.replace("'", '"')
    try:
        return json.loads(line)
    except Exception as e:
        print(f"[WARN] Failed to parse line: {e}")
        return None





from string import Template
PROMPT_GEN_NEW_ANSWER = Template("""
You are **BioReason Executor**, a biomedical reasoning assistant specialized in molecular and cellular mechanisms.  
This is **NOT** a keyword extraction step — the keywords have already been generated earlier.  
Your ONLY goal is to refine the previous conclusion ONCE using the new evidence below, then stop immediately.  
Do NOT start another reasoning round or generate new queries.

<context>
<search_results>
${search_results}
</search_results>
<previous_conclusion>
${prev_answer}
</previous_conclusion>
</context>

### TASK
Based only on the information in <search_results> and <previous_conclusion>, perform the following:
1. Compare the new evidence with the previous conclusion — identify whether it confirms, contradicts, or extends it.
2. Rewrite a **complete new reasoning** in <reason>, integrating all new mechanistic insights.
3. Write a **final, evidence-based conclusion** in <answer>, fully rewritten in your own words.
4. Decide whether the current evidence completely answers the question (<decide>yes</decide> or <decide>no</decide>).
5. If <decide>no</decide>, propose a meaningful next step or follow-up question in <next_query>.

### STRICT OUTPUT RULES
- Output **exactly four** XML-style tags, in this exact order:
  <reason>...</reason>
  <answer>...</answer>
  <decide>yes or no</decide>
  <next_query>...</next_query>
- Begin directly with `<reason>` as your first token.
- **Do NOT output any markdown, commentary, bullet points, or text outside these tags.**
- **Do NOT produce `<keyword>`, `<round>`, `<query>`, or `</end>` tags.**
- **Do NOT attempt to answer or elaborate on <next_query>.**  
  Simply output it and stop.
- **Stop generating immediately** after printing `</next_query>`.  
  Do not repeat, rephrase, or self-call in any way.
- **Immediately stop generation** after printing `</next_query>`. Do not continue, repeat, or self-call.
At the end of your answer, print exactly "</next_query>" and then stop.
Do not generate anything after it — not even a new line.
Begin your structured reasoning refinement now:
""".strip())



PROMPT_GEN_ANSWER = Template("""
You are **BioReason Executor**, a biomedical reasoning assistant specialized in molecular and cellular biology.  
This is **NOT** a keyword extraction step — the keywords have already been generated.  
Your ONLY goal is to produce one round of reasoning and stop.  
You must NOT start any new round, query, or follow-up reasoning after printing your output.

<context>
${search_results}
</context>

### TASK
Based only on the studies provided in <context>, perform the following:
1. Identify consistent mechanistic findings related to the question.
2. Explain the biological reasoning process inside <reason>.
3. Summarize your final, evidence-based conclusion inside <answer>.
4. Decide whether the retrieved context fully answers the question (<decide>yes</decide> or <decide>no</decide>).
5. If <decide>no</decide>, propose a refined next question inside <next_query>.

### STRICT OUTPUT RULES
- Output must contain **exactly four** XML-style tags, in this exact order:
  <reason>...</reason>
  <answer>...</answer>
  <decide>yes or no</decide>
  <next_query>...</next_query>
- Begin directly with `<reason>` as your first token.
- **Do NOT output any markdown, bullet points, or text outside these tags.**
- **Do NOT output placeholder text such as "..." inside any tag.**
- **Do NOT generate `<keyword>`, `</end>`, `<round>`, or any new `<query>` tags.**
- **Do NOT attempt to execute or elaborate on `<next_query>`.**  
  You must simply write it and then stop.
- **Immediately stop generation** after printing `</next_query>`. Do not continue, repeat, or self-call.
At the end of your answer, print exactly "</next_query>" and then stop.
Do not generate anything after it — not even a new line.

Begin your structured reasoning now:
""".strip())


def extract_all_tags(raw_text: str) -> Dict[str, str]:

    import re, html, unicodedata

    result = {"reason": "", "answer": "", "decide": "", "next_query": ""}
    if not isinstance(raw_text, str) or not raw_text.strip():
        return result


    raw_text = raw_text.replace("<decise>", "<decide>").replace("</decise>", "</decide>")
    raw_text = re.sub(r"[\u0000-\u001F]+", "", raw_text)  
    raw_text = html.unescape(unicodedata.normalize("NFKC", raw_text))

    def extract_valid_blocks(tag: str) -> list[str]:
      
        pattern = rf"<{tag}\b[^>]*>(.*?)</{tag}>"
        matches = re.findall(pattern, raw_text, flags=re.I | re.S)
        valid = []
        for content in matches:
            text = content.strip()
       
            if "..." in text:
                continue
            if not text:
                continue
            if re.match(r"^[\s\n\r\t]+$", text):
                continue
       
            text = re.sub(r"[\u0000-\u001F]+", "", text).strip()
            if text:
                valid.append(text)
        return valid


    for tag in result.keys():
        valid_tags = extract_valid_blocks(tag)
        result[tag] = valid_tags[-1] if valid_tags else ""


    result["decide"] = result["decide"].strip().lower()
    if result["decide"] not in {"yes", "no"}:
        result["decide"] = "no"

    if result["decide"] == "yes":
        result["next_query"] = ""

    return result




def _sanitize_prompt_text(text: str) -> str:

    if not isinstance(text, str):
        return text

 
    for marker in ["REQUIRED OUTPUT", "OUTPUT FORMAT", "INSTRUCTIONS:"]:
        idx = text.find(marker)
        if idx > -1:
          
            pass


    cut_markers = [
        "\nimport ", "\nfrom ", "\nclass ", "\ndef ",
        "\n# =====", "\n# ===", "\n@lru_cache", "\nif __name__"
    ]
    cut_pos = len(text)
    for mk in cut_markers:
        p = text.find(mk)
        if p != -1:
            cut_pos = min(cut_pos, p)
    if cut_pos < len(text):
        text = text[:cut_pos].rstrip()

  
    MAX_LEN_CHARS = 8000
    if len(text) > MAX_LEN_CHARS:
        text = text[:MAX_LEN_CHARS].rstrip()

    return text

from functools import lru_cache

@lru_cache(maxsize=2048)
def _cached_router_call(router, tool, kw): 
    try: 
        return router.call(tool, kw) or [] 
    except Exception as e: 
        eprint(f"[WARN] {tool}({kw}) failed: {e}") 
    return []
def _single_round_step(
    model: QwenLLM,
    router: ToolRouter,
    query: str,
    dna_seq: str,
    prev_answer: Optional[str] = None,
    hop: int = 1,
    max_new_tokens_step: int = 5000,
    temperature: float = 0.7,
    do_sample: bool = True,
    base_seed: int = 42,    
) -> Dict[str, Any]:
    
    import json as _json, re, random
    from concurrent.futures import ThreadPoolExecutor
    from transformers import StoppingCriteriaList

 
    seed = base_seed + hop * 1234
    random.seed(seed)
    torch.manual_seed(seed)
    eprint(f"[SEED] Set random seed = {seed}")

  
    temperature = round(random.uniform(0.65, 0.95), 2)
    top_p = round(random.uniform(0.80, 0.95), 2)
    eprint(f"[SAMPLING] temperature={temperature} | top_p={top_p}")

    round_type = "FIRST" if prev_answer is None else f"FOLLOW-UP (ROUND {hop})"
    eprint(f"\n[ ROUND START] === {round_type} ===")
    eprint(f"[QUERY] {query}")

 
    if not hasattr(model, "text_model") or not isinstance(model.text_model, torch.nn.Module):
        if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
            model.text_model = model.model
            eprint("[FIX] Restored model.text_model from model.model.")
        else:
            raise TypeError(f"[FATAL] Invalid model: expected QwenLLM with .text_model")

    model.text_model.eval()


    try:
        dag = _build_dag_with_llm(model, topic=query, dna_seq=dna_seq or "")
        eprint(f"[DAG]  Generated with {len(dag)} nodes.")
    except Exception as ex:
     
        dag = []
    dag_json = _json.dumps(dag, ensure_ascii=False)


    kw_tool_map: dict[str, set[str]] = {}
    for node in dag:
        kw = node.get("keyword", "").strip()
        tools = node.get("tool", [])
        if isinstance(tools, str):
            tools = [tools]
        if kw:
            kw_tool_map.setdefault(kw, set()).update([t.lower().strip() for t in tools if t])
    kw_tool_map = {k: sorted(v) for k, v in kw_tool_map.items() if k}

    eprint(f"[KW→TOOL MAP] {len(kw_tool_map)} entries." if kw_tool_map else "[KW→TOOL MAP] ⚠️ None found.")

 
    all_results: List[Dict[str, Any]] = []

    def fetch(tool, kw):
        try:
            docs = _cached_router_call(router, tool, kw)
            return {"tool": tool, "query": kw, "docs": docs}
        except Exception as e:
            eprint(f"[WARN] {tool}({kw}) failed: {e}")
            return {"tool": tool, "query": kw, "docs": []}

    if kw_tool_map:
        with ThreadPoolExecutor() as ex:
            futures = [
                ex.submit(fetch, tool, kw)
                for kw, tools in kw_tool_map.items()
                for tool in tools
            ]
            for f in futures:
                all_results.append(f.result())

    total_docs = sum(len(r["docs"]) for r in all_results)

    top_docs = rerank_with_pubmedbert_and_gemini(all_results, query)
    formatted = [
        f"{i+1}. {d['title']}. {d['abstract']} (Gemini={d['gemini_score']:.2f})"
        for i, d in enumerate(top_docs)
    ] if top_docs else [f"No valid documents retrieved for {query}."]

    sr_block = "<search_results>\n" + "\n".join(formatted) + "\n</search_results>"
    clean_sr = re.sub(r"</?search_results>", "", sr_block, flags=re.I)

    
    if prev_answer:
        prompt_text = PROMPT_GEN_NEW_ANSWER.substitute(search_results=clean_sr, prev_answer=prev_answer)
    else:
        prompt_text = PROMPT_GEN_ANSWER.substitute(search_results=clean_sr)

    prompt_text = _sanitize_prompt_text(prompt_text)

 
    gen_prompt = f"<round>{hop}</round>\n<query>{query}</query>\n{prompt_text}\n"

   
    tokenizer = getattr(model, "text_tokenizer", None) or getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("No valid tokenizer found in model.")

    eprint("[DEBUG] === Final Prompt Sent to Model ===")
    raw_out = model.generate_text(
        gen_prompt,
        max_new_tokens=min(max_new_tokens_step, 3000),
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )


    tags = extract_all_tags(raw_out)
    reason, answer, decide, next_q = tags["reason"], tags["answer"], tags["decide"], tags["next_query"]

    if decide == "yes":
        next_q = ""

    eprint("------ [ANSWER EXTRACTION DEBUG] ------")
    eprint(f"[EXTRACTED REASON]\n{reason[:400]}")
    eprint(f"[EXTRACTED ANSWER]\n{answer[:400]}")
    eprint(f"[DECIDE] {decide.upper()} | [NEXT_QUERY] {next_q[:200]}")
    eprint("---------------------------------------")

    return {
        "query": query,
        "DAG": dag_json,
        "search_result": sr_block,
        "reason": reason,
        "answer": answer,
        "decide": decide,
        "next_query": next_q,
        "kw_tool_map": kw_tool_map,
        "raw_output": raw_out,        
    }

def generate_with_multi_hop(
    model: QwenLLM,
    router: ToolRouter,
    user_query: str,
    dna_seq: str,
    max_hops: int = 3,
    max_new_tokens_step: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
    base_seed: Optional[int] = None,  # 
):

    import json as _json, gc, torch, random

    if base_seed is None:
        base_seed = random.randint(1000, 999999)
    eprint(f"[INIT]  Using base_seed = {base_seed}")

    hop_records: List[Dict[str, Any]] = []
    query = user_query
    prev_answer = None
    final_answer, last_reason, final_decision = "", "", "no"
    stop_due_to_yes = False

    for hop in range(1, max_hops + 1):
        eprint(f"\n==============================")
        eprint(f"[MULTI-HOP] STEP {hop}/{max_hops}")
        eprint(f"==============================")

        rec = _single_round_step(
            model=model,
            router=router,
            query=query,
            dna_seq=dna_seq,
            prev_answer=prev_answer,
            hop=hop,
            base_seed=base_seed,
        )
        hop_records.append(rec)

     
        final_answer = rec.get("answer", "")
        last_reason = rec.get("reason", "")
        final_decision = rec.get("decide", "no").strip().lower()
        prev_answer = final_answer

     
        eprint(f"[STEP {hop} SUMMARY]")
        eprint(f"  Decide: {final_decision.upper()}")
        eprint(f"  Answer: {final_answer[:200]}")

    
        if final_decision == "yes":
        
            stop_due_to_yes = True
            
      
            raw_out = rec.get("raw_output", "")
           

    
  


        next_q_raw = rec.get("next_query", "").strip()
        if next_q_raw:
         
            next_q_clean = re.sub(r"</?topic>", "", next_q_raw, flags=re.I)
            next_q_clean = re.sub(r"<.*?>", "", next_q_clean, flags=re.S)
            next_q_clean = next_q_clean.strip()


            query = f"<topic>{next_q_clean}</topic>"
        else:
            eprint(f"[Multi-Hop] ")
            break


  
    if stop_due_to_yes:
        eprint(f"[FINAL]  Using answer from YES decision round.")
    else:
        eprint(f"[FINAL] No YES decision found, using last round (decide={final_decision}) answer.")
        if hop_records:
            final_answer = hop_records[-1].get("answer", "")
            last_reason = hop_records[-1].get("reason", "")

            raw_out = hop_records[-1].get("raw_output", "")
            eprint("===== [RAW OUTPUT OF LAST ROUND] =====")
            eprint(raw_out)
            eprint("======================================")



    try:
        if hasattr(model, "dna_model") and model.dna_model is not None:
            model.dna_model.to("cpu")
            del model.dna_model
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            mem_free = torch.cuda.mem_get_info()[0] / 1024**2
   
    except Exception as e:
        eprint(f"[CLEANUP][WARN] {e}")

  
    combined_sr = "\n".join([f"[Round {i+1}] {r['search_result']}" for i, r in enumerate(hop_records)])
    
    gt = get_groundtruth(user_query)
    
    gt_answer = gt.get("answer", "")
    gt_keywords = gt.get("keywords", [])
    kw_tool_map = hop_records[0].get("kw_tool_map", {}) if hop_records else {}
    model_keywords = list(kw_tool_map.keys())

    out = {
        "query": user_query,
        "reason": last_reason,
        "DAG": hop_records[0]["DAG"] if hop_records else "",
        "search_result": combined_sr,
        "answer": final_answer,
        "decision": final_decision,
        "kw_tool_map": {
            k: list(sorted(v)) for k, v in (
                hop_records[0].get("kw_tool_map", {}).items() if hop_records else {}
            )
        },
        "total_hops": len(hop_records)
    }
    

  
    try:
        score_kw = gpt_Keyword_score(user_query, model_keywords, gt_keywords)
        eprint(f"[RL]  Keyword relevance score = {score_kw:.3f}")
        out["rl_keyword_score"] = score_kw
        if hop_records:
            hop_records[0]["rl_keyword_score"] = score_kw
    except Exception as e:
        eprint(f"")


 
    clean_ans = (final_answer or "").strip()

   
    
    score_ans = gpt_answer_score(user_query, clean_ans, gt_answer)
    eprint(f"[RL] Answer relevance score = {score_ans:.3f}")
    out["rl_answer_score"] = score_ans
    if hop_records:
        hop_records[0]["rl_answer_score"] = score_ans
        



    return _json.dumps(out, ensure_ascii=False, indent=2), hop_records

def consolidate_episode_text(fields: Dict[str, str]) -> str:
    q = fields.get('query','')
    r = fields.get('reason','')
    dag_raw_or_json = fields.get('DAG','')  
    sr = fields.get('search_result','')
    a = fields.get('answer','')

 
    parsed = []
    try:
        obj = safe_json_loads(dag_raw_or_json)
        parsed = obj if isinstance(obj, list) else (obj.get("nodes") if isinstance(obj, dict) else [])
    except Exception:
        parsed = _parse_dag_mixed_xml_like(dag_raw_or_json)

    kw_tool_lines = _flatten_kw_tool_lines(parsed)

    return (
        f"<query>{q}</query>\n"
        f"<reason>{r}</reason>\n"
        f"<DAG>{dag_raw_or_json}</DAG>\n"
        f"<search_result>{sr}</search_result>\n"
        f"<answer>{a}</answer>\n"
        f"{kw_tool_lines}"
    )

