# bioreason/search_tools/ToolRouter.py

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Set, Union
import datetime

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


from bioreason.core.io import extract_dna_from_text, load_inputs_from_json
from bioreason.core.utils import set_verbose, eprint, iprint, str2bool, _disable_hf_transfer_if_missing, is_offline, resolve_hf_snapshot, ensure_local_ok
from bioreason.search_tools.pubmed_search import PubMedSearcher  
from bioreason.dna_modules.dna_llm import DNALLM
_VERBOSE = False



class ToolRouter:
   
    def __init__(
        self,
        retriever_url: Optional[str] = None,
        pubmed_url: Optional[str] = None,
        arxiv_url: Optional[str] = None,
        uniprot_url: Optional[str] = None,
        uniprot_blast_url: Optional[str] = None,
        alphafold_url: Optional[str] = None,
        topk: int = 1,
        timeout: int = 25,
        offline: Optional[bool] = None,
        pubmed_embedder: Optional[object] = None,  
    ):
        self.endpoints = {
            "pubmed": pubmed_url or "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            "uniprot": uniprot_url or "https://rest.uniprot.org",
            "uniprot_blast": uniprot_blast_url or "https://rest.uniprot.org/blast",
            "alphafold": alphafold_url or "https://alphafold.ebi.ac.uk/entry",
            "websearch": "https://api.duckduckgo.com", 
        }
        self.topk = max(1, int(topk))
        self.timeout = timeout
        self.offline = bool(int(os.environ.get("OFFLINE", "0"))) if offline is None else offline
        self._session = requests.Session()
        self._pubmed_engine = PubMedSearcher()  
        self._pubmed_embedder = pubmed_embedder  
        self.max_docs = 3   

    @staticmethod
    def _simulate(tool: str, query: str, topk: int = 5) -> List[Dict[str, str]]:
        pref = f"[{tool.upper()}]"
        return [
            {"title": f"{pref} Simulated Result {i} for '{query}'", "abstract": f"Simulated abstract {i} about '{query}'.", "url": ""}
            for i in range(1, topk + 1)
        ]

  
    def _call_websearch_local(self, query: str) -> List[Dict[str, str]]:
        if self.offline:
            eprint("[WARN] OFFLINE=1 → websearch uses simulation.")
            return self._simulate("websearch", query, self.topk)
        try:
            r = self._session.get(
                self.endpoints["websearch"],
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            out: List[Dict[str, str]] = []
            if data.get("AbstractText"):
                out.append(
                    {"title": data.get("Heading") or query, "abstract": data.get("AbstractText"), "url": data.get("AbstractURL") or ""}
                )
            for t in data.get("RelatedTopics", [])[: max(0, self.topk - len(out))]:
                if isinstance(t, dict) and t.get("Text"):
                    out.append({"title": t.get("Text"), "abstract": "", "url": t.get("FirstURL", "")})
            return out[: self.topk] or self._simulate("websearch", query, self.topk)
        except Exception as ex:
            eprint(f"[WARN] websearch error: {ex}")
            return self._simulate("websearch", query, self.topk)
    
  
    def _call_pubmed_api(self, query_or_dag: Union[str, Dict]) -> Optional[List[Dict[str, str]]]:
        if self.offline:
            eprint("[WARN] OFFLINE=1 → pubmed disabled.")
            return None

    
        if isinstance(query_or_dag, dict) and query_or_dag.get("nodes"):
            queries = _build_pubmed_queries_from_nodes(
                query_or_dag,
                last_years=5,
                pub_types=["Review", "Classical Article"],
                language="English"
            )
            query = queries[0] if queries else ""
        else:
            query = str(query_or_dag).strip()

        if not query:
            return None

        try:
            articles = self._pubmed_engine.search_pubmed(query, max_results=max(20, self.topk)) or []
            if not articles:
                eprint(f"[WARN] PubMedSearcher returned 0 results; q='{query}'")
                return None  

            docs: List[Dict[str, str]] = []
            for a in articles[: self.topk]:
                title = (a.get("title") or "").strip()
                abstract_txt = (a.get("abstract") or "").strip()
                authors = a.get("authors") or []
                lead_authors = ", ".join(authors[:3]) if authors else ""
                year = str(a.get("pubdate") or "").strip()
                prefix = f"{lead_authors} ({year}). " if (lead_authors or year) else ""
                abstract = (prefix + abstract_txt).strip()
                doi = (a.get("doi") or "").strip()
                url = f"https://doi.org/{doi}" if doi else self._pubmed_engine.search_pubmed_url(query)
                docs.append({"title": title, "abstract": abstract, "url": url})
            return docs
        except Exception as ex:
            eprint(f"[WARN] PubMedSearcher error: {ex} (q='{query}')")
            return None



   
    def _call_uniprot_api(self, query: str) -> List[Dict[str, str]]:
        if self.offline:
            eprint("[WARN] OFFLINE=1 → uniprot search uses simulation.")
            return self._simulate("uniprot", query, self.topk)
        try:
            q = query
            if q.lower().startswith("uniprot:"):
                acc = q.split(":", 1)[-1].strip()
                return [{"title": f"uniprot:{acc}", "abstract": "", "url": f"https://www.uniprot.org/uniprotkb/{acc}"}]

            r = self._session.get(
                f"{self.endpoints['uniprot']}/uniprotkb/search", params={"query": q, "format": "json", "size": self.topk}, timeout=self.timeout
            )
            r.raise_for_status()
            data = r.json()
            out: List[Dict[str, str]] = []
            for it in data.get("results", [])[: self.topk]:
                acc = it.get("primaryAccession") or ""
                name = ""
                try:
                    name = it["proteinDescription"]["recommendedName"]["fullName"]["value"]
                except Exception:
                    pass
                org = ""
                try:
                    org = it["organism"]["scientificName"]
                except Exception:
                    pass
                title = f"uniprot:{acc} {name}".strip()
                abstract = f"{org}".strip()
                url = f"https://www.uniprot.org/uniprotkb/{acc}"
                out.append({"title": title, "abstract": abstract, "url": url})
            return _normalize_uniprot_docs(out) or self._simulate("uniprot", query, self.topk)
        except Exception as ex:
            eprint(f"[WARN] uniprot search error: {ex}")
            return self._simulate("uniprot", query, self.topk)


    def _call_uniprot_blast(
        self,
        sequence: str,
        program: Optional[str] = None,
        database: str = "uniprotkb",
        taxid: Optional[str] = None,
        max_wait_s: int = 120,
        poll_every_s: float = 2.0,
    ) -> List[Dict[str, str]]:
        if self.offline:
            eprint("[WARN] OFFLINE=1 → uniprot blast uses simulation.")
            return self._simulate("uniprot(blast)", sequence[:24] + "...", self.topk)
        seq = (sequence or "").strip()
        if not seq:
            return self._simulate("uniprot(blast)", "<EMPTY_SEQUENCE>", self.topk)
        try:
            payload = {"query": seq, "program": (program or "blastx"), "database": database}
            if taxid:
                payload["taxId"] = str(taxid)
            rr = self._session.post(f"{self.endpoints['uniprot_blast']}/run", data=payload, timeout=self.timeout)
            rr.raise_for_status()
            job_id = rr.text.strip()

            deadline = time.time() + max_wait_s
            while time.time() < deadline:
                st = self._session.get(f"{self.endpoints['uniprot_blast']}/status/{job_id}", timeout=self.timeout)
                st.raise_for_status()
                status = st.text.strip().lower()
                if status in {"finished", "failed"}:
                    break
                time.sleep(poll_every_s)

            res = self._session.get(
                f"{self.endpoints['uniprot_blast']}/results/{job_id}", params={"format": "tsv"}, timeout=self.timeout
            )
            res.raise_for_status()
            lines = [ln for ln in res.text.splitlines() if ln.strip()]
            docs: List[Dict[str, str]] = []
            for ln in lines[1:][: self.topk]:
                cols = ln.split("\t")
                if not cols:
                    continue
                acc = cols[0].split("|")[-1] if "|" in cols[0] else cols[0]
                evalue = cols[4] if len(cols) > 4 else ""
                title = f"uniprot:{acc} (BLAST hit)"
                abstract = f"e-value={evalue}"
                url = f"https://www.uniprot.org/uniprotkb/{acc}"
                docs.append({"title": title, "abstract": abstract, "url": url})
            return _normalize_uniprot_docs(docs) or self._simulate("uniprot(blast)", "no_hits", self.topk)
        except Exception as ex:
            eprint(f"[WARN] uniprot blast error: {ex}")
            return self._simulate("uniprot(blast)", f"{program or 'auto'}:{database}", self.topk)

 
    def _call_alphafold(self, query: str) -> List[Dict[str, str]]:
        q = (query or "").strip()
        if q.lower().startswith("uniprot:"):
            acc = q.split(":", 1)[-1].strip()
        else:
            acc = _extract_acc_from_text(q) or q
        url = f"{self.endpoints['alphafold']}/{acc}"
        return [{"title": f"AlphaFold: {acc}", "abstract": "", "url": url}]

  
    def call(
        self,
        tool: str,
        query: str,
        *,
        mode: Optional[str] = None,
        sequence: Optional[str] = None,
        program: Optional[str] = None,
        database: str = "uniprotkb",
        taxid: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        tool = (tool or "").lower().strip()
        try:
            if tool == "uniprot":
                m = (mode or "").strip().lower()
                if m == "blast":
                    seq = (sequence or query or "").strip()
                    return self._call_uniprot_blast(seq, program=program, database=database, taxid=taxid)
                q = query.strip()
                if q.lower().startswith("uniprot:"):
                    acc = q.split(":", 1)[-1].strip()
                    q = _mk_uniprot_accession_query(acc)
                elif not re.search(r"\b(gene:|gene_exact:|accession:|organism_name:|organism_id:)\b", q):
                    gene, org = _parse_gene_org_for_uniprot(q)
                    if gene:
                        q = _mk_uniprot_fielded_query(gene, org)
                return self._call_uniprot_api(q)

            elif tool == "pubmed":
                return self._call_pubmed_api(query)

            elif tool == "alphafold":
                return self._call_alphafold(query)

            else:
                return self._call_websearch_local(query)

        except Exception as e:
            eprint(f"[WARN] router.call error ({tool}): {e}. Falling back to simulation.")
            return self._simulate(tool or "websearch", query, self.topk)

def _mk_block(terms: List[str], op: str, field: str = "Title/Abstract") -> str:
    terms = [t for t in terms if t]
    if not terms:
        return ""
    if len(terms) == 1:
        return f"\"{terms[0]}\"[{field}]"
    inner = f" {op} ".join(f"\"{t}\"[{field}]" for t in terms)
    return f"({inner})"


def _mk_date_filter(last_years: int) -> str:
    if not last_years or last_years <= 0:
        return ""
    start_year = datetime.datetime.now().year - last_years
    return f"(\"{start_year}/01/01\"[PDAT] : \"3000\"[PDAT])"   
def _mk_pubtype_filter(pub_types: List[str]) -> str:
    pts = [f"\"{p}\"[Publication Type]" for p in (pub_types or []) if p]
    if not pts:
        return ""
    return pts[0] if len(pts) == 1 else f"({' OR '.join(pts)})"


def _mk_lang_filter(lang: str = "English") -> str:
    return f"\"{lang}\"[Language]" if lang else ""


def _collect_pubmed_terms_from_nodes(dag: Dict) -> List[str]:
    id2node = {n.get("id"): n for n in dag.get("nodes", []) if isinstance(n, dict) and n.get("id")}

    def gather_terms(nid: str, seen: Set[str]) -> List[str]:
        if nid in seen:
            return []
        seen.add(nid)
        node = id2node.get(nid)
        if not node:
            return []
        terms = _split_keywords(node.get("keyword", ""))
        for up in (node.get("depends_on") or []):
            terms += gather_terms(up, seen)
        return terms

    terms_all: List[str] = []
    for n in dag.get("nodes", []):
        if str(n.get("tool", "")).lower() == "pubmed":
            terms_all += gather_terms(n["id"], set())
    return _uniq_keep_order([t for t in terms_all if t])


def _normalize_uniprot_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
 
    normed: List[Dict[str, Any]] = []
    for d in docs or []:
        title = (d.get("title") or "").strip()
        url = (d.get("url") or "").strip()
        abstract = (d.get("abstract") or "").strip()

        m = re.match(r"^uniprot:([A-Z0-9]+)\b", title)
        if m:
            normed.append(d)
            continue

        acc = _extract_acc_from_text(title) or _extract_acc_from_text(abstract) or _extract_acc_from_text(url) or None
        if acc:
            title = (f"uniprot:{acc} " + title).strip()
        else:
            title = f"uniprot:UNKNOWN {title}".strip()

        nd = dict(d)
        nd["title"] = title
        normed.append(nd)
    return normed



def _split_keywords(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\;/\|\、]+", s)
    out = []
    for p in parts:
        t = p.strip()
        if t:
            out.append(t)
    return out


def _mk_uniprot_accession_query(acc: str) -> str:
    a = acc.strip()
    return f"(accession:{a} OR accession_id:{a} OR accessions:{a})"


def _extract_acc_from_text(t: str) -> Optional[str]:
    m = re.search(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])\b", t)
    return m.group(1) if m else None

def _uniq_keep_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
def _parse_gene_org_for_uniprot(hint: str) -> Tuple[str, str]:
    s = (hint or "").strip()
    org_matches = list(re.finditer(r"\b([A-Z][a-z]+ [a-z][a-z]+)\b", s))
    organism = org_matches[-1].group(1) if org_matches else "Homo sapiens"

    def _tokens(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9\-]+", text)

    gene = ""
    for paren in re.findall(r"\(([^)]+)\)", s):
        for tok in _tokens(paren):
            if re.search(r"\d", tok) or (tok.isupper() and 2 <= len(tok) <= 12):
                gene = tok
                break
        if gene:
            break
    if not gene:
        for tok in _tokens(s):
            if tok in organism.split(" "):
                continue
            if re.search(r"\d", tok) or (tok.isupper() and 2 <= len(tok) <= 12):
                gene = tok
                break
    return gene, organism


def _mk_uniprot_fielded_query(gene: str, organism: str) -> str:
    g = gene.strip()
    org = organism.strip().replace('"', "")
    q = f"(gene_exact:{g} OR gene:{g})"
    if org.isdigit():
        q += f" AND organism_id:{org}"
    else:
        q += f' AND organism_name:"{org}"'
    return q


def _build_pubmed_queries_from_nodes(
    dag: Dict,
    *,
    last_years: int = 5,
    pub_types: List[str] = ("Review", "Classical Article"),
    language: str = "English",
) -> List[str]:
    terms = _collect_pubmed_terms_from_nodes(dag)
    if not terms:
        return []

    date_f = _mk_date_filter(last_years)
    ptype_f = _mk_pubtype_filter(list(pub_types))
    lang_f = _mk_lang_filter(language)
    filters = [f for f in [ptype_f, date_f, lang_f] if f]

    def attach_filters(qcore: str) -> str:
        parts = [qcore] + filters
        return " AND ".join(p for p in parts if p)


    q1_core = _mk_block(terms, "AND")
    q1 = attach_filters(q1_core) if q1_core else ""


    mid = max(1, len(terms) // 2)
    must_half = _mk_block(terms[:mid], "AND")
    should_half = _mk_block(terms[mid:], "OR")
    q2_core = " AND ".join(p for p in [must_half, should_half] if p)
    q2 = attach_filters(q2_core) if q2_core else ""

 
    comb2: List[str] = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            comb2.append(_mk_block([terms[i], terms[j]], "AND"))
    q3_core = "(" + " OR ".join(comb2[:10]) + ")" if comb2 else _mk_block(terms, "OR")
    q3 = attach_filters(q3_core) if q3_core else ""

    return _uniq_keep_order([q for q in [q1, q2, q3] if q])
