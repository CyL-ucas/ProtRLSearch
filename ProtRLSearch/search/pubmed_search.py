# bioreason/search_tools/pubmed_search.py
# -*- coding: utf-8 -*-

import os
import re
import datetime
import urllib.parse
from typing import Any, Dict, List, Optional
from Bio import Entrez

from bioreason.core.utils import eprint


PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
EMAIL = os.getenv("ENTREZ_EMAIL", "")  # 必须填，Entrez 要求


class PubMedSearcher:
    BASE_URL = "https://pubmed.ncbi.nlm.nih.gov/"

    def __init__(self, api_key: str = PUBMED_API_KEY, email: str = EMAIL):
        self.api_key = api_key
        self.email = email
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key

   
    def search_pubmed_url(self, query: str) -> str:
 
        encoded = urllib.parse.quote(query)
        return f"{self.BASE_URL}?term={encoded}"

    def _efetch_xml(self, ids: List[str]):
        """
        
        """
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
            record = Entrez.read(handle)
            handle.close()
            return record
        except Exception as e:
            raise RuntimeError(f"efetch error: {e}")

    def search_pubmed(self, query: str, max_results: int = 50) -> Optional[List[Dict[str, Any]]]:
   
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, retmode="xml")
            record = Entrez.read(handle)
            handle.close()

            ids = list(record.get("IdList", []))
            if not ids:
                eprint(f"[PubMed] No results for query='{query}'")
                return None

            fetched = self._efetch_xml(ids)
            articles = fetched.get("PubmedArticle", [])
            results = []
            for art in articles:
                med = art.get("MedlineCitation", {})
                art_info = med.get("Article", {})
                pubmed_data = art.get("PubmedData", {})
                title = str(art_info.get("ArticleTitle", "")).strip()
                abstract = self._parse_abstract(art_info.get("Abstract"))
                authors = self._parse_authors(art_info.get("AuthorList", []))
                pub_year = self._parse_pubyear(art_info, med)
                doi = self._parse_doi(pubmed_data, art_info)
                pmid = str(med.get("PMID", ""))
                url = f"{self.BASE_URL}{pmid}/" if pmid else self.search_pubmed_url(title or query)
                results.append({
                    "title": title, "abstract": abstract,
                    "authors": authors, "pubdate": pub_year,
                    "doi": doi, "url": url, "source": "PubMed"
                })
            return results or None
        except Exception as e:
            eprint(f" PubMed query failed: {e}")
            return None

    def search_from_dag(self, dag_or_text, max_results: int = 50) -> Optional[List[Dict[str, Any]]]:

        if isinstance(dag_or_text, dict) and dag_or_text.get("nodes"):
            keywords = []
            for n in dag_or_text.get("nodes", []):
                if str(n.get("tool", "")).lower() == "pubmed":
                    kw = n.get("keyword", "")
                    if kw:
                        keywords.append(kw)
            query = " ".join(keywords) if keywords else dag_or_text.get("_raw_prompt", "")
        else:
            query = str(dag_or_text)

        return self.search_pubmed(query, max_results=max_results)

    @staticmethod
    def _parse_abstract(abs_obj: Optional[Dict[str, Any]]) -> str:
        if not abs_obj:
            return ""
        texts = abs_obj.get("AbstractText", [])
        if isinstance(texts, str):
            return texts.strip()
        out_parts: List[str] = []
        for item in texts:
            try:
                if isinstance(item, str):
                    part = item
                    label = ""
                else:
                    part = str(item)
                    label = ""
                    try:
                        label = item.attributes.get("Label", "") or item.attributes.get("NlmCategory", "")
                    except Exception:
                        label = ""
                part = (part or "").strip()
                if not part:
                    continue
                if label:
                    out_parts.append(f"{label}: {part}")
                else:
                    out_parts.append(part)
            except Exception:
                continue
        return "\n".join(out_parts).strip()

    @staticmethod
    def _parse_authors(author_list: List[Dict[str, Any]]) -> List[str]:
        out = []
        for a in author_list or []:
            try:
                last = a.get("LastName", "") or ""
                fore = a.get("ForeName", "") or a.get("Initials", "") or ""
                name = (f"{fore} {last}").strip()
                if name:
                    out.append(name)
            except Exception:
                continue
        return out

    @staticmethod
    def _parse_pubyear(art_info: Dict[str, Any], med: Dict[str, Any]) -> str:
        try:
            ad_list = art_info.get("ArticleDate", [])
            if ad_list:
                y = ad_list[0].get("Year")
                if y:
                    return str(y)
        except Exception:
            pass
        try:
            pubdate = art_info.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            for key in ("Year", "MedlineDate"):
                if key in pubdate:
                    return str(pubdate[key])
        except Exception:
            pass
        try:
            dc = med.get("DateCompleted", {})
            if "Year" in dc:
                return str(dc["Year"])
        except Exception:
            pass
        return "Unknown"

    @staticmethod
    def _parse_doi(pubmed_data: Dict[str, Any], art_info: Dict[str, Any]) -> str:
        try:
            for aid in pubmed_data.get("ArticleIdList", []):
                try:
                    if getattr(aid, "attributes", {}).get("IdType") == "doi":
                        return str(aid)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            for eid in art_info.get("ELocationID", []):
                try:
                    if getattr(eid, "attributes", {}).get("EIdType") == "doi":
                        return str(eid)
                except Exception:
                    pass
        except Exception:
            pass
        return ""

    @staticmethod
    def _split_keywords(s: str) -> List[str]:
        return [t.strip() for t in re.split(r"[,\;/\|\、]+", s or "") if t.strip()]

    @staticmethod
    def _mk_block(terms: List[str], op: str, field: str = "Title/Abstract") -> str:
        if not terms:
            return ""
        if len(terms) == 1:
            return f"\"{terms[0]}\"[{field}]"
        return "(" + f" {op} ".join(f"\"{t}\"[{field}]" for t in terms) + ")"

    @staticmethod
    def _mk_date_filter(last_years: int) -> str:
        if last_years <= 0:
            return ""
        start = datetime.datetime.now().year - last_years
        return f"(\"{start}/01/01\"[PDAT] : \"3000\"[PDAT])"

    @staticmethod
    def _mk_pubtype_filter(pub_types: List[str]) -> str:
        pts = [f"\"{p}\"[Publication Type]" for p in pub_types]
        return "" if not pts else (pts[0] if len(pts) == 1 else f"({' OR '.join(pts)})")

    @staticmethod
    def _mk_lang_filter(lang: str = "English") -> str:
        return f"\"{lang}\"[Language]" if lang else ""

    @staticmethod
    def _uniq_keep_order(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    @staticmethod
    def _collect_terms_from_dag(dag) -> List[str]:

        if isinstance(dag, dict):
            nodes = dag.get("nodes", [])
        elif isinstance(dag, list):
            nodes = dag
        else:
            return []

        id2node = {n.get("id"): n for n in nodes if isinstance(n, dict)}
        terms = []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            kw = n.get("keyword")
            if kw:
                terms.append(kw.strip())
        return terms



if __name__ == "__main__":
    searcher = PubMedSearcher()
    query = "cyp17a1 zebrafish knockout"

 