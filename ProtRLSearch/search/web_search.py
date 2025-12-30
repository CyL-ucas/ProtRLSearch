# brave_pubmed_reranker.py
import os
import time
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


class BravePubMedReranker:


    BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        brave_api_key: Optional[str] = None,
        model_name: str = "NeuML/pubmedbert-base-embeddings",
        device: Optional[str] = None,
        timeout: float = 15.0,
        sleep: float = 1.0,
        verbose: bool = True,
        session: Optional[requests.Session] = None,
    ):

        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY", "")
        self.model_name = model_name
        self.device = device
        self.timeout = timeout
        self.sleep = sleep
        self.verbose = verbose

        self._session = session or requests.Session()
        self._embedder: Optional[SentenceTransformer] = None

 
    def search(
        self,
        query: str,
        max_results: int = 20,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
   
        """
        if not query or not query.strip():
            return []

        docs = self._brave_search(query, max_results=max_results)
  

        docs = self._dedup_by_url(docs)
       

        ranked = self._rerank_by_pubmedbert(query, docs, top_k=top_k)

        if self.verbose:
           
            for i, r in enumerate(ranked, 1):
                print(f"[{i}] {r['title']}  (brave)")
                print(f"    URL: {r['href']}")
                print(f"    score: {r['score']:.4f}")

        return ranked

    
    def _brave_search(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key,
        }
        params = {"q": query, "count": max_results}

        try:
            if self.sleep > 0:
                time.sleep(self.sleep)
            resp = self._session.get(self.BRAVE_API_URL, headers=headers, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            if self.verbose:
                print(f"")
            return []

        results: List[Dict[str, Any]] = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title"),
                "href": item.get("url"),
                "body": item.get("description"),
                "source": "brave",
            })
        return results


    @staticmethod
    def _dedup_by_url(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for d in docs:
            url = (d.get("href") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(d)
        return out


    def _load_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name, device=self.device)
          
        return self._embedder

    @staticmethod
    def _concat_text(doc: Dict[str, Any]) -> str:
        title = (doc.get("title") or "").strip()
        body = (doc.get("body") or "").strip()
        return f"{title}. {body}".strip() if body else title

    def _rerank_by_pubmedbert(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if not docs:
            return []

        embedder = self._load_embedder()
        texts = [self._concat_text(d) for d in docs]

       
        q_vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)  # [1, D]
        d_vecs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)  # [N, D]

        sims = (d_vecs @ q_vec.T).reshape(-1)  
        k = min(top_k, len(docs))
        idx = np.argsort(-sims)[:k]

        ranked: List[Dict[str, Any]] = []
        for i in idx:
            item = dict(docs[i])
            item["score"] = float(sims[i])
            ranked.append(item)
        return ranked



if __name__ == "__main__":
    