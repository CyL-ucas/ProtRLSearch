# -*- coding: utf-8 -*-
import json
import logging
import re
import requests
import time
from typing import List, Dict, Optional, Any

UNIPROT_SEARCH_URL  = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_BLAST_BASE  = "https://rest.uniprot.org/blast"
NCBI_BLAST          = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

class UniProtSearcher:
    def __init__(self, timeout: int = 12, topn: int = 20, blast_base: Optional[str] = None):

        self.timeout = int(timeout)
        self.topn = int(topn)
        self.blast_base = (blast_base or UNIPROT_BLAST_BASE).rstrip("/")


    def search_docs(
        self,
        query: str,
        *,
        topk: Optional[int] = None,
        base_url: Optional[str] = None
    ) -> List[Dict[str, str]]:

        url = (base_url or UNIPROT_SEARCH_URL)
        k = int(topk or self.topn)

        preferred_fields = ",".join([
            "accession", "id", "protein_name", "organism_name",
            "gene_primary", "sequence_length"
        ])

        def _request(params):
            r = requests.get(url, params=params, timeout=self.timeout,
                             headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()

 
        params1 = {"query": query, "size": str(k), "fields": preferred_fields, "format": "json"}
        try:
            js = _request(params1)
        except requests.HTTPError as ex:
     
            if ex.response is not None and ex.response.status_code == 400:
                logging.warning("[WARN] UniProt search 400; retry without fields")
                params2 = {"query": query, "size": str(k), "format": "json"}
                try:
                    js = _request(params2)
                except Exception as ex2:
                    logging.warning(f"[WARN] UniProt search failed again: {ex2}")
                    return []
            else:
                logging.warning(f"[WARN] UniProt search error: {ex}")
                return []
        except Exception as ex:
            logging.warning(f"[WARN] UniProt search error: {ex}")
            return []

        entries = (js.get("results") or []) if isinstance(js, dict) else []
        docs: List[Dict[str, str]] = []
        for e in entries[:k]:
            acc = e.get("primaryAccession") or e.get("accession") or ""
            pid = e.get("uniProtkbId") or e.get("id") or ""
    
            pname = ""
            pd = (e.get("proteinDescription") or {})
            if isinstance(pd, dict):
                rn = pd.get("recommendedName") or {}
                if isinstance(rn, dict):
                    pname = (rn.get("fullName") or {}).get("value") or ""
            org = ((e.get("organism") or {}).get("scientificName")
                   or (e.get("organism") or {}).get("commonName") or "")
      
            gene_pref = e.get("genePrimary") or ""
            if not gene_pref:
                genes = (e.get("genes") or [])
                if genes:
                    g0 = genes[0] or {}
                    gene_pref = (g0.get("geneName", {}) or {}).get("value") or ""
            length = (e.get("sequence") or {}).get("length") or e.get("sequenceLength") or ""

            title = f"{acc}{(' / ' + pid) if pid else ''} — {pname}".strip()
            abstract = f"gene={gene_pref} | organism={org} | length={length}"
            url_entry = f"https://www.uniprot.org/uniprotkb/{acc}" if acc else ""
            docs.append({"title": title, "abstract": abstract, "url": url_entry})
        return docs

 
    def _looks_like_protein(self, seq: str) -> bool:
        s = re.sub(r"\s+", "", (seq or "").upper())
        if not s:
            return False
        aa = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
        nt = set("ACGTU")
        non_nt = sum(1 for ch in s if ch not in nt)
        return non_nt / max(len(s), 1) >= 0.4

    def _fmt_pct(self, x):
        try:
            return f"{float(x):.1f}"
        except Exception:
            return str(x) if x is not None else "NA"

    def _fmt_e(self, x):
        try:
            return f"{float(x):.1e}"
        except Exception:
            return str(x) if x is not None else "NA"

    def _fmt_int(self, x):
        try:
            return str(int(x))
        except Exception:
            return str(x) if x is not None else "NA"

    def _blast_run(self, sequence: str, program: Optional[str], database: str, taxid: Optional[str]) -> Optional[str]:
 
        pgm = (program or "").lower().strip()
        if not pgm:
            pgm = "blastp" if self._looks_like_protein(sequence) else "blastx"

        bases = [
            self.blast_base,                        
            (self.blast_base.rstrip("/") + "/"),     
        ]
        paths = ["run", "run/"]                     

        data = {"sequence": sequence, "program": pgm, "database": database}
        if taxid:
            data["taxId"] = str(taxid)

        last_err = None
        for b in bases:
            b = b.rstrip("/")
            for p in paths:
                url = f"{b}/{p}"
                try:
                    rr = requests.post(url, data=data, timeout=self.timeout,
                                       headers={"Accept": "text/plain"})
                    rr.raise_for_status()
                    job = rr.text.strip()
                    if job:
                        return job
                except Exception as ex:
                    last_err = ex
                    logging.warning(f"[WARN] uniprot_blast run try failed: {url} -> {ex}")
                    continue

        logging.warning(f"[WARN] uniprot_blast run error after retries: {last_err}")
        return None

    def _blast_poll(self, job: str, max_wait_s: int = 120, poll_every_s: float = 2.0) -> bool:
        status_url = f"{self.blast_base}/status/{job}"
        t0 = time.time()
        status = "RUNNING"
        while time.time() - t0 < max_wait_s:
            try:
                rs = requests.get(status_url, timeout=self.timeout, headers={"Accept": "text/plain"})
                rs.raise_for_status()
                status = (rs.text or "").strip().upper()
                if status == "FINISHED":
                    return True
                if status in {"ERROR", "FAILED", "NOT_FOUND"}:
                    logging.warning(f"[WARN] uniprot_blast status={status}")
                    return False
            except Exception as ex:
                logging.warning(f"[WARN] uniprot_blast poll error: {ex}")
                break
            time.sleep(poll_every_s)
        logging.warning(f"[WARN] uniprot_blast timeout or not finished (status={status})")
        return False

    def _blast_results(self, job: str) -> List[Dict[str, Any]]:
        res_url = f"{self.blast_base}/results/{job}"
        try:
            rj = requests.get(res_url, params={"format": "json"}, timeout=self.timeout, headers={"Accept": "application/json"})
            rj.raise_for_status()
            js = rj.json() or {}
        except Exception as ex:
            logging.warning(f"[WARN] uniprot_blast results error: {ex}")
            return []

        raw_hits = (js.get("hits") or js.get("results") or [])
        hits: List[Dict[str, Any]] = []
        for h in raw_hits:
            acc = h.get("accession") or h.get("id") or h.get("accessionId") or ""
            desc = (h.get("description") or h.get("proteinName") or h.get("hitDescription") or "").strip()
            ident  = h.get("identity") or h.get("identityPercent") or h.get("percentageIdentity")
            evalue = h.get("evalue") or h.get("expectValue")
            alnlen = h.get("alignmentLength") or h.get("alnLen")

            hsps = h.get("hsps") or []
            if hsps and (ident is None or evalue is None or alnlen is None):
                def _hsp_key(x):
                    try:
                        evf = float(x.get("evalue"))
                    except Exception:
                        evf = 1e9
                    try:
                        idf = float(x.get("identity") or x.get("identityPercent") or -1.0)
                    except Exception:
                        idf = -1.0
                    return (evf, -idf)
                best = sorted(hsps, key=_hsp_key)[0]
                ident  = ident  if ident  is not None else (best.get("identity") or best.get("identityPercent"))
                evalue = evalue if evalue is not None else best.get("evalue")
                alnlen = alnlen if alnlen is not None else (best.get("alignLen") or best.get("alignmentLength"))

            try:
                se = float(evalue)
            except Exception:
                se = 1e9
            try:
                si = float(ident)
            except Exception:
                si = -1.0

            hits.append({
                "acc": acc,
                "desc": desc,
                "identity": ident, "evalue": evalue, "alnlen": alnlen,
                "_sort_e": se, "_sort_i": si
            })

        hits.sort(key=lambda d: (d.get("_sort_e", 1e9), -d.get("_sort_i", -1.0)))
        return hits

  
    def blast_docs_via_ncbi(
        self,
        sequence: str,
        *,
        program: Optional[str] = None,
        database: str = "swissprot",
        topk: Optional[int] = None,
        expect: float = 0.001,
        hitlist_size: int = 50,
        max_wait_s: int = 120,
        poll_every_s: float = 2.0,
    ) -> List[Dict[str, str]]:
    
        seq = (sequence or "").strip()
        if not seq:
            return []

        pgm = (program or ("blastp" if self._looks_like_protein(seq) else "blastx")).lower()

        headers = {
            "User-Agent": "BioReason-UniProt/1.0 (+https://example.org)"
        }

    
        put_data = {
            "CMD": "Put",
            "PROGRAM": pgm,
            "DATABASE": database,       
            "QUERY": seq,
            "EXPECT": str(expect),
            "HITLIST_SIZE": str(hitlist_size),
            "FORMAT_TYPE": "XML",
        }

        last_exc = None
        for _ in range(3):
            try:
                r = requests.post(NCBI_BLAST, data=put_data, timeout=self.timeout, headers=headers)
                r.raise_for_status()
                break
            except Exception as ex:
                last_exc = ex
                time.sleep(1.5)
        else:
            logging.warning(f"[WARN] NCBI PUT failed after retries: {last_exc}")
            return []

        text = r.text or ""
        m_rid = re.search(r"RID\s*=\s*([A-Z0-9\-]+)", text)
        if not m_rid:
     
            logging.warning("[WARN] NCBI PUT ok but RID not found; raw head: " + (text[:200].replace("\n", " ")))
            return []
        rid = m_rid.group(1)


        t0 = time.time()
        while True:
            if time.time() - t0 > max_wait_s:
                logging.warning("[WARN] NCBI poll timeout")
                return []

            time.sleep(max(0.5, poll_every_s))
            try:
                rr = requests.get(
                    NCBI_BLAST,
                    params={"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"},
                    timeout=self.timeout,
                    headers=headers
                )
                rr.raise_for_status()
            except Exception as ex:
                logging.warning(f"[WARN] NCBI poll error: {ex}")
                continue

            info = rr.text or ""
            if "Status=READY" in info:
        
                break
            if "Status=FAILED" in info or "Status=UNKNOWN" in info:
                logging.warning(f"[WARN] NCBI status failed/unknown; info head: {(info[:200].replace(chr(10),' '))}")
                return []

   
        try:
            rx = requests.get(
                NCBI_BLAST,
                params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"},
                timeout=self.timeout,
                headers=headers
            )
            rx.raise_for_status()
        except Exception as ex:
            logging.warning(f"[WARN] NCBI get XML error: {ex}")
            return []
        xml = rx.text or ""


        docs: List[Dict[str, str]] = []
        for hit in re.findall(r"<Hit>(.*?)</Hit>", xml, flags=re.S):
            acc_m  = re.search(r"<Hit_accession>([^<]+)</Hit_accession>", hit)
            def_m  = re.search(r"<Hit_def>([^<]+)</Hit_def>", hit)
            if not acc_m:
                continue
            acc = acc_m.group(1).strip()
            desc = (def_m.group(1).strip() if def_m else "")

            best_hsp = None
            best_key = (1e9, -1.0)  # (evalue, -bit_score)
            for hsp in re.findall(r"<Hsp>(.*?)</Hsp>", hit, flags=re.S):
                ev_m  = re.search(r"<Hsp_evalue>([^<]+)</Hsp_evalue>", hsp)
                bs_m  = re.search(r"<Hsp_bit-score>([^<]+)</Hsp_bit-score>", hsp)
                id_m  = re.search(r"<Hsp_identity>(\d+)</Hsp_identity>", hsp)
                al_m  = re.search(r"<Hsp_align-len>(\d+)</Hsp_align-len>", hsp)

                try:
                    ev = float(ev_m.group(1)) if ev_m else 1e9
                except Exception:
                    ev = 1e9
                try:
                    bs = float(bs_m.group(1)) if bs_m else -1.0
                except Exception:
                    bs = -1.0

                key = (ev, -bs)
                if key < best_key:
                    best_key = key
                    best_hsp = {"id_m": id_m, "al_m": al_m, "ev_m": ev_m}


            if best_hsp:
                ident = best_hsp["id_m"].group(1) if best_hsp["id_m"] else None
                alen  = best_hsp["al_m"].group(1) if best_hsp["al_m"] else None
                evalue = best_hsp["ev_m"].group(1) if best_hsp["ev_m"] else None
            else:
                ident = alen = evalue = None

   
            ident_pct = "NA"
            try:
                if ident and alen and int(alen) > 0:
                    ident_pct = f"{(int(ident) / int(alen)) * 100:.1f}"
            except Exception:
                pass

            title = f"{acc} — identity={ident_pct}%, evalue={evalue if evalue else 'NA'}, aln_len={alen if alen else 'NA'}"
  
            url   = f"https://www.uniprot.org/uniprotkb/{acc}"
            docs.append({"title": title, "abstract": desc, "url": url})

        k = int(topk or self.topn)
        return docs[:k]


    def blast_docs(
        self,
        sequence: str,
        *,
        program: Optional[str] = None,
        database: str = "uniprotkb",
        taxid: Optional[str] = None,
        max_wait_s: int = 120,
        poll_every_s: float = 2.0,
        topk: Optional[int] = None
    ) -> List[Dict[str, str]]:
        seq = (sequence or "").strip()
        if not seq:
            return []

        job = self._blast_run(seq, program, database, taxid)
        if not job:
            return []

        if not self._blast_poll(job, max_wait_s=max_wait_s, poll_every_s=poll_every_s):
            return []

        k = int(topk or self.topn)
        hits = self._blast_results(job)[:k]
        docs: List[Dict[str, str]] = []
        for h in hits:
            acc = h.get("acc") or ""
            ident_s  = self._fmt_pct(h.get("identity"))
            evalue_s = self._fmt_e(h.get("evalue"))
            alnlen_s = self._fmt_int(h.get("alnlen"))
            title    = f"{acc or 'UNKNOWN'} — identity={ident_s}%, evalue={evalue_s}, aln_len={alnlen_s}"
            url      = f"https://www.uniprot.org/uniprotkb/{acc}" if acc else ""
            docs.append({
                "title": title,
                "abstract": (h.get("desc") or "").strip(),
                "url": url
            })
        return docs


