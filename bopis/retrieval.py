# retrieval.py
"""Reference matching over the fitted Q/A pool.

Standard library only at import time: numpy, scikit-learn and
sentence-transformers are pulled in lazily through ``importlib`` inside the
functions that need them, the same convention the rest of the core uses so the
measurement core stays stdlib-clean (Amendment A-6). If they are absent the
retriever degrades to a word-overlap fallback rather than failing to import.
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

np = None
TfidfVectorizer = None
_numeric_ready = False


def _ensure_numeric() -> bool:
    """Load numpy + the TF-IDF vectorizer on first use, once."""
    global np, TfidfVectorizer, _numeric_ready
    if not _numeric_ready:
        _numeric_ready = True
        try:
            np = importlib.import_module("numpy")
            TfidfVectorizer = importlib.import_module(
                "sklearn.feature_extraction.text"
            ).TfidfVectorizer
        except Exception:
            np = None
            TfidfVectorizer = None
    return np is not None

from bopis.dataset import (
    Prompt,
    SOURCE_CNN,
    SOURCE_NQ,
    SOURCE_SQUAD,
    SOURCE_TRIVIA,
    load_cnn_dailymail,
    load_natural_questions,
    load_squad_v2,
    load_trivia_qa,
)

DEFAULT_POOL_PATH = os.path.join("data", "reference_pool.json")
DEFAULT_EMB_PATH = os.path.join("data", "reference_embeddings.pkl")
DEFAULT_FITTED_PATH = os.path.join("data", "fitted_references.json")

@dataclass
class ReferenceMatch:
    reference: str
    similarity: float
    matched_question: str
    matched_source: str
    match_index: int

class ReferenceRetriever:
    def __init__(
        self,
        pool_path: str = DEFAULT_POOL_PATH,
        emb_path: str = DEFAULT_EMB_PATH,
        min_similarity: float = 0.5,
        tfidf_accept_similarity: float = 0.1,
        model_name: str = 'all-MiniLM-L6-v2',
        use_sentence_transformers: bool = True,
    ) -> None:
        self.pool_path = pool_path
        self.emb_path = emb_path
        self.min_similarity = min_similarity
        self.tfidf_accept_similarity = tfidf_accept_similarity
        self.model_name = model_name
        self.use_sentence_transformers = use_sentence_transformers
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.sources: List[str] = []
        self.embeddings = None
        self._vectorizer = None
        self._st_model = None
        self._backend = 'none'

    @property
    def backend(self) -> str:
        """Which matching backend is in use: sentence-transformers, tfidf, etc."""
        return self._backend

    @property
    def accept_similarity(self) -> float:
        """Accept threshold for a match, in this backend's similarity scale.

        Embedding cosines cluster high, so 0.5 is right there. TF-IDF cosine on
        short queries against full questions rarely clears 0.5 even for a good
        match, so it accepts lower -- the caller still flags anything under 0.5
        as low-confidence rather than silently using it (the design's rule).
        """
        return self.tfidf_accept_similarity if self._backend == 'tfidf' else self.min_similarity

    def _ensure_st(self) -> bool:
        if not self.use_sentence_transformers:
            return False
        try:
            st = importlib.import_module("sentence_transformers")
            if self._st_model is None:
                self._st_model = st.SentenceTransformer(self.model_name)
            return True
        except Exception:
            return False

    def build_pool(
        self,
        seed: int = 20260101,
        include_external: bool = True,
        include_dolly: Optional[Sequence[Prompt]] = None,
        max_per_dataset: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[str, str, str]]:
        pairs: List[Tuple[str, str, str]] = []
        mp = max_per_dataset or {}
        if include_external:
            pairs.extend((p.instruction.strip(), p.response.strip(), p.source) for p in load_natural_questions(seed, n=int(mp.get(SOURCE_NQ, 60))) if p.instruction.strip() and p.response.strip())
            pairs.extend((p.instruction.strip(), p.response.strip(), p.source) for p in load_trivia_qa(seed, n=int(mp.get(SOURCE_TRIVIA, 60))) if p.instruction.strip() and p.response.strip())
            pairs.extend((p.instruction.strip(), p.response.strip(), p.source) for p in load_cnn_dailymail(seed, n=int(mp.get(SOURCE_CNN, 50))) if p.instruction.strip() and p.response.strip())
            pairs.extend((p.instruction.strip(), p.response.strip(), p.source) for p in load_squad_v2(seed, n=int(mp.get(SOURCE_SQUAD, 30))) if p.instruction.strip() and p.response.strip())
        if include_dolly:
            for p in include_dolly:
                q = p.instruction.strip(); a = p.response.strip()
                if q and a: pairs.append((q,a,p.source))
        seen=set(); uniq=[]
        for q,a,s in pairs:
            k=(q.lower(),a.lower())
            if k in seen: continue
            seen.add(k); uniq.append((q,a,s))
        self.questions=[q for q,_,_ in uniq]; self.answers=[a for _,a,_ in uniq]; self.sources=[s for _,_,s in uniq]
        os.makedirs(os.path.dirname(self.pool_path), exist_ok=True)
        with open(self.pool_path,'w',encoding='utf-8') as f:
            json.dump([{'question':q,'answer':a,'source':s} for q,a,s in uniq], f, ensure_ascii=False, indent=2)
        self._embed()
        return uniq

    def load_pool(self) -> bool:
        if not os.path.exists(self.pool_path):
            return False
        try:
            with open(self.pool_path,'r',encoding='utf-8') as f:
                items = json.load(f)
            self.questions=[it['question'] for it in items]; self.answers=[it['answer'] for it in items]; self.sources=[it.get('source','') for it in items]
        except Exception:
            return False
        if self._load_embeddings():
            return True
        # Pool exists but the embedding cache does not (or was built with a
        # different backend): re-embed now rather than report an empty pool.
        self._embed()
        return bool(self.questions)

    def _embed(self):
        if len(self.questions)==0:
            self.embeddings=None; self._backend='empty'; return
        _ensure_numeric()
        if self._ensure_st() and np is not None:
            try:
                self.embeddings=self._st_model.encode(self.questions, normalize_embeddings=True, convert_to_numpy=True)
                self._backend='sentence-transformers'; self._save_embeddings(); return
            except Exception: pass
        if TfidfVectorizer is not None and np is not None:
            self._vectorizer=TfidfVectorizer(lowercase=True, ngram_range=(1,2))
            self.embeddings=self._vectorizer.fit_transform(self.questions).toarray()
            norms=np.linalg.norm(self.embeddings, axis=1, keepdims=True); norms[norms==0]=1.0
            self.embeddings=self.embeddings/norms
            self._backend='tfidf'; self._save_embeddings(); return
        self.embeddings=None; self._backend='fallback'

    def _save_embeddings(self):
        if self.embeddings is None: return
        try:
            os.makedirs(os.path.dirname(self.emb_path), exist_ok=True)
            with open(self.emb_path,'wb') as f:
                pickle.dump({'backend':self._backend,'embeddings':self.embeddings,'vectorizer':self._vectorizer}, f)
        except Exception: pass

    def _load_embeddings(self) -> bool:
        if not os.path.exists(self.emb_path):
            self._embed(); return self.embeddings is not None
        try:
            with open(self.emb_path,'rb') as f:
                data=pickle.load(f)
            self._backend=data.get('backend','unknown'); self.embeddings=data.get('embeddings'); self._vectorizer=data.get('vectorizer')
            if self._backend=='sentence-transformers' and self.use_sentence_transformers: self._ensure_st()
            return self.embeddings is not None and len(self.embeddings)==len(self.questions)
        except Exception:
            self._embed(); return self.embeddings is not None

    def _tfidf_match(self,q:str,thr:float)->Tuple[int,float]:
        _ensure_numeric()
        if self._vectorizer is None or self.embeddings is None or np is None: return -1,-1.0
        qv=self._vectorizer.transform([q]).toarray()
        qv=qv/(np.linalg.norm(qv)+1e-12)
        sims=self.embeddings @ qv[0]
        best_idx=int(np.argmax(sims)); best_sim=float(sims[best_idx]); return best_idx,best_sim

    def find_match(self, query:str, min_similarity: Optional[float]=None) -> Optional[ReferenceMatch]:
        if not query or len(self.questions)==0: return None
        _ensure_numeric()
        thr=min_similarity if min_similarity is not None else self.accept_similarity
        q=query.strip()
        if self._backend=='sentence-transformers' and self._st_model is not None and np is not None:
            try:
                qv=self._st_model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
                sims=self.embeddings @ qv
                best_idx=int(np.argmax(sims)); best_sim=float(sims[best_idx])
            except Exception:
                best_idx,best_sim=self._tfidf_match(q,thr)
        elif self._backend=='tfidf' and self._vectorizer is not None and np is not None:
            best_idx,best_sim=self._tfidf_match(q,thr)
        else:
            best_idx,best_sim=-1,-1.0; qlow=q.lower()
            for i,qq in enumerate(self.questions):
                qq=qq.lower()
                if not qq: continue
                qa=set(qlow.split()); qb=set(qq.split()); inter=len(qa&qb); union=len(qa|qb); sim=inter/max(1,union)
                if sim>best_sim: best_sim=sim; best_idx=i
        if best_idx<0 or best_sim<thr: return None
        return ReferenceMatch(reference=self.answers[best_idx], similarity=best_sim, matched_question=self.questions[best_idx], matched_source=self.sources[best_idx] if best_idx<len(self.sources) else '', match_index=best_idx)



    def build_from_fitted(
        self,
        fitted_path: str = DEFAULT_FITTED_PATH,
        include_in_proxy: bool = True,
    ) -> List[Tuple[str, str, str]]:
        if not os.path.exists(fitted_path):
            raise FileNotFoundError(f'fitted references not found: {fitted_path}')
        with open(fitted_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
        pairs: List[Tuple[str, str, str]] = []
        for it in items:
            if not include_in_proxy and it.get('in_proxy'):
                continue
            q = (it.get('question') or '').strip()
            a = (it.get('answer') or '').strip()
            if q and a:
                pairs.append((q, a, it.get('source', 'fitted')))
        seen = set(); uniq=[]
        for q,a,s in pairs:
            k = (q.lower(), a.lower())
            if k in seen: continue
            seen.add(k); uniq.append((q,a,s))
        self.questions=[q for q,_,_ in uniq]; self.answers=[a for _,a,_ in uniq]; self.sources=[s for _,_,s in uniq]
        os.makedirs(os.path.dirname(self.pool_path), exist_ok=True)
        with open(self.pool_path,'w',encoding='utf-8') as f:
            json.dump([{'question':q,'answer':a,'source':s} for q,a,s in uniq], f, ensure_ascii=False, indent=2)
        self._embed()
        return uniq


def get_default_retriever() -> ReferenceRetriever:
    r=ReferenceRetriever()
    r.load_pool()
    return r
