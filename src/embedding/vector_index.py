"""
임베딩 기반 검색 모듈입니다.

이 파일의 역할
1. metadata CSV를 읽습니다.
2. sentence-transformers 임베딩을 생성합니다.
3. FAISS 인덱스를 만듭니다.
4. query 임베딩으로 상위 결과를 검색합니다.
5. PCA 2차원 축소 결과를 만들어 Streamlit 시각화에 사용합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from src.search.keyword_search import build_preview, load_metadata_dataframe


class VectorSearchEngine:
    """
    sentence-transformers + FAISS 기반 검색 엔진입니다.

    cosine similarity 검색을 위해 임베딩을 L2 정규화한 뒤
    IndexFlatIP(내적 검색)를 사용합니다.
    """

    def __init__(self, df: pd.DataFrame, model_name: str):
        self.df = df.reset_index(drop=True).copy()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        embeddings = self.model.encode(
            self.df["search_text"].tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        self.embeddings = self._normalize_embeddings(embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        # 발표용 시각화를 위해 데이터 포인트 PCA를 미리 계산해 둡니다.
        self.pca = PCA(n_components=2, random_state=42)
        self.base_projection = self.pca.fit_transform(self.embeddings)

    @classmethod
    def from_csv(cls, metadata_path: str | Path, model_name: str) -> "VectorSearchEngine":
        df = load_metadata_dataframe(metadata_path)
        return cls(df, model_name=model_name)

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """cosine similarity 계산을 위해 벡터를 정규화합니다."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return embeddings / norms

    def encode_query(self, query: str) -> np.ndarray:
        """질의를 임베딩하고 정규화합니다."""
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        return self._normalize_embeddings(query_embedding)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """임베딩 기반 상위 top_k 검색 결과를 반환합니다."""
        query = (query or "").strip()
        if not query:
            return []

        top_k = max(1, min(top_k, len(self.df)))
        query_embedding = self.encode_query(query)
        scores, indices = self.index.search(query_embedding, top_k)

        results: List[Dict] = []
        for rank, (score, row_index) in enumerate(zip(scores[0], indices[0]), start=1):
            row = self.df.iloc[int(row_index)]
            results.append(
                {
                    "rank": rank,
                    "row_id": int(row_index),
                    "file_name": row["file_name"],
                    "file_path": row["file_path"],
                    "score": float(score),
                    "preview": row["preview"],
                    "full_transcript": row["search_text"],
                }
            )
        return results

    def get_document(self, row_id: int) -> Dict:
        """선택된 결과의 전체 정보를 반환합니다."""
        row = self.df.iloc[row_id]
        return {
            "file_name": row["file_name"],
            "file_path": row["file_path"],
            "processed_txt_path": row["processed_txt_path"],
            "original_transcript": row["original_transcript"],
            "stt_transcript": row["stt_transcript"],
            "search_text": row["search_text"],
        }

    def get_projection_dataframe(
        self,
        query: str,
        highlighted_row_ids: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """
        PCA 2차원 좌표를 Streamlit 시각화용 DataFrame으로 반환합니다.

        - corpus 포인트
        - query 포인트
        - top 결과 강조 플래그
        를 함께 담습니다.
        """
        highlighted_row_ids = set(highlighted_row_ids or [])

        base_df = pd.DataFrame(
            {
                "x": self.base_projection[:, 0],
                "y": self.base_projection[:, 1],
                "label": "corpus",
                "file_name": self.df["file_name"],
                "preview": self.df["preview"],
                "is_top_result": [row_id in highlighted_row_ids for row_id in range(len(self.df))],
                "row_id": list(range(len(self.df))),
            }
        )

        if query.strip():
            query_embedding = self.encode_query(query)
            query_projection = self.pca.transform(query_embedding)
            query_df = pd.DataFrame(
                {
                    "x": [float(query_projection[0, 0])],
                    "y": [float(query_projection[0, 1])],
                    "label": ["query"],
                    "file_name": ["QUERY"],
                    "preview": [build_preview(query, max_chars=120)],
                    "is_top_result": [False],
                    "row_id": [-1],
                }
            )
            return pd.concat([base_df, query_df], ignore_index=True)

        return base_df
