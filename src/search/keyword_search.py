"""
키워드 기반 검색 모듈입니다.

이 파일의 역할
1. metadata CSV를 읽습니다.
2. 검색에 사용할 텍스트 컬럼을 정리합니다.
3. TF-IDF 기반 키워드 검색 인덱스를 만듭니다.
4. query를 받아 상위 결과를 반환합니다.

초보자도 이해하기 쉽게, 복잡한 전처리 대신
한국어에서 비교적 튼튼한 char n-gram TF-IDF 방식을 사용합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DUMMY_TRANSCRIPT_PATH = PROJECT_ROOT / "data" / "metadata" / "dummy_search_transcripts.csv"


def format_portable_path(path: Path) -> str:
    """프로젝트 내부 경로는 상대경로로 저장해 배포 환경에서도 재사용합니다."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def resolve_project_path(
    path_value: str,
    fallback_dir: Path | None = None,
    fallback_name: str = "",
) -> Path | None:
    """
    metadata에 들어 있는 절대/상대 경로를 현재 프로젝트 기준으로 다시 해석합니다.

    기존 Windows 절대경로가 남아 있어도 같은 파일명이 data 폴더에 있으면
    현재 프로젝트 경로로 치환할 수 있게 만듭니다.
    """
    value = str(path_value or "").strip()
    candidates: List[Path] = []

    if value:
        raw_path = Path(value)
        candidates.append(raw_path)

        if not raw_path.is_absolute():
            candidates.append(PROJECT_ROOT / raw_path)

        if fallback_dir is not None and raw_path.name:
            candidates.append(fallback_dir / raw_path.name)

    if fallback_dir is not None and fallback_name:
        candidates.append(fallback_dir / fallback_name)

    if not candidates:
        return None

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]


def normalize_metadata_paths(df: pd.DataFrame) -> pd.DataFrame:
    """metadata의 파일 경로를 프로젝트 기준 portable path로 정리합니다."""
    normalized_df = df.copy()
    file_paths: List[str] = []
    processed_paths: List[str] = []

    for row in normalized_df.itertuples(index=False):
        file_name = str(getattr(row, "file_name", "") or "").strip()
        processed_name = f"{Path(file_name).stem}.txt" if file_name else ""

        file_path = resolve_project_path(
            getattr(row, "file_path", ""),
            fallback_dir=RAW_DIR,
            fallback_name=file_name,
        )
        processed_txt_path = resolve_project_path(
            getattr(row, "processed_txt_path", ""),
            fallback_dir=PROCESSED_DIR,
            fallback_name=processed_name,
        )

        file_paths.append(format_portable_path(file_path) if file_path else "")
        processed_paths.append(format_portable_path(processed_txt_path) if processed_txt_path else "")

    normalized_df["file_path"] = file_paths
    normalized_df["processed_txt_path"] = processed_paths
    return normalized_df


def build_metadata_from_dummy_transcripts(dummy_csv_path: Path) -> pd.DataFrame:
    """dummy transcript CSV를 검색용 metadata 형식으로 변환합니다."""
    if not dummy_csv_path.exists():
        raise FileNotFoundError(f"dummy transcript CSV를 찾을 수 없습니다: {dummy_csv_path}")

    df = pd.read_csv(dummy_csv_path, encoding="utf-8-sig").fillna("")
    if df.empty:
        raise ValueError(f"dummy transcript CSV가 비어 있습니다: {dummy_csv_path}")

    if "transcript" not in df.columns:
        raise ValueError(f"dummy transcript CSV에 transcript 컬럼이 없습니다: {dummy_csv_path}")

    df = df.reset_index(drop=True).copy()
    df["file_name"] = [f"audio_{index + 1:03d}.wav" for index in range(len(df))]
    df["file_path"] = ""
    df["processed_txt_path"] = ""
    df["original_transcript"] = df["transcript"].astype(str).str.strip()
    df["stt_transcript"] = ""
    return df[
        [
            "file_name",
            "file_path",
            "processed_txt_path",
            "original_transcript",
            "stt_transcript",
            "transcript",
        ]
    ]


def resolve_transcript(row: pd.Series) -> str:
    """
    metadata 한 행에서 실제 검색에 사용할 텍스트를 우선순위에 따라 고릅니다.

    우선순위
    1. stt_transcript
    2. original_transcript
    3. transcript
    4. processed_txt_path 에 저장된 txt 내용
    """
    for column_name in ["stt_transcript", "original_transcript", "transcript"]:
        value = str(row.get(column_name, "") or "").strip()
        if value:
            return value

    processed_txt_path = str(row.get("processed_txt_path", "") or "").strip()
    if processed_txt_path:
        txt_path = resolve_project_path(processed_txt_path, fallback_dir=PROCESSED_DIR)
        if txt_path is not None and txt_path.exists():
            try:
                return txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                return ""

    return ""


def build_preview(text: str, max_chars: int = 100) -> str:
    """결과 표시에 사용할 짧은 미리보기 문자열을 만듭니다."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def load_metadata_dataframe(metadata_path: str | Path) -> pd.DataFrame:
    """
    metadata CSV를 읽고 검색용 DataFrame으로 정리합니다.

    반환 컬럼에는 최소한 아래가 포함됩니다.
    - file_name
    - file_path
    - processed_txt_path
    - original_transcript
    - stt_transcript
    - search_text
    - preview
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        if metadata_path.name == "audio_metadata.csv" and DEFAULT_DUMMY_TRANSCRIPT_PATH.exists():
            df = build_metadata_from_dummy_transcripts(DEFAULT_DUMMY_TRANSCRIPT_PATH)
        else:
            raise FileNotFoundError(f"metadata CSV를 찾을 수 없습니다: {metadata_path}")
    else:
        df = pd.read_csv(metadata_path, encoding="utf-8-sig")

    if df.empty:
        raise ValueError(f"metadata CSV가 비어 있습니다: {metadata_path}")

    # 예전 metadata 형식과 새 metadata 형식을 모두 흡수합니다.
    for column_name in [
        "file_name",
        "file_path",
        "processed_txt_path",
        "original_transcript",
        "stt_transcript",
        "transcript",
    ]:
        if column_name not in df.columns:
            df[column_name] = ""

    df = normalize_metadata_paths(df.fillna(""))
    df["search_text"] = df.apply(resolve_transcript, axis=1)
    df["preview"] = df["search_text"].apply(build_preview)

    # 검색 불가능한 빈 텍스트는 제거합니다.
    df = df[df["search_text"].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("검색에 사용할 transcript가 없습니다.")

    return df


@dataclass
class KeywordSearchResult:
    rank: int
    row_id: int
    file_name: str
    file_path: str
    score: float
    preview: str
    full_transcript: str


class KeywordSearchEngine:
    """
    TF-IDF 기반 키워드 검색 엔진입니다.

    한국어 형태소 분석기를 따로 쓰지 않고도 비교적 잘 동작하도록
    char_wb n-gram 기반 TF-IDF를 사용합니다.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=False,
            min_df=1,
        )
        self.document_matrix = self.vectorizer.fit_transform(self.df["search_text"])

    @classmethod
    def from_csv(cls, metadata_path: str | Path) -> "KeywordSearchEngine":
        df = load_metadata_dataframe(metadata_path)
        return cls(df)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        query에 대해 상위 top_k 결과를 반환합니다.
        점수는 cosine similarity 기반입니다.
        """
        query = (query or "").strip()
        if not query:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = linear_kernel(query_vector, self.document_matrix).flatten()

        top_k = max(1, min(top_k, len(self.df)))
        ranked_indices = scores.argsort()[::-1][:top_k]

        results: List[Dict] = []
        for rank, row_index in enumerate(ranked_indices, start=1):
            row = self.df.iloc[row_index]
            results.append(
                {
                    "rank": rank,
                    "row_id": int(row_index),
                    "file_name": row["file_name"],
                    "file_path": row["file_path"],
                    "score": float(scores[row_index]),
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
