"""
키워드 검색과 임베딩 검색을 한 화면에서 비교하는 Streamlit 앱입니다.

주요 기능
1. 같은 query에 대해 TF-IDF 키워드 검색과 임베딩 검색 결과를 나란히 비교
2. top_k 조절
3. 개별 결과의 전체 transcript 확인
4. PCA 기반 벡터 공간 시각화
5. 예시 질의셋으로 Top-1 / Top-3 평가

실행 예시
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from src.embedding.vector_index import VectorSearchEngine
from src.search.keyword_search import KeywordSearchEngine, load_metadata_dataframe


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "audio_metadata.csv"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# 더미 데이터 100개를 기준으로 한 예시 평가 질의입니다.
# 실제 metadata에 파일이 없으면 평가에서 자동으로 제외됩니다.
SAMPLE_EVAL_QUERIES: List[Dict] = [
    {"query": "강남권 사무실 이전 후보 건물을 다시 검토한 내용", "expected_file": "audio_001.wav"},
    {"query": "팀 공헌도를 성과 평가에 반영하자는 회의", "expected_file": "audio_022.wav"},
    {"query": "외주 용역 대신 자동화로 비용을 줄이자는 예산 논의", "expected_file": "audio_042.wav"},
    {"query": "유사한 의미 질의도 잘 찾도록 검색 임계값을 조정한 개발 회의", "expected_file": "audio_067.wav"},
    {"query": "검색 광고보다 콘텐츠 광고 전환율이 높았다는 마케팅 리뷰", "expected_file": "audio_081.wav"},
    {"query": "대출 금리를 고려하면 장기 임차가 낫다는 부동산 검토", "expected_file": "audio_018.wav"},
    {"query": "추천 채용 성공률을 높이려면 인터뷰 연결을 빨리 하자는 인사 안건", "expected_file": "audio_037.wav"},
    {"query": "미사용 인스턴스를 정리해서 클라우드 비용을 낮추자는 예산 회의", "expected_file": "audio_053.wav"},
]


def format_result_table(results: List[Dict]) -> pd.DataFrame:
    """검색 결과를 표 형태로 보기 좋게 변환합니다."""
    if not results:
        return pd.DataFrame(columns=["rank", "file_name", "score", "preview", "file_path"])

    df = pd.DataFrame(results)
    df = df[["rank", "file_name", "score", "preview", "file_path"]].copy()
    df["score"] = df["score"].map(lambda x: round(float(x), 4))
    return df


def extract_result_options(results: List[Dict]) -> List[str]:
    """상세 보기 selectbox에 넣을 옵션 문자열을 만듭니다."""
    options = []
    for item in results:
        options.append(
            f"{item['rank']}위 | {item['file_name']} | score={item['score']:.4f}"
        )
    return options


def display_result_detail(title: str, engine_name: str, results: List[Dict], engine) -> None:
    """선택된 검색 결과의 전체 transcript를 보여줍니다."""
    st.markdown(f"#### {title}")
    if not results:
        st.info("표시할 결과가 없습니다.")
        return

    options = extract_result_options(results)
    selected_label = st.selectbox(
        f"{engine_name} 결과 상세 보기",
        options=options,
        key=f"{engine_name}_detail_select",
    )
    selected_index = options.index(selected_label)
    selected_result = results[selected_index]
    document = engine.get_document(selected_result["row_id"])

    st.write(f"파일명: `{document['file_name']}`")
    st.write(f"파일 경로: `{document['file_path']}`")
    if document["processed_txt_path"]:
        st.write(f"TXT 경로: `{document['processed_txt_path']}`")

    with st.expander("전체 transcript 보기", expanded=True):
        st.write(document["search_text"])

    if document["original_transcript"]:
        with st.expander("original_transcript 보기", expanded=False):
            st.write(document["original_transcript"])

    if document["stt_transcript"]:
        with st.expander("stt_transcript 보기", expanded=False):
            st.write(document["stt_transcript"])


def run_simple_evaluation(
    keyword_engine: KeywordSearchEngine,
    vector_engine: VectorSearchEngine,
    available_files: set[str],
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    샘플 질의셋으로 Top-1 / Top-3 정확도를 비교합니다.
    metadata에 실제 존재하는 파일만 평가에 포함합니다.
    """
    evaluation_rows: List[Dict] = []
    filtered_samples = [
        sample for sample in SAMPLE_EVAL_QUERIES if sample["expected_file"] in available_files
    ]

    if not filtered_samples:
        return pd.DataFrame(), {}

    keyword_top1_hits = 0
    keyword_top3_hits = 0
    vector_top1_hits = 0
    vector_top3_hits = 0

    for sample in filtered_samples:
        query = sample["query"]
        expected_file = sample["expected_file"]

        keyword_results = keyword_engine.search(query, top_k=3)
        vector_results = vector_engine.search(query, top_k=3)

        keyword_files = [item["file_name"] for item in keyword_results]
        vector_files = [item["file_name"] for item in vector_results]

        keyword_top1 = len(keyword_files) > 0 and keyword_files[0] == expected_file
        keyword_top3 = expected_file in keyword_files
        vector_top1 = len(vector_files) > 0 and vector_files[0] == expected_file
        vector_top3 = expected_file in vector_files

        keyword_top1_hits += int(keyword_top1)
        keyword_top3_hits += int(keyword_top3)
        vector_top1_hits += int(vector_top1)
        vector_top3_hits += int(vector_top3)

        evaluation_rows.append(
            {
                "query": query,
                "expected_file": expected_file,
                "keyword_top1": keyword_top1,
                "keyword_top3": keyword_top3,
                "embedding_top1": vector_top1,
                "embedding_top3": vector_top3,
                "keyword_top3_predictions": ", ".join(keyword_files),
                "embedding_top3_predictions": ", ".join(vector_files),
            }
        )

    total = len(filtered_samples)
    metrics = {
        "keyword_top1_accuracy": keyword_top1_hits / total,
        "keyword_top3_accuracy": keyword_top3_hits / total,
        "embedding_top1_accuracy": vector_top1_hits / total,
        "embedding_top3_accuracy": vector_top3_hits / total,
        "num_eval_queries": total,
    }

    return pd.DataFrame(evaluation_rows), metrics


@st.cache_data(show_spinner=False)
def load_metadata_cached(metadata_path: str) -> pd.DataFrame:
    """metadata CSV를 캐시해서 불러옵니다."""
    return load_metadata_dataframe(metadata_path)


@st.cache_resource(show_spinner=False)
def build_keyword_engine_cached(metadata_path: str) -> KeywordSearchEngine:
    """키워드 검색 엔진을 캐시합니다."""
    return KeywordSearchEngine.from_csv(metadata_path)


@st.cache_resource(show_spinner=False)
def build_vector_engine_cached(metadata_path: str, model_name: str) -> VectorSearchEngine:
    """임베딩 검색 엔진과 FAISS 인덱스를 캐시합니다."""
    return VectorSearchEngine.from_csv(metadata_path, model_name=model_name)


def build_scatter_figure(projection_df: pd.DataFrame):
    """PCA 결과를 발표용 scatter plot으로 그립니다."""
    color_values = projection_df.apply(
        lambda row: "query"
        if row["label"] == "query"
        else ("top_result" if row["is_top_result"] else "corpus"),
        axis=1,
    )
    projection_df = projection_df.copy()
    projection_df["plot_group"] = color_values

    figure = px.scatter(
        projection_df,
        x="x",
        y="y",
        color="plot_group",
        symbol="plot_group",
        hover_name="file_name",
        hover_data={
            "preview": True,
            "x": False,
            "y": False,
            "plot_group": False,
            "row_id": True,
        },
        color_discrete_map={
            "corpus": "#A3A3A3",
            "top_result": "#2563EB",
            "query": "#DC2626",
        },
        symbol_map={
            "corpus": "circle",
            "top_result": "diamond",
            "query": "x",
        },
        height=620,
    )

    figure.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    figure.update_layout(
        title="PCA 기반 벡터 공간 시각화",
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        legend_title="포인트 유형",
        template="plotly_white",
    )
    return figure


def main() -> None:
    st.set_page_config(
        page_title="음성 검색 비교 실험 앱",
        page_icon="🔎",
        layout="wide",
    )

    st.title("음성 검색 비교 실험 앱")
    st.caption(
        "같은 질의에 대해 키워드 기반 검색과 임베딩 기반 검색을 한 화면에서 비교합니다."
    )

    with st.sidebar:
        st.header("설정")
        metadata_path = st.text_input(
            "Metadata CSV 경로",
            value=str(DEFAULT_METADATA_PATH),
        )
        embedding_model_name = st.text_input(
            "Sentence-Transformers 모델",
            value=DEFAULT_EMBEDDING_MODEL,
        )
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5, step=1)
        st.markdown("---")
        st.markdown("### 예시 질의")
        sample_queries = [
            "",
            "사무실 이전 후보 건물을 다시 검토하자는 내용",
            "성과 평가에 팀 공헌도를 반영하자는 회의",
            "클라우드 비용을 줄이기 위해 미사용 인스턴스를 정리한 논의",
            "검색 기능에서 비슷한 의미 질의도 찾도록 조정한 개발 회의",
            "콘텐츠 광고 전환율이 높았던 마케팅 리뷰",
        ]
        selected_sample_query = st.selectbox(
            "샘플 선택",
            options=sample_queries,
            index=0,
        )

    query = st.text_input(
        "질문을 입력하세요",
        value=selected_sample_query,
        placeholder="예: 사무실 이전 후보 건물을 다시 검토하자는 회의 내용",
    )

    if not metadata_path.strip():
        st.error("metadata CSV 경로를 입력하세요.")
        return

    try:
        with st.spinner("metadata와 검색 인덱스를 로드하는 중입니다..."):
            metadata_df = load_metadata_cached(metadata_path)
            keyword_engine = build_keyword_engine_cached(metadata_path)
            vector_engine = build_vector_engine_cached(metadata_path, embedding_model_name)
    except Exception as exc:
        st.error(f"앱 초기화 실패: {exc}")
        st.stop()

    st.success(f"문서 수: {len(metadata_df)}개")

    tab_compare, tab_visualize, tab_evaluate, tab_dataset = st.tabs(
        ["검색 비교", "벡터 시각화", "정량 평가", "데이터셋 보기"]
    )

    with tab_compare:
        st.subheader("같은 질의에 대한 검색 결과 비교")
        if not query.strip():
            st.info("질의를 입력하면 두 검색 결과를 비교할 수 있습니다.")
        else:
            keyword_results = keyword_engine.search(query, top_k=top_k)
            vector_results = vector_engine.search(query, top_k=top_k)

            overlap = len(
                set(item["file_name"] for item in keyword_results)
                & set(item["file_name"] for item in vector_results)
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("문서 수", len(metadata_df))
            metric_col2.metric("Top-K", top_k)
            metric_col3.metric("상위 결과 겹침 수", overlap)

            left_col, right_col = st.columns(2)

            with left_col:
                st.markdown("### 키워드 기반 검색")
                st.caption("TF-IDF char n-gram 기반")
                st.dataframe(
                    format_result_table(keyword_results),
                    use_container_width=True,
                    hide_index=True,
                )
                display_result_detail("키워드 검색 상세", "keyword", keyword_results, keyword_engine)

            with right_col:
                st.markdown("### 임베딩 기반 검색")
                st.caption(f"Sentence-Transformers + FAISS")
                st.dataframe(
                    format_result_table(vector_results),
                    use_container_width=True,
                    hide_index=True,
                )
                display_result_detail("임베딩 검색 상세", "embedding", vector_results, vector_engine)

    with tab_visualize:
        st.subheader("벡터 공간 비교")
        st.caption("전체 문서를 PCA로 2차원에 투영하고, 질의와 상위 결과를 강조합니다.")

        if not query.strip():
            st.info("질의를 입력하면 query 포인트와 top 결과가 함께 표시됩니다.")
            highlighted_ids: List[int] = []
        else:
            top_vector_results = vector_engine.search(query, top_k=top_k)
            highlighted_ids = [item["row_id"] for item in top_vector_results]

        projection_df = vector_engine.get_projection_dataframe(
            query=query,
            highlighted_row_ids=highlighted_ids,
        )
        figure = build_scatter_figure(projection_df)
        st.plotly_chart(figure, use_container_width=True)

        with st.expander("시각화 해석 팁", expanded=False):
            st.write(
                "회색 점은 전체 문서, 파란 점은 임베딩 검색 상위 결과, 빨간 X는 현재 질의입니다. "
                "질의와 가까운 문서일수록 임베딩 공간에서 의미적으로 더 유사할 가능성이 큽니다."
            )

    with tab_evaluate:
        st.subheader("샘플 질의 기반 간단 평가")
        st.caption("Top-1 Accuracy, Top-3 Accuracy를 비교합니다.")

        available_files = set(metadata_df["file_name"].tolist())
        eval_df, eval_metrics = run_simple_evaluation(
            keyword_engine=keyword_engine,
            vector_engine=vector_engine,
            available_files=available_files,
        )

        if eval_df.empty:
            st.warning(
                "평가용 샘플 파일명이 현재 metadata에 없습니다. "
                "더미 데이터 100개를 생성한 뒤 다시 실행하세요."
            )
            st.code(
                "예시 평가 매핑 구조:\n"
                "SAMPLE_EVAL_QUERIES = [\n"
                "    {'query': '강남권 사무실 이전 후보 건물을 다시 검토한 내용', 'expected_file': 'audio_001.wav'},\n"
                "    {'query': '외주 용역 대신 자동화로 비용을 줄이자는 예산 논의', 'expected_file': 'audio_042.wav'},\n"
                "]"
            )
        else:
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            metric_col1.metric("평가 질의 수", int(eval_metrics["num_eval_queries"]))
            metric_col2.metric("Keyword Top-1", f"{eval_metrics['keyword_top1_accuracy']:.2%}")
            metric_col3.metric("Keyword Top-3", f"{eval_metrics['keyword_top3_accuracy']:.2%}")
            metric_col4.metric("Embedding Top-1", f"{eval_metrics['embedding_top1_accuracy']:.2%}")
            metric_col5.metric("Embedding Top-3", f"{eval_metrics['embedding_top3_accuracy']:.2%}")

            st.dataframe(eval_df, use_container_width=True, hide_index=True)

    with tab_dataset:
        st.subheader("현재 metadata 미리보기")
        preview_df = metadata_df[
            [
                "file_name",
                "file_path",
                "processed_txt_path",
                "original_transcript",
                "stt_transcript",
                "search_text",
            ]
        ].copy()
        st.dataframe(preview_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
