# Streamlit Deployment Guide

## 현재 저장소 기준

- Streamlit 엔트리포인트: `app.py`
- 기본 metadata 경로: `data/metadata/audio_metadata.csv`
- fallback metadata 경로: `data/metadata/dummy_search_transcripts.csv`
- 원본 WAV와 `data/processed/` 없이도 검색 가능

앱은 기본적으로 `audio_metadata.csv`를 먼저 읽고, 파일이 없으면 `dummy_search_transcripts.csv`로 자동 전환됩니다.

## 최소 배포 파일

- `app.py`
- `requirements.txt`
- `src/embedding/vector_index.py`
- `src/search/keyword_search.py`
- `data/metadata/dummy_search_transcripts.csv`

## 더 안정적인 배포 파일

- `app.py`
- `requirements.txt`
- `src/`
- `data/metadata/audio_metadata.csv`
- `data/metadata/dummy_search_transcripts.csv`

이 구성이면 기본 경로와 fallback 경로를 둘 다 유지할 수 있습니다.

## Streamlit Community Cloud 설정

1. 저장소: `taeyoon0825/audio_search_project`
2. Branch: `main`
3. Main file path: `app.py`
4. Python dependencies: 루트의 `requirements.txt`

첫 배포에서는 sentence-transformers 모델을 내려받기 때문에 초기 기동이 평소보다 오래 걸릴 수 있습니다.

## 제외해도 되는 항목

- `venv/`
- `data/raw/`
- `data/processed/`

현재 검색 기능은 metadata 안의 transcript 컬럼만으로 동작하므로 위 폴더들은 배포에 필수는 아닙니다.
