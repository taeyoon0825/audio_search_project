# Audio Search Project

현재 프로젝트를 기준으로 확인한 결과:

- `data/` 약 46.9MB
- `src/` 약 0.08MB
- `venv/` 약 1458MB

즉, Git에 올리기 부담이 큰 것은 더미데이터보다 `venv/`입니다.
현재 더미데이터는 개별 파일이 모두 작아서 일반 GitHub 저장소에 올릴 수 있는 수준입니다.

## 배포 관점 핵심 정리

1. `venv/`는 Git에 올리지 않습니다.
2. Streamlit 배포용 의존성은 `requirements.txt`만 사용합니다.
3. 더미데이터를 그대로 검색 가능하게 하려면 `data/metadata/audio_metadata.csv`는 반드시 포함합니다.
4. 현재 `audio_metadata.csv` 100행 모두가 `original_transcript` 또는 `stt_transcript`를 직접 가지고 있어서, 검색만 필요하면 `data/raw/`와 `data/processed/`는 없어도 됩니다.
5. WAV 재생이나 TXT fallback, 원본 파일 접근까지 필요하면 `data/raw/`, `data/processed/`를 함께 포함합니다.
6. 현재 `.gitignore`는 기본적으로 `data/raw/`, `data/processed/`를 제외하도록 설정되어 있습니다.

## 추천 배포 방식

가장 단순한 방식:

- Git에 `app.py`, `src/`, `data/`, `requirements.txt`, `.gitignore`를 올립니다.
- Streamlit Community Cloud에서 `app.py`를 엔트리포인트로 지정합니다.

더 가볍게 올리고 싶다면:

- `data/metadata/audio_metadata.csv`

현재는 이 한 파일만 올려도 검색 기능은 유지됩니다.
이 경우 `data/raw/`, `data/processed/`를 모두 제외할 수 있어서 저장소가 가장 가벼워집니다.

## 로컬 실행

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit 배포 순서

1. Git 저장소를 새로 만듭니다.
2. `.gitignore` 기준으로 `venv/`를 제외하고 커밋합니다.
3. GitHub에 push 합니다.
4. Streamlit Community Cloud에서 저장소를 연결합니다.
5. Main file path를 `app.py`로 지정합니다.

## 데이터가 더 커질 때

지금처럼 약 46.9MB 수준이면 저장소 포함 방식이 가장 단순합니다.
나중에 데이터가 수백 MB 이상으로 커지면 아래 방식으로 바꾸는 것이 낫습니다.

- Git에는 코드와 작은 metadata만 올리기
- 큰 데이터는 S3, Google Drive, Hugging Face Hub Release 같은 외부 저장소에 두기
- 앱 시작 시 다운로드해서 로컬 캐시에 저장하기

## 데이터 재생성

더미 WAV/STT를 다시 만들고 싶으면 로컬 전용 패키지를 추가로 설치합니다.

```bash
pip install -r requirements-tools.txt
```
