"""
Whisper를 사용해 data/raw 폴더의 WAV 파일을 일괄 전사하는 스크립트입니다.

주요 기능
1. data/raw 폴더의 audio_001.wav 같은 파일을 순회합니다.
2. 각 WAV 파일을 Whisper로 한국어 STT 합니다.
3. STT 결과를 data/processed 폴더에 .txt 파일로 저장합니다.
4. data/metadata/audio_metadata.csv 를 갱신하여
   - file_name
   - file_path
   - processed_txt_path
   - original_transcript
   - stt_transcript
   정보를 유지합니다.

초보자도 바로 실행할 수 있도록 argparse, 로그, 예외 처리, 폴더 자동 생성이 포함되어 있습니다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
METADATA_FILE = METADATA_DIR / "audio_metadata.csv"

METADATA_COLUMNS = [
    "file_name",
    "file_path",
    "processed_txt_path",
    "original_transcript",
    "stt_transcript",
]


def to_portable_path(path: Path) -> str:
    """프로젝트 내부 파일은 상대경로로 저장해 배포 환경에서도 그대로 씁니다."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def ensure_directories() -> None:
    """필요한 폴더가 없으면 자동으로 생성합니다."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def read_existing_metadata(metadata_path: Path) -> Dict[str, Dict[str, str]]:
    """
    기존 metadata CSV를 읽어서 file_name 기준 딕셔너리로 반환합니다.

    기존 CSV가 없더라도 오류를 내지 않고 빈 딕셔너리를 반환합니다.
    """
    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
    except Exception as exc:
        print(f"[경고] metadata 파일을 읽는 중 오류가 발생했습니다: {exc}")
        return {}

    metadata_map: Dict[str, Dict[str, str]] = {}
    for row in rows:
        file_name = (row.get("file_name") or "").strip()
        if not file_name:
            continue

        # 필수 컬럼이 없더라도 안전하게 기본값을 채웁니다.
        metadata_map[file_name] = {
            "file_name": file_name,
            "file_path": row.get("file_path", ""),
            "processed_txt_path": row.get("processed_txt_path", ""),
            "original_transcript": row.get("original_transcript", ""),
            "stt_transcript": row.get("stt_transcript", ""),
        }

    return metadata_map


def save_metadata(metadata_path: Path, rows: List[Dict[str, str]]) -> None:
    """최종 메타데이터를 UTF-8 BOM 형식으로 저장합니다."""
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def transcribe_audio_files(
    whisper_model_name: str,
    language: str,
    raw_dir: Path,
    processed_dir: Path,
    metadata_path: Path,
) -> None:
    """
    WAV 파일들을 Whisper로 일괄 전사하고 메타데이터를 갱신합니다.
    """
    ensure_directories()

    audio_files = sorted(raw_dir.glob("audio_*.wav"))
    if not audio_files:
        print(f"[오류] 전사할 WAV 파일이 없습니다: {raw_dir}")
        print("[안내] 먼저 src/utils/generate_tts_wav.py 를 실행해 WAV 파일을 생성하세요.")
        return

    try:
        import whisper
    except ImportError as exc:
        raise ImportError(
            "whisper 패키지를 찾을 수 없습니다. "
            "`pip install openai-whisper` 후 다시 실행하세요."
        ) from exc

    print(f"[정보] Whisper 모델 로드 중: {whisper_model_name}")
    model = whisper.load_model(whisper_model_name)

    metadata_map = read_existing_metadata(metadata_path)
    updated_rows: List[Dict[str, str]] = []

    total_count = len(audio_files)
    success_count = 0
    failed_count = 0

    for index, audio_path in enumerate(audio_files, start=1):
        print(f"\n[{index:03d}/{total_count:03d}] 전사 시작: {audio_path.name}")

        processed_txt_path = processed_dir / f"{audio_path.stem}.txt"

        # 기존 metadata가 있으면 original_transcript를 유지하고,
        # 없으면 빈 문자열로 채웁니다.
        existing_row = metadata_map.get(audio_path.name, {})
        original_transcript = existing_row.get("original_transcript", "")

        try:
            result = model.transcribe(str(audio_path), language=language)
            stt_text = result["text"].strip()

            processed_txt_path.write_text(stt_text, encoding="utf-8")

            updated_rows.append(
                {
                    "file_name": audio_path.name,
                    "file_path": to_portable_path(audio_path),
                    "processed_txt_path": to_portable_path(processed_txt_path),
                    "original_transcript": original_transcript,
                    "stt_transcript": stt_text,
                }
            )

            print(f"[완료] TXT 저장: {processed_txt_path.name}")
            success_count += 1

        except Exception as exc:
            failed_count += 1
            print(f"[실패] {audio_path.name} 처리 중 오류 발생: {exc}")

            # 실패한 파일도 메타데이터에 남겨서 추적할 수 있게 합니다.
            updated_rows.append(
                {
                    "file_name": audio_path.name,
                    "file_path": to_portable_path(audio_path),
                    "processed_txt_path": to_portable_path(processed_txt_path),
                    "original_transcript": original_transcript,
                    "stt_transcript": "",
                }
            )

    save_metadata(metadata_path, updated_rows)

    print("\n[요약] Whisper 일괄 전사 완료")
    print(f"[요약] 성공: {success_count}개")
    print(f"[요약] 실패: {failed_count}개")
    print(f"[요약] metadata 저장 위치: {metadata_path}")


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 정의합니다."""
    parser = argparse.ArgumentParser(
        description="data/raw 폴더의 WAV 파일을 Whisper로 일괄 전사합니다."
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper 모델 이름 (예: tiny, base, small, medium)",
    )
    parser.add_argument(
        "--language",
        default="ko",
        help="Whisper 전사 언어 코드 (기본값: ko)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"WAV 폴더 경로 (기본값: {RAW_DIR})",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"TXT 저장 폴더 경로 (기본값: {PROCESSED_DIR})",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=METADATA_FILE,
        help=f"metadata CSV 경로 (기본값: {METADATA_FILE})",
    )
    return parser.parse_args()


def main() -> int:
    """스크립트 진입점입니다."""
    try:
        args = parse_args()
        transcribe_audio_files(
            whisper_model_name=args.model,
            language=args.language,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            metadata_path=args.metadata_file,
        )
        return 0
    except Exception as exc:
        print(f"[치명적 오류] batch_transcribe 실행 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
