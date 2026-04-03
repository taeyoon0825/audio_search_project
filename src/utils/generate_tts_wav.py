"""
metadata CSV의 transcript 컬럼을 읽어 WAV 파일을 생성하는 스크립트입니다.

기본 동작
1. data/metadata/dummy_audio_dataset.csv 를 읽습니다.
2. transcript 컬럼 내용을 TTS로 음성 합성합니다.
3. 결과 WAV 파일을 data/raw/audio_001.wav 형태로 저장합니다.
4. data/metadata/audio_metadata.csv 를 함께 생성합니다.

지원 TTS 엔진
- gtts      : 가장 간단하게 시작하기 좋음. 다만 인터넷 연결이 필요합니다.
- pyttsx3   : 오프라인 가능. 다만 한국어 음성 품질은 Windows 설치 음성에 따라 달라집니다.
- edge-tts  : 한국어 품질이 비교적 자연스러운 편입니다. 인터넷 연결이 필요합니다.

권장
- 기본은 gtts
- 한국어 품질을 더 높이고 싶으면 edge-tts
- 완전 오프라인이 필요하면 pyttsx3 시도
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "dummy_audio_dataset.csv"
FALLBACK_INPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "dummy_search_transcripts.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
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
    """프로젝트 내부 파일은 상대경로로 저장해 다른 환경에서도 그대로 씁니다."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def ensure_directories() -> None:
    """필요한 폴더가 없으면 자동으로 생성합니다."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    입력 CSV를 읽어 리스트로 반환합니다.

    UTF-8 BOM 파일과 일반 UTF-8 파일을 모두 안전하게 처리합니다.
    """
    if not csv_path.exists():
        if csv_path == DEFAULT_INPUT_CSV and FALLBACK_INPUT_CSV.exists():
            print(f"[안내] 기본 입력 CSV가 없어 대체 파일을 사용합니다: {FALLBACK_INPUT_CSV}")
            csv_path = FALLBACK_INPUT_CSV
        else:
            raise FileNotFoundError(f"입력 CSV 파일이 없습니다: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError("입력 CSV에 데이터가 없습니다.")

    required_columns = {"id", "category", "person", "position", "period", "topic", "transcript", "keywords"}
    input_columns = set(rows[0].keys())
    missing_columns = required_columns.difference(input_columns)
    if missing_columns:
        raise ValueError(f"입력 CSV에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    return rows


def run_ffmpeg_convert_to_wav(source_path: Path, target_path: Path, sample_rate: int) -> None:
    """
    ffmpeg를 사용해 음성 파일을 WAV로 변환합니다.

    gTTS, edge-tts는 보통 mp3로 저장되므로 최종 WAV를 만들기 위해 ffmpeg를 사용합니다.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError(
            "ffmpeg를 찾을 수 없습니다. ffmpeg 설치 후 PATH 등록이 필요합니다."
        )

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(target_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg 변환 실패\n"
            f"source={source_path}\n"
            f"target={target_path}\n"
            f"stderr={result.stderr.strip()}"
        )


def generate_with_gtts(text: str, wav_output_path: Path, sample_rate: int) -> None:
    """
    gTTS로 mp3를 만든 뒤 ffmpeg로 wav로 변환합니다.

    장점: 구현이 단순하고 한국어 발음이 무난합니다.
    단점: 인터넷 연결이 필요합니다.
    """
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise ImportError("gTTS가 설치되어 있지 않습니다. `pip install gTTS` 후 다시 실행하세요.") from exc

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_mp3 = Path(temp_dir) / "temp_tts.mp3"
        tts = gTTS(text=text, lang="ko")
        tts.save(str(temp_mp3))
        run_ffmpeg_convert_to_wav(temp_mp3, wav_output_path, sample_rate)


def find_korean_windows_voice(engine: "pyttsx3.Engine") -> Optional[str]:
    """
    pyttsx3에서 한국어에 가까운 Windows 음성을 찾아봅니다.

    Windows 환경에 따라 음성 이름이 다를 수 있어서
    name, id, languages 문자열을 모두 확인합니다.
    """
    try:
        voices = engine.getProperty("voices")
    except Exception:
        return None

    for voice in voices:
        text_blob = " ".join(
            [
                str(getattr(voice, "name", "")),
                str(getattr(voice, "id", "")),
                str(getattr(voice, "languages", "")),
            ]
        ).lower()

        if any(keyword in text_blob for keyword in ["korean", "ko-kr", "heami", "heeami", "한국", "ko_kr"]):
            return str(getattr(voice, "id", ""))

    return None


def generate_with_pyttsx3(text: str, wav_output_path: Path, voice_name: Optional[str]) -> None:
    """
    pyttsx3를 사용해 로컬에서 WAV 파일을 생성합니다.

    장점: 오프라인 가능
    단점: 한국어 음성 품질과 사용 가능 여부가 Windows 음성 팩에 의존합니다.
    """
    try:
        import pyttsx3
    except ImportError as exc:
        raise ImportError(
            "pyttsx3가 설치되어 있지 않습니다. `pip install pyttsx3 pypiwin32` 후 다시 실행하세요."
        ) from exc

    engine = pyttsx3.init()

    selected_voice = voice_name or find_korean_windows_voice(engine)
    if selected_voice:
        try:
            engine.setProperty("voice", selected_voice)
        except Exception:
            print(f"[경고] pyttsx3 음성 선택 실패: {selected_voice}")

    # 말하는 속도를 너무 빠르지 않게 조정합니다.
    try:
        current_rate = engine.getProperty("rate")
        engine.setProperty("rate", max(120, int(current_rate) - 20))
    except Exception:
        pass

    try:
        engine.save_to_file(text, str(wav_output_path))
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass

    if not wav_output_path.exists():
        raise RuntimeError(
            "pyttsx3가 WAV 파일을 생성하지 못했습니다. "
            "Windows 한국어 음성 팩이 없거나 save_to_file 지원이 불안정할 수 있습니다."
        )


async def _edge_tts_save(text: str, mp3_output_path: Path, voice_name: str) -> None:
    """edge-tts 비동기 저장 함수입니다."""
    import edge_tts

    communicator = edge_tts.Communicate(text=text, voice=voice_name)
    await communicator.save(str(mp3_output_path))


def generate_with_edge_tts(text: str, wav_output_path: Path, sample_rate: int, voice_name: Optional[str]) -> None:
    """
    edge-tts로 mp3를 만든 뒤 ffmpeg로 wav로 변환합니다.

    장점: 한국어 품질이 자연스러운 편
    단점: 인터넷 연결이 필요합니다.
    """
    try:
        import edge_tts  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "edge-tts가 설치되어 있지 않습니다. `pip install edge-tts` 후 다시 실행하세요."
        ) from exc

    selected_voice = voice_name or "ko-KR-SunHiNeural"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_mp3 = Path(temp_dir) / "temp_tts.mp3"
        asyncio.run(_edge_tts_save(text, temp_mp3, selected_voice))
        run_ffmpeg_convert_to_wav(temp_mp3, wav_output_path, sample_rate)


def generate_single_wav(
    text: str,
    wav_output_path: Path,
    engine_name: str,
    sample_rate: int,
    voice_name: Optional[str],
) -> None:
    """선택한 엔진에 맞춰 단일 WAV 파일을 생성합니다."""
    if engine_name == "gtts":
        generate_with_gtts(text, wav_output_path, sample_rate)
        return

    if engine_name == "pyttsx3":
        generate_with_pyttsx3(text, wav_output_path, voice_name)
        return

    if engine_name == "edge-tts":
        generate_with_edge_tts(text, wav_output_path, sample_rate, voice_name)
        return

    raise ValueError(f"지원하지 않는 TTS 엔진입니다: {engine_name}")


def save_metadata(metadata_path: Path, rows: List[Dict[str, str]]) -> None:
    """최종 metadata CSV를 저장합니다."""
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def generate_tts_dataset(
    input_csv: Path,
    raw_dir: Path,
    metadata_path: Path,
    engine_name: str,
    sample_rate: int,
    voice_name: Optional[str],
) -> None:
    """
    입력 CSV를 기반으로 WAV 파일들과 초기 metadata CSV를 생성합니다.
    """
    ensure_directories()
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(input_csv)
    metadata_rows: List[Dict[str, str]] = []

    total_count = len(rows)
    success_count = 0
    failed_count = 0

    for index, row in enumerate(rows, start=1):
        transcript = (row.get("transcript") or "").strip()
        if not transcript:
            print(f"[건너뜀] {index}번째 행의 transcript가 비어 있습니다.")
            failed_count += 1
            continue

        file_name = f"audio_{index:03d}.wav"
        output_path = raw_dir / file_name

        print(f"\n[{index:03d}/{total_count:03d}] TTS 생성 시작: {file_name}")

        try:
            generate_single_wav(
                text=transcript,
                wav_output_path=output_path,
                engine_name=engine_name,
                sample_rate=sample_rate,
                voice_name=voice_name,
            )

            metadata_rows.append(
                {
                    "file_name": file_name,
                    "file_path": to_portable_path(output_path),
                    "processed_txt_path": "",
                    "original_transcript": transcript,
                    "stt_transcript": "",
                }
            )

            print(f"[완료] WAV 저장: {output_path}")
            success_count += 1

        except Exception as exc:
            failed_count += 1
            print(f"[실패] {file_name} 생성 중 오류 발생: {exc}")

    save_metadata(metadata_path, metadata_rows)

    print("\n[요약] TTS WAV 생성 완료")
    print(f"[요약] 성공: {success_count}개")
    print(f"[요약] 실패: {failed_count}개")
    print(f"[요약] WAV 폴더: {raw_dir}")
    print(f"[요약] metadata 저장 위치: {metadata_path}")


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 정의합니다."""
    parser = argparse.ArgumentParser(
        description="metadata CSV의 transcript를 읽어 WAV 파일을 생성합니다."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"입력 CSV 경로 (기본값: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"WAV 저장 폴더 (기본값: {RAW_DIR})",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=METADATA_FILE,
        help=f"생성 metadata CSV 경로 (기본값: {METADATA_FILE})",
    )
    parser.add_argument(
        "--engine",
        choices=["gtts", "pyttsx3", "edge-tts"],
        default="gtts",
        help="사용할 TTS 엔진 (기본값: gtts)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="최종 WAV 샘플레이트 (기본값: 22050)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="pyttsx3 또는 edge-tts에서 사용할 음성 이름",
    )
    return parser.parse_args()


def main() -> int:
    """스크립트 진입점입니다."""
    try:
        args = parse_args()
        generate_tts_dataset(
            input_csv=args.input_csv,
            raw_dir=args.raw_dir,
            metadata_path=args.metadata_file,
            engine_name=args.engine,
            sample_rate=args.sample_rate,
            voice_name=args.voice,
        )
        return 0
    except Exception as exc:
        print(f"[치명적 오류] TTS 생성 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
