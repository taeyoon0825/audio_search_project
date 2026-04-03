import argparse
import base64
import csv
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = Path("data/metadata/dummy_search_transcripts.csv")
DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_METADATA_CSV = Path("data/metadata/generated_audio_metadata.csv")


def to_portable_path(path: Path) -> str:
    """프로젝트 내부 파일은 상대경로로 기록합니다."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def build_powershell_script(text: str, output_path: Path, voice_name: str | None) -> str:
    escaped_text = text.replace("'", "''")
    escaped_output = str(output_path.resolve()).replace("'", "''")

    lines = [
        "Add-Type -AssemblyName System.Speech",
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
    ]

    if voice_name:
        escaped_voice = voice_name.replace("'", "''")
        lines.append(f"$synth.SelectVoice('{escaped_voice}')")

    lines.extend(
        [
            f"$synth.SetOutputToWaveFile('{escaped_output}')",
            f"$synth.Speak('{escaped_text}')",
            "$synth.Dispose()",
        ]
    )

    return "\n".join(lines)


def run_powershell_tts(text: str, output_path: Path, voice_name: str | None) -> None:
    script = build_powershell_script(text=text, output_path=output_path, voice_name=voice_name)
    encoded_script = base64.b64encode(script.encode("utf-16-le")).decode("ascii")

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-EncodedCommand",
            encoded_script,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"TTS generation failed for {output_path.name}: {stderr or 'unknown PowerShell error'}"
        )


def generate_audio_files(
    input_csv: Path,
    output_dir: Path,
    metadata_csv: Path,
    voice_name: str | None,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows_in = list(csv.DictReader(csv_file))

    required_columns = {"id", "category", "person", "position", "period", "topic", "transcript", "keywords"}
    input_columns = set(rows_in[0].keys()) if rows_in else set()
    missing_columns = required_columns.difference(input_columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    rows = []

    for index, row in enumerate(rows_in, start=1):
        transcript = str(row["transcript"]).strip()
        if not transcript:
            continue

        file_name = f"audio_{index:03d}.wav"
        output_path = output_dir / file_name

        print(f"[{index:03d}/{len(rows_in):03d}] {file_name} 생성 중")
        run_powershell_tts(text=transcript, output_path=output_path, voice_name=voice_name)

        rows.append(
            {
                "id": row["id"],
                "category": row["category"],
                "person": row["person"],
                "position": row["position"],
                "period": row["period"],
                "topic": row["topic"],
                "file_name": file_name,
                "file_path": to_portable_path(output_path),
                "transcript": transcript,
                "keywords": row["keywords"],
            }
        )

    output_columns = [
        "id",
        "category",
        "person",
        "position",
        "period",
        "topic",
        "file_name",
        "file_path",
        "transcript",
        "keywords",
    ]
    with metadata_csv.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n생성 완료: {len(rows)}개 음성 파일")
    print(f"출력 폴더: {output_dir.resolve()}")
    print(f"메타데이터: {metadata_csv.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV의 transcript 컬럼을 읽어 WAV 파일로 생성합니다."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"입력 CSV 경로 (기본값: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"WAV 저장 폴더 (기본값: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help=f"생성 결과 메타데이터 CSV 경로 (기본값: {DEFAULT_METADATA_CSV})",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="사용할 Windows 음성 이름. 지정하지 않으면 시스템 기본 음성을 사용합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_audio_files(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        voice_name=args.voice,
    )


if __name__ == "__main__":
    main()
