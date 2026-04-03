import whisper
from pathlib import Path

AUDIO_PATH = Path("data/raw/test.wav")
OUTPUT_PATH = Path("data/processed/test.txt")

def main():
    if not AUDIO_PATH.exists():
        print(f"파일이 없음: {AUDIO_PATH}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Whisper 모델 로드 중...")
    model = whisper.load_model("base")

    print("음성 변환 중...")
    result = model.transcribe(str(AUDIO_PATH), language="ko")

    text = result["text"].strip()

    print("\n=== 변환 결과 ===")
    print(text)

    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"\n저장 완료: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
