"""
audio_transcriber.py

A script to transcribe audio files into text with timestamps using OpenAI's Whisper model.

Usage:
    python audio_transcriber.py <audio_path> [--output <output_path>] [--metadata_output <metadata_output_path>]
"""
import argparse
import json
import logging
import whisper

def seconds_to_hms(seconds: float) -> str:
    """
    Convert seconds to a HH:MM:SS formatted string.

    Args:
        seconds: The time duration in seconds.

    Returns:
        str: The formatted time string in HH:MM:SS format.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def transcribe_audio_with_timestamps(audio_path: str) -> dict:
    """
    Transcribe an audio file using Whisper and return the transcription result.

    Args:
        audio_path: Path to the audio file to be transcribed.

    Returns:
        dict: The transcription results, including text, segments, and metadata.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, verbose=True)
    logging.info("Transcription generated")
    return result

def process_transcription(result: dict) -> list:
    """
    Extract timestamps and text segments from the transcription result.

    Args:
        result: The transcription result obtained from Whisper.

    Returns:
        list: A list of dictionaries.
    """
    segments = []
    if "segments" in result:
        for segment in result["segments"]:
            segments.append({
                "timestamp": seconds_to_hms(segment["start"]),
                "text": segment["text"].strip()
            })
    else:
        segments.append({
            "timestamp": "00:00:00",
            "text": result.get("text", "")
        })
    return segments

def main():
    """
    Main function to handle argument parsing, transcription, and saving results to JSON files.

    """
    # Set up logging configuration.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Transcribe an MP3 file to text with timestamps using Whisper and output metadata."
    )
    parser.add_argument("audio_path", type=str, help="Path to the MP3 audio file.")
    parser.add_argument(
        "--output", type=str, default="transcription.json",
        help="Output JSON file path for the full transcription (default: transcription.json)."
    )
    parser.add_argument(
        "--metadata_output", type=str, default="transcript_metadata.json",
        help="Output JSON file path for the metadata (default: transcript_metadata.json)."
    )
    args = parser.parse_args()

    # Transcribe the audio file.
    result = transcribe_audio_with_timestamps(args.audio_path)
    
    # Process the transcription into timestamped segments.
    segments = process_transcription(result)
    
    # Prepare the full transcription data.
    output_data = {
        "audio_data": segments
    }
    
    # Save the full transcription to a JSON file.
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    logging.info(f"Transcription saved to {args.output}")
    
    # Save the metadata.
    with open(args.metadata_output, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=4)
    logging.info(f"Transcript metadata saved to {args.metadata_output}")

if __name__ == "__main__":
    main()
