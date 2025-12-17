from yt_dlp import YoutubeDL
import typer
from pathlib import Path
import requests
from openai import OpenAI
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
import os
import re
import json
from joblib import Memory
import srt
import tiktoken

# TODO: use concurrensy or threading to speed up return to console (get summary whilst downloading vid or vicea versa)

_tokenizer = tiktoken.get_encoding("cl100k_base")

console = Console()
app = typer.Typer()

CACHE_DIR = "./cache/"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["openkey"],
)
memory = Memory(CACHE_DIR, verbose=0)


@app.command()
def cli(
    url: str,
    output: Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Output Summary file.",
    ),
    language: str = typer.Option(
        "en",
        "-l",
        "--language",
        help="Summary language",
    ),
    model: str = typer.Option(
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "-m",
        "--model",
        help="OpenRouter model to use for summarization",
    ),
    keepfiles: str = typer.Option(
        "",
        "-k",
        "--keepfiles",
        help="Keep downloaded subtitle and video files. Options: 'v' for video, 'a' for audio, 's' for subtitles (comma-separated)",
    ),
):

    keepfiles = [k.strip().lower() for k in keepfiles.split(",") if k.strip()]
    # Download subtitles spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        download_task = progress.add_task(
            "Downloading subtitles...", total=None, start=False
        )
        sub = downloader(url=url, reqlang=language, keepfiles=keepfiles)
        progress.update(download_task, description="✓ Subtitles downloaded")

        systemprompt = "You are a helpful assistant that summarizes video subtitles into concise summaries. output the summary in markdown format. "
        userprompt = f"Provide a concise summary in {language} of the following subtitles from a video"
        summarize_task = progress.add_task(
            "Summarizing subtitles...", total=None, start=False
        )
        summary = summarizer(sub, model, userprompt, systemprompt)
        progress.update(summarize_task, description="✓ Summary complete")
        if output:
            saving_task = progress.add_task(
                "Saving summary to file...", total=None, start=False
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            summary_file = output
            summary_file.write_text("# Summary\n\n" + summary, encoding="utf-8")
            progress.update(saving_task, description="✓ Summary saved to file")

    console.print(Markdown("\n# Summary\n" + summary))


# Downloads video and subtitles , only saves Video to a file
@memory.cache()
def downloader(
    url: str,
    reqlang: str,
    keepfiles: list = [],
) -> str:
    video = True if "v" in keepfiles else False
    audio = True if "a" in keepfiles else False
    subtitle = True if "s" in keepfiles else False
    ydl_opts = {
        "subtitleslangs": [reqlang],
        "writesubtitles": True,
        "skip_download": not (video or audio),
        "subtitlesformat": "srt",
        "writeautomaticsub": True,
        "format_sort": ["+size", "+res"],
        "quiet": True,
        "no_warnings": True,
    }
    if audio and not video:
        ydl_opts["format"] = "bestaudio/best"
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ]

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url)
        base = Path(ydl.prepare_filename(info)).with_suffix("")
        sub_file = base.with_suffix(f".{reqlang}.srt")
        if not sub_file.exists():
            # Try automatic captions
            sub_file = base.with_suffix(f".{reqlang}.auto.srt")
            if not sub_file.exists():
                raise ValueError(f"No subtitles found for language: {reqlang}")
        sub_content = sub_file.read_text(encoding="utf-8")
        if not subtitle:
            try:
                sub_file.unlink()
            except Exception as e:
                print(f"Error deleting files: {e}")

    return sub_content


@memory.cache()
def split_into_chunks(subtitle: str) -> list[str]:
    # split subtitle into chunks based on silences(long time gaps between timestamps)
    subtitles = list(srt.parse(subtitle))
    chunks = []
    current_chunk = []
    last_end = None
    for sub in subtitles:
        if last_end is not None:
            gap = (sub.start - last_end).total_seconds()
            if gap > 10:  # if gap is more than 10 seconds, start a new chunk
                chunks.append(srt.compose(current_chunk))
                current_chunk = []
        current_chunk.append(sub)
        last_end = sub.end
    if current_chunk:
        chunks.append(srt.compose(current_chunk))
    return chunks


@memory.cache()
def clean_subtitle(text: str) -> str:
    text = re.sub(
        r"^\d+\n|\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    # remove html tags from subtitles
    text = re.sub(r"<[^>]+>", "", text)
    # replace multiple newlines with single newline
    text = re.sub(r"\n+", "\n", text)
    # replace multiple spaces with single space
    text = re.sub(r" +", " ", text)
    # trim leading and trailing spaces
    text = text.strip()
    return text


@memory.cache()
def get_model_context_length(model_id: str) -> int | None:

    # OpenRouter's models endpoint returns full metadata including context_length
    response = requests.get("https://openrouter.ai/api/v1/models")
    response.raise_for_status()

    models_data = response.json()["data"]

    # Find your specific model
    for model in models_data:
        if model["id"] == model_id:
            # Returns context_length (e.g., 32768)
            return model.get("context_length")

    return None  # Model not found


@memory.cache()
def get_token_count(text: str) -> int:
    return len(_tokenizer.encode(text))


@memory.cache()
def count_chat_tokens(systemprompt: str, userprompt: str, text: str) -> int:
    combined_text = f"{systemprompt}\n{userprompt}\n{text}"
    return get_token_count(combined_text)


@memory.cache()
def get_safe_context_length(
    model: str, margin: float = 0.85, response_buffer: int = 500
) -> int:
    max_tokens = get_model_context_length(model)
    if not max_tokens:
        raise ValueError(f"Could not retrieve model info for model: {model}")
    return int(max_tokens * margin) - response_buffer


@memory.cache()
def summarizer(text: str, model: str, userprompt: str, systemprompt: str) -> str:

    input_budget = get_safe_context_length(model)
    cleaned_text = clean_subtitle(text)
    token_count = count_chat_tokens(systemprompt, userprompt, cleaned_text)
    # if text is too long, split into chunks and summarize each chunk
    if token_count > input_budget:
        chunks = split_into_chunks(text)
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, model, userprompt, systemprompt)
            summaries.append(summary)
        combined_summary = "\n\n".join(summaries)

        combined_token_count = count_chat_tokens(
            systemprompt, userprompt, combined_summary
        )

        if combined_token_count > input_budget:
            meta_prompt = (
                "Combine and summarize these summaries into a single coherent summary"
            )

            return summarizer(combined_summary, model, meta_prompt, systemprompt)
        else:
            return combined_summary

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{systemprompt}",
            },
            {
                "role": "user",
                "content": f"{userprompt}:\n{cleaned_text}",
            },
        ],
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    app()
