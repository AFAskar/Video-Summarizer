from yt_dlp import YoutubeDL
import typer
from pathlib import Path
import requests
from openai import OpenAI
from rich import print
from rich.console import Console
from rich.markdown import Markdown
import os
import re
import json
from joblib import Memory
import srt
import tiktoken

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
        help="Output Video name.",
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
):

    sub = downloader(url, output, language)
    systemprompt = " You are a helpful assistant that summarizes video subtitles into concise summaries. output the summary in markdown format."
    userprompt = f"Provide a concise summary in {language} of the following subtitles from a video"
    summary = summarizer(sub, model, userprompt, systemprompt)

    # Display summary in console
    console.print(Markdown("# Summary\n" + summary))
    # Save summary to a file
    summary_file = output.with_suffix(".summary.md") if output else Path("summary.md")
    summary_file.write_text(summary, encoding="utf-8")
    print(f"\n[bold blue]Summary saved to {summary_file}[/bold blue]")


# Downloads video and subtitles , only saves Video to a file
@memory.cache(ignore=["path"])
def downloader(url: str, path: Path, reqlang: str, subtitleformat: str = "srt") -> str:
    ydl_opts = {
        "subtitleslangs": [reqlang],
        "writesubtitles": True,
        "skip_download": not bool(path),
        "subtitlesformat": subtitleformat,
        "writeautomaticsub": True,
        "outtmpl": str(path) if path else "%(title)s.%(ext)s",
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url)
        base = Path(ydl.prepare_filename(info)).with_suffix("")
        sub_file = base.with_suffix(f".{reqlang}.{subtitleformat}")
        if not sub_file.exists():
            # Try automatic captions
            sub_file = base.with_suffix(f".{reqlang}.auto.{subtitleformat}")
            if not sub_file.exists():
                raise ValueError(f"No subtitles found for language: {reqlang}")
        sub_content = sub_file.read_text(encoding="utf-8")
    # clean subtitle content
    # remove timestamps and line numbers

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
def get_model_context_length(model_id):

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
        combined_summary = " ".join(summaries)
        # if combined summary is still too long, summarize again
        if count_chat_tokens(systemprompt, userprompt, combined_summary) > input_budget:
            return summarizer(combined_summary, model, userprompt, systemprompt)
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
