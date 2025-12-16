from yt_dlp import YoutubeDL
import typer
from pathlib import Path
import requests
from openai import OpenAI
from rich import print
import os
import re
import json
from joblib import Memory

app = typer.Typer()

"""
Recursive Summarization means breaking down a large text into smaller chunks, 
summarizing each chunk individually, and then combining those summaries into a final summary. 
each of those summaries can be further summarized if they are still too long.
"""
# TODO: add persistent cache for the LLM calls and for the yt download, you can use joblib memory or build your own using json, or use redis
# TODO: recursive summarization using chunking
# TODO: bonus: split based on silences using the subtitle timestmps
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
    systemprompt = " You are a helpful assistant that summarizes video subtitles into concise summaries."
    userprompt = f"Provide a concise summary in {language} of the following subtitles from a video"
    summary = summarizer(sub, model, userprompt, systemprompt)

    print(summary)


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
        os.remove(sub_file)
    # clean subtitle content
    # remove timestamps and line numbers
    sub_content = re.sub(
        r"^\d+\n|\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n",
        "",
        sub_content,
        flags=re.MULTILINE,
    )
    # remove html tags from subtitles
    sub_content = re.sub(r"<[^>]+>", "", sub_content)
    # replace multiple newlines with single newline
    sub_content = re.sub(r"\n+", "\n", sub_content)
    # replace multiple spaces with single space
    sub_content = re.sub(r" +", " ", sub_content)
    # trim leading and trailing spaces
    sub_content = sub_content.strip()
    return sub_content


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
def summarizer(text: str, model: str, userprompt: str, systemprompt: str) -> str:
    # get max tokens for the model
    max_tokens = get_model_context_length(model)
    if not max_tokens:
        raise ValueError(f"Could not retrieve model info for model: {model}")
    max_tokens = max_tokens - 500  # leave some buffer for response tokens
    text = text[:2000]  # rough estimate of input tokens
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{systemprompt}",
            },
            {
                "role": "user",
                "content": f"{userprompt}:\n{text.strip()}",
            },
        ],
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    app()
