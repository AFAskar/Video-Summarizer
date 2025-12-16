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
    summary = summarizer(sub, model)
    print(summary)


def get_from_cache(
    url: str,
    cache_file="cache.json",
):
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
        return cache.get(url)
    return None


# Downloads video and subtitles , only saves Video to a file
@memory.cache(ignore=["path"])
def downloader(url: str, path: Path, reqlang: str) -> str:
    ydl_opts = {
        "subtitleslangs": [reqlang],
        "writesubtitles": False,
    }

    if path:
        ydl_opts["outtmpl"] = str(path)

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subtitles = info.get("subtitles", {})
        langsub = subtitles.get(reqlang) or info.get("automatic_captions", {}).get(
            reqlang
        )
        if not langsub:
            raise ValueError(f"No subtitles found for language: {reqlang}")
        sub_url = langsub[0]["url"]
        response = requests.get(sub_url)
        sub_content = response.text

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
