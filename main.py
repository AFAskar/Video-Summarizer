from yt_dlp import YoutubeDL
import typer
from pathlib import Path
import requests
from openai import OpenAI
from rich import print
import os

app = typer.Typer()

"""
Recursive Summarization means breaking down a large text into smaller chunks, 
summarizing each chunk individually, and then combining those summaries into a final summary. 
each of those summaries can be further summarized if they are still too long.
"""
# TODO: add persistent cache for the LLM calls and for the yt download, you can use joblib memory or build your own using json, or use redis
# TODO: recursive summarization using chunking
# TODO: bonus: split based on silences using the subtitle timestmps


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
    # print(summarizer(sub, model))
    print(sub)


# Downloads video and subtitles , only saves Video to a file
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
    return sub_content


def summarizer(text: str, model: str) -> str:
    max_tokens = 256000
    if len(text) < max_tokens:
        pass

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["openkey"],
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text.",
            },
            {"role": "user", "content": f"Summarize the following text:\n{text}"},
        ],
        max_tokens=max_tokens,
    )

    return completion.choices[0].message["content"]


if __name__ == "__main__":
    app()
