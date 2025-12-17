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
from joblib import Memory
import srt
import tiktoken
from functools import partial
from faster_whisper import WhisperModel
from multiprocessing.pool import ThreadPool
from datetime import timedelta
import validators
from ythelper import download_audio, download_multi_subs, download_subtitle, ytsearch

# TODO: add concurrensy for downloading audio in the background while downloading subtitles
# TODO: Implement Fallback to Whisper if no subtitles found (blocker above)
# TODO: add option to send a search query instead of URL (function done needs to be integrated)
_tokenizer = tiktoken.get_encoding("cl100k_base")
console = Console()
app = typer.Typer()

model_size = "small.en"

# sst = WhisperModel(model_size, device="cpu", compute_type="int8")

CACHE_DIR = "./cache/"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["openkey"],
)
memory = Memory(CACHE_DIR, verbose=0)

SYSTEMPROMPT = "You are a helpful assistant that answers questions based on the information provided to you."

SUMMARYPROMPT = """You are a helpful assistant that summarizes video subtitles into concise summaries. output the summary in markdown format. use headings and bullet points where appropriate."""

SUMMARYUSERPROMPT = (
    "Provide a concise summary in {language} of the following subtitles from a video"
)
USERPROMPT = "Provide an Answer to the Users query Provided Here"


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
    query: str = typer.Option(None, "-q", "--query", help="What do you want to ask"),
    keepfiles: str = typer.Option(
        "",
        "-k",
        "--keepfiles",
        help="Keep downloaded subtitle and audio files. Options: 'a' for audio, 's' for subtitles (comma-separated)",
    ),
):
    isurl = validators.url(url)
    parrec = partial(
        chunk_summarize_recursive,
        model=model,
        userprompt=(
            SUMMARYUSERPROMPT.format(language=language) if isurl else USERPROMPT
        ),
        systemprompt=(SUMMARYPROMPT if isurl else SYSTEMPROMPT),
        switch=True,
    )
    systemprompt = SUMMARYPROMPT if isurl else SYSTEMPROMPT
    userprompt = (
        (SUMMARYUSERPROMPT.format(language=language) if isurl else USERPROMPT),
    )
    daudio = "a" in keepfiles
    dsub = "s" in keepfiles
    # Download subtitles spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        download_task = progress.add_task("Downloading subtitles...", total=None)
        if isurl:
            sub = [].append(
                downloader(url=url, reqlang=language, audio=daudio, subtitle=dsub)
            )
        else:
            term = ytsearch(query if query else url)
            ls = download_multi_subs(term)
            sub = "\n\n\n---\n\n\n".join(list(map(parrec, ls)))

        progress.remove_task(download_task)
        console.print("✓ Subtitles downloaded")

        summarize_task = progress.add_task("Summarizing subtitles...", total=None)
        # run this once per video

        summary = chunk_summarize_recursive(sub, model, userprompt, systemprompt)
        # results in summaries list, the total length of all the summaries should fit in the input context size, this is something to consider when calculating the budget (it should be divided by the number of videos)
        # and all of those should be joined and then put into one LLM call which answers the question (or produces the final summary)
        progress.remove_task(summarize_task)
        console.print("✓ Summary complete")

        if output:
            saving_task = progress.add_task("Saving summary to file...", total=None)
            output.parent.mkdir(parents=True, exist_ok=True)
            summary_file = output
            summary_file.write_text("# Summary\n\n" + summary, encoding="utf-8")
            progress.remove_task(saving_task)
            console.print(f"✓ Summary saved to {summary_file}")

    console.print(Markdown("\n# Summary\n" + summary))


def issrt(text: list[str]) -> bool:
    try:
        list(srt.parse(text[0]))
        return True
    except Exception:
        return False


@memory.cache()
def generate_transcript_using_whisper(audio_file: Path, language: str = "en") -> str:
    segments, info = sst.transcribe(
        str(audio_file),
        beam_size=5,
        word_timestamps=True,
        language=language,
    )

    # Convert segments to SRT format
    srt_entries = []
    for i, segment in enumerate(segments, start=1):
        sub = srt.Subtitle(
            index=i,
            start=timedelta(seconds=segment.start),
            end=timedelta(seconds=segment.end),
            content=segment.text.strip(),
        )
        srt_entries.append(sub)

    # Compose into SRT format string
    transcript = srt.compose(srt_entries)
    return transcript


@memory.cache()
def downloader(
    url: str, reqlang: str, audio: bool = False, subtitle: bool = False
) -> str:
    if audio:
        file = download_audio(url, audio_only=True)

    try:
        sub_content = download_subtitle(url, reqlang, subtitle=subtitle)
    except ValueError as e:
        # use whisper to generate subtitles if not found
        console.print("No subtitles found, generating using Whisper...")

        if audio:
            # use directly
            sub_content = generate_transcript_using_whisper(file, language=reqlang)
        else:
            file = download_audio(url, audio_only=True)
            sub_content = generate_transcript_using_whisper(file, language=reqlang)

    return sub_content


def semantic_chunker(subtitle: list[str]) -> list[str]:
    # split subtitle into chunks based on silences(long time gaps between timestamps)
    subtitles = list(srt.parse(subtitle[0]))
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

    # Clean each chunk after composing
    return [clean_subtitle(chunk) for chunk in chunks]


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


def get_token_count(text: str) -> int:
    return len(_tokenizer.encode(text))


def count_chat_tokens(
    text: str,
    systemprompt: str = SYSTEMPROMPT,
    userprompt: str = USERPROMPT,
) -> int:
    combined_text = f"{systemprompt}\n{userprompt}\n{text}"
    return get_token_count(combined_text)


def get_safe_context_length(
    model: str = "nvidia/nemotron-3-nano-30b-a3b:free",
    margin: float = 0.85,
    response_buffer: int = 500,
) -> int:
    max_tokens = get_model_context_length(model)
    if not max_tokens:
        raise ValueError(f"Could not retrieve model info for model: {model}")
    return int(max_tokens * margin) - response_buffer


def non_semantic_chunker(chunks: list[str]) -> list[str]:
    max_chunk_size = get_safe_context_length()
    current_chunk = ""
    chunked_list = []
    for chunk in chunks:
        token_count = count_chat_tokens(chunk)
        if token_count + count_chat_tokens(current_chunk) <= max_chunk_size:
            current_chunk += "\n\n" + chunk
        else:
            if current_chunk:
                chunked_list.append(current_chunk.strip())
            current_chunk = chunk
    if current_chunk:
        chunked_list.append(current_chunk.strip())
    return chunked_list


def chunk_summarize_recursive(
    chunks: list[str],
    model: str,
    userprompt: str = USERPROMPT,
    systemprompt: str = SYSTEMPROMPT,
    switch: bool = False,
) -> str:
    input_budget = get_safe_context_length(model)
    if switch:
        input_budget = input_budget // len(chunks)
    token_count = count_chat_tokens(text)
    if token_count > input_budget:
        if issrt(chunks):
            chunks = semantic_chunker(chunks)
        else:
            chunks = non_semantic_chunker(chunks, max_chunk_size=2000)
        summarize_chunk = partial(
            chat_completion,
            model=model,
            userprompt=userprompt,
            systemprompt=systemprompt,
        )
        summaries = list(ThreadPool(4).imap(summarize_chunk, chunks))

        combined_summary = "\n\n".join(summaries)

        combined_token_count = count_chat_tokens(combined_summary)

        if combined_token_count > input_budget:
            meta_prompt = (
                "Combine and summarize these summaries into a single coherent summary"
            )

            return chat_completion(combined_summary, model, userprompt=meta_prompt)
        else:
            return combined_summary
    return chat_completion(text, model)


@memory.cache()
def chat_completion(
    text: str,
    model: str = "nvidia/nemotron-3-nano-30b-a3b:free",
    systemprompt: str = SYSTEMPROMPT.format(language="en"),
    userprompt: str = USERPROMPT,
) -> str:

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{systemprompt}",
            },
            {
                "role": "user",
                "content": f"{userprompt}:\n{text}",
            },
        ],
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    app()
