from yt_dlp import YoutubeDL
from pathlib import Path
from multiprocessing.pool import ThreadPool
from tqdm import tqdm


def ytsearch(query: str) -> list[str]:
    urls = []
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "default_search": "ytsearch10",
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)
        for entry in info["entries"]:
            urls.append(f"https://www.youtube.com/watch?v={entry['id']}")

    return urls


def download_multi_subs(urls: list[str], reqlang: str = "en") -> list[str]:
    return list(
        tqdm(
            ThreadPool().imap(lambda url: download_subtitle(url, reqlang), urls),
            total=len(urls),
        )
    )


def download_subtitle(url: str, reqlang: str = "en", subtitle: bool = False) -> str:
    ydl_opts = {
        "subtitleslangs": [reqlang],
        "writesubtitles": True,
        "skip_download": True,
        "subtitlesformat": "srt",
        "writeautomaticsub": True,
        "format_sort": ["+size", "+res"],
        "quiet": True,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url)
        base = Path(ydl.prepare_filename(info)).with_suffix("")
        sub_file = base.with_suffix(f".{reqlang}.srt")
        if not sub_file.exists():
            sub_file = base.with_suffix(f".{reqlang}.auto.srt")
            if not sub_file.exists():
                sub_file = base.with_suffix(f".{reqlang}.auto.vtt")
            if not sub_file.exists():
                raise ValueError(f"No subtitles found for language: {reqlang}")
        sub_content = sub_file.read_text(encoding="utf-8")
        if not subtitle:
            sub_file.unlink()

    return sub_content


# Download Audio based on keepfiles return file name
def download_audio(url: str) -> str:
    ydl_opts = {
        "skip_download": False,
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "no_overwrites": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url)
    return ydl.prepare_filename(info)
