"""File -> text. Headings are preserved (markdown '#' or numbered lines) so the chunker can be section-aware."""
from pathlib import Path

import httpx

ALLOWED = {".pdf", ".html", ".htm", ".md", ".txt"}


def load_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    if ext in (".html", ".htm"):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        for level in range(1, 5):
            for h in soup.find_all(f"h{level}"):
                h.string = f"{'#' * level} {h.get_text(' ', strip=True)}"
        return soup.get_text("\n")
    return path.read_text(encoding="utf-8", errors="ignore")


async def download(url: str, dest: Path) -> Path:
    """Manual-upload fallback: fetch a guidance PDF/HTML you have chosen (e.g. from gov.uk) into data/raw."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest
