import asyncio
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import Settings

EVALUATION_PATH_RE = re.compile(r"^/evaluations/([A-Za-z0-9_-]+)$")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
SIZE_RE = re.compile(r"([\d,.]+)\s*(KB|MB)", re.IGNORECASE)


@dataclass(slots=True)
class ArchiveEvaluation:
    external_id: str
    title: str
    publication_year: int | None = None
    language: str | None = None
    project_title: str | None = None
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    institutions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    contract_codes: list[str] = field(default_factory=list)
    source_url: str = ""
    pdf_url: str | None = None
    text_url: str | None = None
    file_size_kb: int | None = None
    raw_metadata: dict[str, str] = field(default_factory=dict)


class AidDataArchiveClient:
    """Small, polite adapter around AidData's public USAID evaluation archive."""

    def __init__(self, settings: Settings) -> None:
        self.base_url = str(settings.archive_base_url).rstrip("/") + "/"
        self.delay = settings.archive_request_delay_seconds
        self.client = httpx.AsyncClient(
            timeout=settings.archive_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "AidLens/0.1 (+research evidence indexing)"},
        )

    async def __aenter__(self) -> "AidDataArchiveClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.client.aclose()

    async def _throttle(self) -> None:
        if self.delay > 0:
            await asyncio.sleep(self.delay)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
    async def _get(self, url: str) -> httpx.Response:
        await self._throttle()
        response = await self.client.get(url)
        response.raise_for_status()
        return response

    async def list_evaluation_ids(self, page: int = 1) -> list[str]:
        response = await self._get(urljoin(self.base_url, f"?page={page}"))
        soup = BeautifulSoup(response.text, "html.parser")
        ids: list[str] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            match = EVALUATION_PATH_RE.match(str(anchor["href"]))
            if match and match.group(1) not in seen:
                external_id = match.group(1)
                seen.add(external_id)
                ids.append(external_id)
        return ids

    async def fetch_evaluation(self, external_id: str) -> ArchiveEvaluation:
        source_url = urljoin(self.base_url, f"evaluations/{external_id}")
        response = await self._get(source_url)
        soup = BeautifulSoup(response.text, "html.parser")

        heading = soup.find("h1")
        if not isinstance(heading, Tag):
            raise ValueError(f"Evaluation {external_id} has no title")
        title = heading.get_text(" ", strip=True)

        metadata = self._metadata(soup)
        context = " ".join(str(item) for item in heading.find_all_next(string=True)[:30])
        year_match = YEAR_RE.search(context)

        links = [a for a in soup.find_all("a", href=True) if isinstance(a, Tag)]
        pdf_url = self._asset_url(links, ".pdf")
        text_url = self._asset_url(links, ".txt")

        abstract = None
        abstract_heading = soup.find(
            lambda tag: tag.name in {"h2", "h3"}
            and "abstract" in tag.get_text(" ", strip=True).lower()
        )
        if isinstance(abstract_heading, Tag):
            next_text = abstract_heading.find_next(
                string=lambda value: bool(value and value.strip())
            )
            if next_text:
                candidate = str(next_text).strip()
                if candidate.lower() != "no abstract provided.":
                    abstract = candidate

        return ArchiveEvaluation(
            external_id=external_id,
            title=title,
            publication_year=int(year_match.group(1)) if year_match else None,
            language=self._infer_language(context),
            project_title=self._extract_project_title(context),
            abstract=abstract,
            authors=self._split_pipe(metadata.get("Authors")),
            institutions=self._split_pipe(metadata.get("Institution")),
            keywords=self._split_pipe(metadata.get("Keywords")),
            locations=self._infer_locations(context),
            contract_codes=self._split_pipe(metadata.get("Contract/Code")),
            source_url=source_url,
            pdf_url=pdf_url,
            text_url=text_url,
            file_size_kb=self._parse_size_kb(metadata.get("File size")),
            raw_metadata=metadata,
        )

    async def fetch_text(self, url: str) -> str:
        response = await self._get(url)
        return response.text.replace("\x00", "").strip()

    @staticmethod
    def _metadata(soup: BeautifulSoup) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for dt in soup.find_all("dt"):
            if not isinstance(dt, Tag):
                continue
            dd = dt.find_next_sibling("dd")
            if isinstance(dd, Tag):
                metadata[dt.get_text(" ", strip=True)] = dd.get_text(" ", strip=True)
        if metadata:
            return metadata

        labels = {
            "Authors",
            "Contract/Code",
            "Institution",
            "Keywords",
            "ID",
            "File size",
            "Source",
        }
        for node in soup.find_all(string=True):
            label = str(node).strip()
            if label not in labels:
                continue
            parent = node.parent
            if not isinstance(parent, Tag):
                continue
            sibling = parent.find_next_sibling()
            if isinstance(sibling, Tag):
                metadata[label] = sibling.get_text(" ", strip=True)
        return metadata

    def _asset_url(self, links: list[Tag], suffix: str) -> str | None:
        for anchor in links:
            href = str(anchor.get("href", ""))
            if href.lower().split("?")[0].endswith(suffix):
                return urljoin(self.base_url, href)
        return None

    @staticmethod
    def _split_pipe(value: str | None) -> list[str]:
        if not value:
            return []
        return [part.strip() for part in value.split("|") if part.strip()]

    @staticmethod
    def _parse_size_kb(value: str | None) -> int | None:
        if not value:
            return None
        match = SIZE_RE.search(value)
        if not match:
            return None
        amount = float(match.group(1).replace(",", ""))
        return round(amount * 1024) if match.group(2).upper() == "MB" else round(amount)

    @staticmethod
    def _infer_language(context: str) -> str | None:
        return "English" if re.search(r"\bEnglish\b", context, re.IGNORECASE) else None

    @staticmethod
    def _extract_project_title(context: str) -> str | None:
        match = re.search(r"Project title:\s*(.+?)(?=(?:\b[A-Z][a-z]+\b){1,3}\s|$)", context)
        return match.group(1).strip(" |;") if match else None

    @staticmethod
    def _infer_locations(context: str) -> list[str]:
        # Geography is free text in the archive. Canonical entity resolution is deferred
        # to the ML pipeline rather than pretending a brittle heuristic is ground truth.
        return []
