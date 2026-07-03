from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
import sqlite3
import sys
from typing import Iterable
from urllib.parse import urljoin
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spotfire_expr_normalizer.normalizer import (  # noqa: E402
    SUPPORTED_SPOTFIRE_FUNCTIONS,
    _duckdb_function_names,
)


DOC_ROOT = "https://docs.tibco.com/pub/sfire-analyst/12.5.0/doc/html/ko-KR/TIB_sfire_client/client/topics/ko-KR/"
FUNCTIONS_URL = urljoin(DOC_ROOT, "functions.html")
DB_PATH = REPO_ROOT / ".docs" / "spotfire_function_catalog.sqlite"
UNSUPPORTED_MD_PATH = REPO_ROOT / ".docs" / "ETL0202_SPOTFIRE_UNSUPPORTED_FUNCTIONS.md"
PACKAGE_DB_PATH = SRC_ROOT / "spotfire_expr_normalizer" / "data" / "spotfire_function_catalog.sqlite"
PACKAGE_UNSUPPORTED_MD_PATH = SRC_ROOT / "spotfire_expr_normalizer" / "data" / "ETL0202_SPOTFIRE_UNSUPPORTED_FUNCTIONS.md"


@dataclass(slots=True)
class FunctionEntry:
    category: str
    function_name: str
    signature: str
    description: str
    examples: str
    source_url: str


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        href = attr_map.get("href")
        if href:
            self.links.append(href)


class FunctionTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_article_title = False
        self._in_row = False
        self._in_cell = False
        self._in_code = False
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._current_code: list[str] = []
        self.rows: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        classes = set(str(attr_map.get("class") or "").split())
        if tag == "h1" and "topictitle1" in classes:
            self._in_article_title = True
        elif tag == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_row and tag == "td":
            self._in_cell = True
            self._current_cell = []
        elif self._in_cell and tag == "code":
            self._in_code = True
            self._current_code = []
        elif self._in_cell and tag in {"p", "br", "div"}:
            self._current_cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1":
            self._in_article_title = False
        elif self._in_cell and tag == "code":
            code = _normalize_space("".join(self._current_code))
            if code:
                self._current_cell.append(code)
            self._in_code = False
            self._current_code = []
        elif self._in_row and tag == "td":
            self._current_row.append(_normalize_space("".join(self._current_cell)))
            self._in_cell = False
            self._current_cell = []
        elif tag == "tr" and self._in_row:
            if len(self._current_row) >= 2 and self._looks_like_function_signature(self._current_row[0]):
                self.rows.append((self._current_row[0], self._current_row[1]))
            self._in_row = False
            self._current_row = []

    def handle_data(self, data: str) -> None:
        if self._in_article_title:
            self.title += data
        if self._in_code:
            self._current_code.append(data)
        elif self._in_cell:
            self._current_cell.append(data)

    @staticmethod
    def _looks_like_function_signature(value: str) -> bool:
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(", value))


def main() -> None:
    category_urls = _discover_category_urls()
    entries: list[FunctionEntry] = []
    for url in category_urls:
        entries.extend(_parse_function_page(url))
    entries = _dedupe_entries(entries)
    _write_catalog(entries)
    _write_unsupported_markdown(entries)
    PACKAGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    PACKAGE_DB_PATH.write_bytes(DB_PATH.read_bytes())
    PACKAGE_UNSUPPORTED_MD_PATH.write_text(UNSUPPORTED_MD_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {DB_PATH} ({len(entries)} functions)")
    print(f"wrote {UNSUPPORTED_MD_PATH}")
    print(f"synced {PACKAGE_DB_PATH}")
    print(f"synced {PACKAGE_UNSUPPORTED_MD_PATH}")


def _discover_category_urls() -> list[str]:
    html = _fetch(FUNCTIONS_URL)
    parser = LinkParser()
    parser.feed(html)
    urls: list[str] = []
    for href in parser.links:
        url = urljoin(FUNCTIONS_URL, href)
        if not url.startswith(DOC_ROOT):
            continue
        if not url.endswith("_functions.html") and not url.endswith("cast_method.html"):
            continue
        if url not in urls:
            urls.append(url)
    return urls


def _parse_function_page(url: str) -> list[FunctionEntry]:
    html = _fetch(url)
    parser = FunctionTableParser()
    parser.feed(html)
    category = _dedupe_repeated_title(_normalize_space(parser.title)) or Path(url).stem
    entries: list[FunctionEntry] = []
    for signature, description in parser.rows:
        name_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", signature)
        if name_match is None:
            continue
        examples = _extract_examples(description)
        entries.append(
            FunctionEntry(
                category=category,
                function_name=name_match.group(1),
                signature=signature,
                description=description,
                examples=examples,
                source_url=url,
            )
        )
    return entries


def _dedupe_entries(entries: Iterable[FunctionEntry]) -> list[FunctionEntry]:
    unique: dict[tuple[str, str, str], FunctionEntry] = {}
    for item in entries:
        key = (item.function_name.lower(), item.signature, item.source_url)
        unique.setdefault(key, item)
    return list(unique.values())


def _write_catalog(entries: list[FunctionEntry]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE spotfire_functions (
                function_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                category TEXT NOT NULL,
                signature TEXT NOT NULL,
                description TEXT NOT NULL,
                examples TEXT NOT NULL,
                source_url TEXT NOT NULL,
                implemented INTEGER NOT NULL,
                compat_status TEXT NOT NULL,
                PRIMARY KEY (normalized_name, signature, source_url)
            )
            """
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO spotfire_functions (
                function_name,
                normalized_name,
                category,
                signature,
                description,
                examples,
                source_url,
                implemented,
                compat_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.function_name,
                    item.function_name.lower(),
                    item.category,
                    item.signature,
                    item.description,
                    item.examples,
                    item.source_url,
                    1 if _compat_status(item.function_name) != "unsupported" else 0,
                    _compat_status(item.function_name),
                )
                for item in entries
            ],
        )
        connection.execute(
            "CREATE INDEX idx_spotfire_functions_implemented ON spotfire_functions (implemented, normalized_name)"
        )


def _write_unsupported_markdown(entries: list[FunctionEntry]) -> None:
    unique: dict[str, FunctionEntry] = {}
    for item in entries:
        normalized = item.function_name.lower()
        if _compat_status(item.function_name) != "unsupported":
            continue
        unique.setdefault(normalized, item)

    documented_names = {item.function_name.lower() for item in entries}
    explicit_names = documented_names & {item.lower() for item in SUPPORTED_SPOTFIRE_FUNCTIONS}
    passthrough_names = {
        item.function_name.lower()
        for item in entries
        if _compat_status(item.function_name) == "duckdb_passthrough"
    }
    lines = [
        "# ETL0202 Spotfire Unsupported Functions",
        "",
        "Generated from the Spotfire Analyst 12.5 Korean function documentation.",
        "",
        f"- Source index: {FUNCTIONS_URL}",
        f"- Catalog DB: `{DB_PATH.relative_to(REPO_ROOT).as_posix()}`",
        f"- Documented functions: {len(documented_names)}",
        f"- Explicitly mapped by ETL0202 normalizer: {len(explicit_names)}",
        f"- Accepted as DuckDB passthrough: {len(passthrough_names)}",
        f"- Not implemented yet: {len(unique)}",
        "",
        "| Function | Category | Signature |",
        "| --- | --- | --- |",
    ]
    for name in sorted(unique):
        item = unique[name]
        lines.append(f"| `{item.function_name}` | {item.category} | `{item.signature}` |")
    UNSUPPORTED_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNSUPPORTED_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compat_status(function_name: str) -> str:
    normalized = function_name.strip().lower()
    if normalized in {item.lower() for item in SUPPORTED_SPOTFIRE_FUNCTIONS}:
        return "explicit_mapper"
    if normalized in _duckdb_function_names():
        return "duckdb_passthrough"
    return "unsupported"


def _fetch(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _dedupe_repeated_title(value: str) -> str:
    half = len(value) // 2
    if half and len(value) % 2 == 0 and value[:half] == value[half:]:
        return value[:half]
    return value


def _extract_examples(description: str) -> str:
    marker = "예:"
    if marker not in description:
        return ""
    return description.split(marker, 1)[1].strip()


if __name__ == "__main__":
    main()
