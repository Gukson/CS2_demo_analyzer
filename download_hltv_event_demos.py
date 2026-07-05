#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener


HLTV_BASE_URL = "https://www.hltv.org"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)


@dataclass(frozen=True)
class MatchPage:
    url: str
    name: str


@dataclass(frozen=True)
class DemoLink:
    url: str
    match_name: str
    demo_id: str


class HltvClient:
    def __init__(
        self,
        timeout: int,
        retries: int,
        retry_delay: float,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self._curl_session = self._build_curl_session()
        self.opener = None if self._curl_session is not None else build_opener(HTTPCookieProcessor())

    @staticmethod
    def _build_curl_session() -> object | None:
        try:
            from curl_cffi import requests as curl_requests
        except ImportError:
            return None
        return curl_requests.Session(impersonate="chrome")

    def read_text(self, url: str, referer: str | None = None) -> str:
        for attempt in range(self.retries + 1):
            try:
                return self._read_text_once(url, referer=referer)
            except Exception as exc:
                self._sleep_before_retry(exc, attempt)
        raise RuntimeError("Nieoczekiwany błąd retry podczas czytania strony.")

    def _read_text_once(self, url: str, referer: str | None = None) -> str:
        headers = dict(self.headers)
        if referer:
            headers["Referer"] = referer
        if self._curl_session is not None:
            response = self._curl_session.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        request = Request(url, headers=headers)
        assert self.opener is not None
        with self.opener.open(request, timeout=self.timeout) as response:
            body = response.read()
        return body.decode("utf-8", errors="replace")

    def download(self, url: str, target: Path, referer: str | None = None) -> None:
        for attempt in range(self.retries + 1):
            try:
                self._download_once(url, target, referer=referer)
                return
            except Exception as exc:
                self._sleep_before_retry(exc, attempt)
        raise RuntimeError("Nieoczekiwany błąd retry podczas pobierania pliku.")

    def _download_once(self, url: str, target: Path, referer: str | None = None) -> None:
        headers = dict(self.headers)
        headers["Accept"] = "application/octet-stream,*/*;q=0.8"
        if referer:
            headers["Referer"] = referer
        tmp_target = target.with_suffix(target.suffix + ".part")
        if self._curl_session is not None:
            with self._curl_session.stream("GET", url, headers=headers, timeout=self.timeout) as response:
                response.raise_for_status()
                with tmp_target.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            tmp_target.replace(target)
            return
        request = Request(url, headers=headers)
        assert self.opener is not None
        with self.opener.open(request, timeout=self.timeout) as response:
            with tmp_target.open("wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
        tmp_target.replace(target)

    def _sleep_before_retry(self, exc: Exception, attempt: int) -> None:
        if not is_retryable_http_error(exc) or attempt >= self.retries:
            raise exc
        wait_seconds = self.retry_delay * (attempt + 1)
        print(
            f"  Chwilowy błąd HTTP ({exc}); ponawiam za {wait_seconds:.1f}s...",
            file=sys.stderr,
        )
        time.sleep(wait_seconds)


def is_retryable_http_error(exc: Exception) -> bool:
    status_code = getattr(exc, "code", None)
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", status_code)
    return status_code in {408, 429, 500, 502, 503, 504}


def parse_event_id(event_url: str) -> str:
    parsed = urlparse(event_url)
    event_from_query = parse_qs(parsed.query).get("event")
    if event_from_query:
        return event_from_query[0]
    match = re.search(r"/events/(\d+)(?:/|$)", parsed.path)
    if not match:
        raise ValueError("Nie umiem znaleźć ID eventu w podanym linku HLTV.")
    return match.group(1)


def event_slug(event_url: str, event_id: str) -> str:
    parsed = urlparse(event_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 3 and parts[0] == "events" and parts[1] == event_id:
        return safe_filename(unquote(parts[2]))
    return f"hltv-event-{event_id}"


def safe_filename(value: str, fallback: str = "hltv-demo") -> str:
    value = unescape(value).strip().lower()
    value = re.sub(r"[^a-z0-9._ -]+", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    value = re.sub(r"-+", "-", value).strip(".-")
    return value or fallback


def unique_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def extract_match_pages(html: str) -> list[MatchPage]:
    matches: list[MatchPage] = []
    pattern = re.compile(
        r'href="(?P<href>/matches/(?P<id>\d+)/(?P<slug>[^"#?]+)(?:[^"]*)?)"',
        re.IGNORECASE,
    )
    for match in pattern.finditer(html):
        href = unescape(match.group("href"))
        slug = unquote(match.group("slug"))
        url = urljoin(HLTV_BASE_URL, href)
        matches.append(MatchPage(url=url, name=safe_filename(slug, f"match-{match.group('id')}")))

    by_url: dict[str, MatchPage] = {}
    for match in matches:
        by_url.setdefault(match.url, match)
    return list(by_url.values())


def extract_demo_links(html: str, match_page: MatchPage) -> list[DemoLink]:
    demo_urls = unique_preserving_order(
        urljoin(HLTV_BASE_URL, unescape(match.group(1)))
        for match in re.finditer(r'href="([^"]*/download/demo/[^"#?]+(?:[^"]*)?)"', html, re.IGNORECASE)
    )
    links: list[DemoLink] = []
    for demo_url in demo_urls:
        demo_id_match = re.search(r"/download/demo/([^/?#]+)", demo_url)
        demo_id = safe_filename(unquote(demo_id_match.group(1))) if demo_id_match else "demo"
        links.append(DemoLink(url=demo_url, match_name=match_page.name, demo_id=demo_id))
    return links


def collect_match_pages(
    client: HltvClient,
    event_id: str,
    expected_event_slug: str,
    delay_seconds: float,
    max_pages: int,
) -> list[MatchPage]:
    all_matches: dict[str, MatchPage] = {}
    for page_index in range(max_pages):
        offset = page_index * 100
        url = f"{HLTV_BASE_URL}/results?event={quote(event_id)}&offset={offset}"
        print(f"Sprawdzam wyniki: {url}")
        html = client.read_text(url)
        page_matches = [
            match for match in extract_match_pages(html) if expected_event_slug in match.name
        ]
        new_count = 0
        for match in page_matches:
            if match.url not in all_matches:
                all_matches[match.url] = match
                new_count += 1
        if not page_matches or new_count == 0:
            break
        time.sleep(delay_seconds)
    return list(all_matches.values())


def collect_demo_links(
    client: HltvClient,
    matches: list[MatchPage],
    delay_seconds: float,
) -> list[DemoLink]:
    demos: list[DemoLink] = []
    for index, match_page in enumerate(matches, start=1):
        print(f"[{index}/{len(matches)}] Szukam demek: {match_page.name}")
        try:
            html = client.read_text(match_page.url, referer=f"{HLTV_BASE_URL}/results")
        except Exception as exc:
            print(f"  Pomijam mecz, błąd strony: {exc}", file=sys.stderr)
            continue
        links = extract_demo_links(html, match_page)
        if links:
            print(f"  Znaleziono demek: {len(links)}")
            demos.extend(links)
        else:
            print("  Brak linku do demka.")
        time.sleep(delay_seconds)
    return demos


def target_path_for_demo(output_dir: Path, demo: DemoLink, ordinal: int) -> Path:
    name = safe_filename(f"{ordinal:03d}-{demo.match_name}-{demo.demo_id}")
    return output_dir / f"{name}.rar"


def download_demos(
    client: HltvClient,
    demos: list[DemoLink],
    output_dir: Path,
    delay_seconds: float,
    dry_run: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, demo in enumerate(demos, start=1):
        target = target_path_for_demo(output_dir, demo, index)
        print(f"[{index}/{len(demos)}] {target.name}")
        print(f"  {demo.url}")
        if dry_run:
            continue
        if target.exists() and target.stat().st_size > 0:
            print("  Już istnieje, pomijam.")
            continue
        try:
            client.download(demo.url, target, referer=HLTV_BASE_URL)
        except Exception as exc:
            print(f"  Błąd pobierania: {exc}", file=sys.stderr)
            continue
        print(f"  Zapisano: {target}")
        time.sleep(delay_seconds)


def default_downloads_dir() -> Path:
    xdg_download = os.environ.get("XDG_DOWNLOAD_DIR")
    if xdg_download:
        return Path(os.path.expandvars(os.path.expanduser(xdg_download)))
    return Path.home() / "Downloads"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pobiera wszystkie dostępne dema HLTV z podanych eventów, po kolei.",
    )
    parser.add_argument(
        "event_urls",
        metavar="event_url",
        nargs="*",
        help="Linki do eventów HLTV, np. https://www.hltv.org/events/9168/...",
    )
    parser.add_argument(
        "--event-list",
        type=Path,
        default=None,
        help="Plik tekstowy z linkami do eventów, po jednym w linii. Linie puste i zaczynające się od # są pomijane.",
    )
    parser.add_argument(
        "--event-delay",
        type=float,
        default=10.0,
        help="Pauza między kolejnymi eventami w sekundach.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Liczba ponowień przy chwilowych błędach HTTP, np. 429.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=30.0,
        help="Bazowa pauza przed ponowieniem requestu w sekundach.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Katalog zapisu. Przy jednym evencie domyślnie: ~/Downloads/<slug-eventu>/. "
            "Przy wielu eventach: podkatalog <slug-eventu> w tym katalogu."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Tylko wypisz znalezione demka, bez pobierania.")
    parser.add_argument("--delay", type=float, default=2.0, help="Pauza między requestami w sekundach.")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout pojedynczego requestu w sekundach.")
    parser.add_argument("--max-result-pages", type=int, default=10, help="Limit stron wyników HLTV po 100 meczów.")
    return parser


def read_event_list(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def collect_event_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.event_urls)
    if args.event_list is not None:
        urls.extend(read_event_list(args.event_list))
    return unique_preserving_order(urls)


def resolve_output_dir(base_output_dir: Path | None, slug: str, event_count: int) -> Path:
    if base_output_dir is None:
        return default_downloads_dir() / slug
    if event_count == 1:
        return base_output_dir
    return base_output_dir / slug


def process_event(
    client: HltvClient,
    event_url: str,
    base_output_dir: Path | None,
    event_count: int,
    delay_seconds: float,
    max_result_pages: int,
    dry_run: bool,
) -> bool:
    try:
        event_id = parse_event_id(event_url)
    except ValueError as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return False

    slug = event_slug(event_url, event_id)
    output_dir = resolve_output_dir(base_output_dir, slug, event_count)
    print(f"\n=== Event: {slug} ({event_id}) ===")

    matches = collect_match_pages(client, event_id, slug, delay_seconds, max_result_pages)
    print(f"Znaleziono meczów: {len(matches)}")
    demos = collect_demo_links(client, matches, delay_seconds)
    print(f"Znaleziono demek: {len(demos)}")
    download_demos(client, demos, output_dir, delay_seconds, dry_run)
    print(f"Gotowe. Katalog: {output_dir}")
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    event_urls = collect_event_urls(args)
    if not event_urls:
        print("Błąd: podaj co najmniej jeden link eventu albo --event-list.", file=sys.stderr)
        return 2

    client = HltvClient(timeout=args.timeout, retries=args.retries, retry_delay=args.retry_delay)
    event_count = len(event_urls)
    success_count = 0

    try:
        for index, event_url in enumerate(event_urls, start=1):
            print(f"\nKolejka eventów: {index}/{event_count}")
            if process_event(
                client=client,
                event_url=event_url,
                base_output_dir=args.output_dir,
                event_count=event_count,
                delay_seconds=args.delay,
                max_result_pages=args.max_result_pages,
                dry_run=args.dry_run,
            ):
                success_count += 1
            if index < event_count:
                time.sleep(args.event_delay)
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"Błąd sieci: {exc}", file=sys.stderr)
        if isinstance(exc, HTTPError) and exc.code == 403:
            print(
                "HLTV odrzuciło prosty request. Spróbuj: python3 -m pip install -e '.[hltv]'",
                file=sys.stderr,
            )
        return 1
    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        if "403" in str(exc) or "Forbidden" in str(exc):
            print(
                "HLTV odrzuciło prosty request. Spróbuj: python3 -m pip install -e '.[hltv]'",
                file=sys.stderr,
            )
        return 1

    print(f"\nZakończono kolejkę: {success_count}/{event_count} eventów.")
    return 0 if success_count == event_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
