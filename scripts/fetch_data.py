"""Incremental GitHub ingestion with Mongo checkpoint metadata."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
GITHUB_REPO_URL = "https://api.github.com/repos/{full_name}"


@dataclass
class GitHubConfig:
    token: str
    start_year: int
    end_year: int
    min_stars: int
    per_page: int
    max_pages_per_year: int
    sleep_seconds: float
    incremental_mode: bool


def _build_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _request_with_retry(
    session: requests.Session, url: str, headers: Dict[str, str], params: Optional[Dict] = None, retries: int = 3
) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, headers=headers, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            sleep_time = attempt * 4
            LOGGER.warning("Network error on attempt %s/%s: %s. Retrying in %s sec.", attempt, retries, exc, sleep_time)
            time.sleep(sleep_time)
            continue
        if response.status_code == 403:
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset_epoch = response.headers.get("X-RateLimit-Reset")
            if remaining == "0" and reset_epoch:
                wait_seconds = max(int(reset_epoch) - int(time.time()) + 2, 2)
                LOGGER.warning("Rate limit reached. Sleeping %s seconds.", wait_seconds)
                time.sleep(wait_seconds)
                continue
        if response.status_code in (429, 500, 502, 503, 504):
            sleep_time = attempt * 3
            LOGGER.warning("Transient error %s. Retry in %s sec", response.status_code, sleep_time)
            time.sleep(sleep_time)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"GitHub request failed: {url}")


def _fetch_topics(session: requests.Session, headers: Dict[str, str], full_name: str) -> List[str]:
    try:
        response = _request_with_retry(session, GITHUB_REPO_URL.format(full_name=full_name), headers)
        return response.json().get("topics", []) or []
    except Exception as exc:
        LOGGER.warning("Skipping topics for %s due to request failure: %s", full_name, exc)
        return []


def _extract_repo(item: Dict, topics: List[str]) -> Dict:
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "full_name": item.get("full_name"),
        "language": item.get("language"),
        "stars": item.get("stargazers_count", 0),
        "forks": item.get("forks_count", 0),
        "open_issues": item.get("open_issues_count", 0),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "description": item.get("description"),
        "topics": topics,
        "html_url": item.get("html_url"),
    }


def _get_pipeline_meta_collection():
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        return None
    client = MongoClient(mongo_uri)
    return client["github_trends"]["pipeline_meta"]


def _load_last_checkpoint() -> Optional[str]:
    collection = _get_pipeline_meta_collection()
    if collection is None:
        return None
    doc = collection.find_one({"_id": "github_fetch_checkpoint"})
    return doc.get("last_fetched_at") if doc else None


def _save_checkpoint(record_count: int) -> None:
    collection = _get_pipeline_meta_collection()
    if collection is None:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    collection.update_one(
        {"_id": "github_fetch_checkpoint"},
        {"$set": {"last_fetched_at": now_iso, "record_count": record_count}},
        upsert=True,
    )
    LOGGER.info("Checkpoint updated in pipeline_meta.")


def fetch_repositories(config: GitHubConfig) -> List[Dict]:
    session = requests.Session()
    headers = _build_headers(config.token)
    collected: Dict[int, Dict] = {}

    if config.incremental_mode:
        last_fetch = _load_last_checkpoint()
        if last_fetch:
            since_dt = datetime.fromisoformat(last_fetch.replace("Z", "+00:00"))
        else:
            since_dt = datetime.now(timezone.utc) - timedelta(days=1)

        query = f"pushed:>={since_dt.strftime('%Y-%m-%d')} stars:>{config.min_stars}"
        LOGGER.info("Incremental mode enabled. Query window starts from %s", since_dt.strftime("%Y-%m-%d"))
        year_ranges = [("incremental", query)]
    else:
        year_ranges = []
        for year in range(config.start_year, config.end_year + 1):
            year_query = f"created:{year}-01-01..{year}-12-31 stars:>{config.min_stars}"
            year_ranges.append((str(year), year_query))

    for label, query in year_ranges:
        LOGGER.info("Fetching repositories for: %s", label)
        for page in range(1, config.max_pages_per_year + 1):
            params = {"q": query, "sort": "stars", "order": "desc", "per_page": config.per_page, "page": page}
            response = _request_with_retry(session, GITHUB_SEARCH_URL, headers, params=params)
            items = response.json().get("items", [])
            if not items:
                break

            for item in items:
                repo_id = item.get("id")
                if repo_id in collected:
                    continue
                full_name = item.get("full_name")
                topics = _fetch_topics(session, headers, full_name) if full_name else []
                collected[repo_id] = _extract_repo(item, topics)

            LOGGER.info("%s | page %s | cumulative repos: %s", label, page, len(collected))
            time.sleep(config.sleep_seconds)

    _save_checkpoint(len(collected))
    return list(collected.values())


def save_as_json(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s records to %s", len(records), output_path)


def main() -> None:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("Missing GITHUB_TOKEN")

    config = GitHubConfig(
        token=token,
        start_year=int(os.getenv("START_YEAR", "2020")),
        end_year=int(os.getenv("END_YEAR", "2024")),
        min_stars=int(os.getenv("MIN_STARS", "50")),
        per_page=int(os.getenv("PER_PAGE", "100")),
        max_pages_per_year=int(os.getenv("MAX_PAGES_PER_YEAR", "10")),
        sleep_seconds=float(os.getenv("SLEEP_SECONDS", "1.0")),
        incremental_mode=os.getenv("INCREMENTAL_MODE", "false").lower() == "true",
    )
    repos = fetch_repositories(config)
    save_as_json(repos, Path("data/raw_repos.json"))


if __name__ == "__main__":
    main()
