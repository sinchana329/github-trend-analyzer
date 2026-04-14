"""
Store repository JSON data in MongoDB Atlas.

Features:
- Bulk upsert to avoid duplicates
- Unique index on repo id
- Secondary indexes on language and created_at
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient, UpdateOne

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def load_records(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_collection():
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("Missing MONGODB_URI in environment or .env file.")

    client = MongoClient(mongo_uri)
    db = client["github_trends"]
    collection = db["repos"]

    collection.create_index([("id", ASCENDING)], unique=True, name="uniq_repo_id")
    collection.create_index([("language", ASCENDING)], name="idx_language")
    collection.create_index([("created_at", ASCENDING)], name="idx_created_at")
    return collection


def upsert_records(records: List[dict]) -> None:
    if not records:
        LOGGER.warning("No records found to insert.")
        return

    collection = get_collection()
    ingestion_ts = datetime.now(timezone.utc).isoformat()
    operations = []
    for doc in records:
        if doc.get("id") is None:
            continue
        enriched = dict(doc)
        enriched["ingested_at"] = ingestion_ts
        operations.append(UpdateOne({"id": doc["id"]}, {"$set": enriched}, upsert=True))
    if not operations:
        LOGGER.warning("No valid operations generated.")
        return

    result = collection.bulk_write(operations, ordered=False)
    LOGGER.info(
        "MongoDB upsert complete | inserted=%s modified=%s matched=%s upserts=%s",
        result.inserted_count,
        result.modified_count,
        result.matched_count,
        len(result.upserted_ids) if result.upserted_ids else 0,
    )


def main() -> None:
    data_path = Path("data/raw_repos.json")
    records = load_records(data_path)
    upsert_records(records)


if __name__ == "__main__":
    main()
