#!/usr/bin/env python
"""
Download an OpenWebText subset from HuggingFace and convert it to JSONL
(the format expected by Megatron-LM's tools/preprocess_data.py).

Usage:
    python download_openwebtext.py --target-docs 700000 --docs-per-file 100000

Output:
    raw/openwebtext_part_000.jsonl    ({"text": "..."} per line)
    ...
"""

import argparse
import json
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="Skylion007/openwebtext",
        help="HuggingFace dataset to download",
    )
    parser.add_argument("--config", default="plain_text")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--target-docs", type=int, default=700_000,
        help="Total number of documents to download (~2GB raw text)",
    )
    parser.add_argument(
        "--docs-per-file", type=int, default=100_000,
        help="Documents per JSONL part file",
    )
    parser.add_argument(
        "--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Streaming {args.dataset}/{args.config} ({args.split}), "
          f"taking up to {args.target_docs} docs...")
    ds = load_dataset(
        args.dataset, args.config, split=args.split, streaming=True
    ).take(args.target_docs)

    file_idx = 0
    doc_count = 0
    total_bytes = 0
    f = open(
        os.path.join(args.out_dir, f"openwebtext_part_{file_idx:03d}.jsonl"),
        "w", encoding="utf-8",
    )

    for row in ds:
        line = json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n"
        f.write(line)
        doc_count += 1
        total_bytes += len(line.encode("utf-8"))

        if doc_count % 50_000 == 0:
            print(f"  downloaded {doc_count:,} docs, "
                  f"{total_bytes / 1024**3:.2f} GB", flush=True)

        if doc_count % args.docs_per_file == 0:
            f.close()
            file_idx += 1
            f = open(
                os.path.join(args.out_dir, f"openwebtext_part_{file_idx:03d}.jsonl"),
                "w", encoding="utf-8",
            )

    f.close()
    print(f"Done: {doc_count:,} docs, {total_bytes / 1024**3:.2f} GB, "
          f"{file_idx + 1} file(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
