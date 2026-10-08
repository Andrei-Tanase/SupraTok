# Datasets

No corpora are redistributed. Obtain each from its original source under its own licence
and place it where the benchmark expects it:

```
datasets/
├── nemotron/nemotron_scale_1.jsonl   English tokenizer training  (one {"text": …} per line)
├── wikitext103/wiki.valid.txt        English evaluation          (plain text)
├── wikipedia/<lang>.jsonl            multilingual training       (one {"text": …} per line)
└── flores200/devtest/<code>.devtest  multilingual evaluation     (one sentence per line)
```

`benchmarks/figure1_benchmark.py` and `benchmarks/ablation_benchmark.py` need the first two,
`benchmarks/multilingual_benchmark.py` the last two.

## Nemotron-CC — training

Source: <https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html>, distributed
as `.jsonl.zstd` shards. This streams the first 100,000 documents of the first
high-quality shard of real (not synthetic) text; it needs `pip install zstandard`.

```python
import gzip
import io
import json
import os
import urllib.request

import zstandard

os.makedirs("datasets/nemotron", exist_ok=True)
BASE = "https://data.commoncrawl.org/"
with urllib.request.urlopen(BASE + "contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz") as response:
    paths = gzip.decompress(response.read()).decode().split()
shard = next(p for p in paths if "/quality=high/kind=actual/" in p)
with urllib.request.urlopen(BASE + shard) as response, \
        open("datasets/nemotron/nemotron_scale_1.jsonl", "w", encoding="utf-8") as out:
    lines = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(response), encoding="utf-8")
    for i, line in enumerate(lines):
        if i >= 100_000:
            break
        out.write(json.dumps({"text": json.loads(line)["text"]}, ensure_ascii=False) + "\n")
```

## WikiText-103 — evaluation

The paper evaluates English compression on the validation split. Save it as plain text at
`datasets/wikitext103/wiki.valid.txt`; this needs `pip install datasets`.

```python
import os

from datasets import load_dataset

os.makedirs("datasets/wikitext103", exist_ok=True)
split = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
with open("datasets/wikitext103/wiki.valid.txt", "w", encoding="utf-8") as out:
    out.write("".join(split["text"]))
```

## Wikipedia — multilingual training

The paper trains its multilingual tokenizers on CulturaX, which is gated; Wikipedia stands
in for it. This keeps the first 8,000 articles per language of the 2023-11-01 dump whose
stripped text has at least 50 characters, the loader rule of Table E1; it needs
`pip install datasets`.

```python
import json
import os

from datasets import load_dataset

os.makedirs("datasets/wikipedia", exist_ok=True)
for lang in ["en", "de", "nl", "fr", "es", "it", "ru", "uk", "pl", "zh", "ja", "ko", "tr",
             "fi", "th", "ar"]:
    rows = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train",
                        streaming=True)
    with open(f"datasets/wikipedia/{lang}.jsonl", "w", encoding="utf-8") as out:
        kept = 0
        for row in rows:
            text = row["text"].strip()
            if len(text) >= 50:
                out.write(json.dumps({"text": text, "language": lang}, ensure_ascii=False) + "\n")
                kept += 1
                if kept == 8000:
                    break
```

## FLORES-200 — multilingual evaluation

```python
import os
import tarfile
import urllib.request

os.makedirs("datasets/flores200", exist_ok=True)
url = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
with urllib.request.urlopen(url) as response, \
        tarfile.open(fileobj=response, mode="r|gz") as archive:
    for member in archive:
        if "/devtest/" in member.name and member.isfile():
            member.name = "devtest/" + os.path.basename(member.name)
            archive.extract(member, "datasets/flores200")
```

## Disk and memory

The 100k-document Nemotron subset is about 300 MB. The benchmark's defaults (20,000
documents) peak around 2.5 GB of RAM; 100,000 documents around 13 GB, and the
multilingual benchmark around 17 GB.
