# 02 · Ingestion Engine

## The cascade principle

Both the loader and the chunker are **cascades**: an ordered list of strategies where each is
tried, validated, and fallen back on. This is the single most important design idea in the
ingestion pipeline, because academic PDFs fail in ways you cannot predict from the filename.

A cascade only helps if failure is *detected*. A splitter that returns one 200 KB "chunk" has not
raised an exception — it has silently produced something useless. So each tier's output is
validated before acceptance.

---

## Loader cascade (`app/ingestion/loaders/pdf.py`)

| Tier | Library | Why it is at this position |
|---|---|---|
| 1 | `pypdf` | Fastest; handles the large majority of digital PDFs |
| 2 | `pdfplumber` | Slower, markedly better on the multi-column layouts journals use |
| 3 | `pymupdf` (`fitz`) | A different engine entirely — opens files the other two cannot |

**Fallbacks apply per page, not per document.** `Firstpaper.pdf` in this corpus is 124 pages. If
pypdf handles 120 and chokes on 4, tier 2 opens exactly those 4 pages. And recovered pages are
merged back **in their original position** — if they were appended at the end instead, every page
number in every citation downstream would be wrong.

Pages still empty after all three tiers are reported, not silently dropped:

```
! 6/124 pages yielded no text (pages [12, 13, 40, 41, 77, 78]) — likely scanned images.
```

That is information you need: it tells you a section of your corpus is invisible to retrieval.

### Other formats

| Format | Loader | Page granularity |
|---|---|---|
| `.txt` / `.md` | stdlib | one page |
| `.html` / `.htm` | BeautifulSoup (lxml → html.parser) | one page |
| `.docx` | `python-docx`, tables included | one page |
| `.pptx` | `python-pptx` | **one page per slide** |

`python-docx`/`python-pptx` are used instead of `unstructured` deliberately: both are tiny, pure
Python, have no system dependencies, and do not break on a fresh install.

---

## Cleaning (`app/ingestion/cleaning.py`)

Four defects, each of which poisons retrieval in a specific way:

| Defect | Example | Effect if untreated |
|---|---|---|
| Ligatures | `beneﬁt`, `ﬁeld` | A query for "benefit" never matches |
| Hyphenation across lines | `haemo-\nprotozoa` | The key term is split in half |
| Soft wrapping | newline every ~80 chars | Sentence boundaries are destroyed, so the chunker splits mid-sentence |
| Running heads | journal name + page number on all 124 pages | Dominates chunk text and drags every embedding toward the same meaningless centroid |

Ligatures are handled by Unicode NFKC normalisation. Running heads are found statistically: a
short line whose digits are masked (`p. 4` and `p. 5` collapse to `p. #`) and which appears on
≥60% of pages is boilerplate. That threshold only applies to documents of 5+ pages — on a 3-page
paper, a line appearing twice is far more likely to be real content.

---

## Chunker cascade (`app/ingestion/chunking/splitter.py`)

| Tier | Strategy | When it is used |
|---|---|---|
| 1 | `RecursiveCharacterTextSplitter` | Default. Splits on the largest natural boundary that fits |
| 2 | Paragraph packing | `langchain-text-splitters` missing, or tier 1 output rejected |
| 3 | Sliding window | Pathological input — a 200 KB document with no newlines |

### Validation

Output is rejected if:

- there are zero chunks, or all are blank
- the longest chunk exceeds `2 × CHUNK_SIZE` (the splitter did not actually split)
- total retained characters are under 50% of the source (content was lost)

The 2× slack is intentional: the recursive splitter legitimately overshoots when a single atomic
unit exceeds `chunk_size`. 2× over is a genuine failure.

### Page attribution

This is what makes "p. 4-5" in a citation trustworthy.

1. Pages are concatenated with a known joiner, recording each page's start offset.
2. Each chunk is located in the source with a cursor-advancing `find`.
3. The chunk's start and end offsets map to page numbers by binary search.

If a chunk cannot be located (a splitter normalised whitespace inside it), the page is recorded as
`None` and displayed as `n/a` — never guessed.

### Sizing

Default `CHUNK_SIZE=1400`, `CHUNK_OVERLAP=200`. For research papers this sits at roughly one to
two paragraphs — large enough to contain a complete finding with its sample size and confidence
interval, small enough that five of them fit comfortably in the context budget.

Chunks shorter than `MIN_CHUNK_CHARS` (120) are dropped. They are page-number leftovers and
footnote markers: they pollute retrieval and cost API quota to embed.

---

## Idempotency and resume (`app/ingestion/manifest.py`)

`ingestion_manifest.json` records, per file: the SHA-256 of its raw bytes, chunk and point counts,
the embedding backend and dimension used, status, and any warnings.

A file is skipped when **all** of these hold:

- the previous run's status was `ok`
- the file's bytes are unchanged
- the embedding dimension has not changed

The manifest is checkpointed after **every file**, so a run interrupted at file 12 of 16 loses at
most one file's work. Re-running the same command resumes.

Writes use write-then-rename, so a crash mid-write leaves the previous manifest intact rather than
a truncated one.

---

## Deduplication — three layers

Duplicate content costs three separate things: embedding quota, vector storage,
and — worst — retrieval slots, because twin passages compete for the same top-5
and crowd out other papers. Each layer catches what the layer below cannot.

### Layer 1 — byte-identical files, before any parsing

`split_byte_duplicates()` groups the corpus by SHA-256 of the raw bytes *before
anything is opened*. A duplicated PDF therefore costs nothing at all: no
extraction, no chunking, no embedding, no points.

The canonical file is the one with the **shortest name, ties broken
alphabetically**. That deterministically prefers `paper.pdf` over `paper-2.pdf`
or `paper (copy).pdf` — almost always the one you meant to keep.

Duplicates are recorded in the manifest with `status="duplicate"` and
`duplicate_of`, and any points a *previous* run indexed under their name are
deleted. Without that purge, a collection built before dedup existed keeps
serving twin passages forever.

### Layer 2 — same text, different bytes, before embedding

Two files can differ as bytes yet extract to identical text: re-saved PDF,
different producer metadata, a re-download. Layer 1 misses these, so after
cleaning, the document's content hash is checked against everything already
indexed (`Manifest.find_duplicate_of`).

The same shortest-name rule applies here. If the file being processed has the
preferred name, the already-indexed twin is *demoted* instead — its points are
deleted and its record flipped to `duplicate`. This makes the outcome
independent of which file happened to be ingested first, which matters when a
manifest was written before dedup existed and has both marked `ok`.

### Layer 3 — repeated chunks and repeated strings

- Chunks whose text is byte-for-byte identical to an earlier chunk **in the same
  document** are dropped before embedding (repeated tables, duplicated
  abstracts).
- `embed_texts()` collapses repeated strings within a call: the text is embedded
  once and the vector fanned back out to every position that shared it.
- The on-disk embedding cache catches repeats *across* runs and documents.

Chunks are **not** deduplicated across different documents. Two papers
legitimately share sentences ("Blood smears were stained with Giemsa"), and
dropping one paper's copy would break that paper's citation for a passage it
genuinely contains.

---

### Deterministic point IDs

```python
point_id = uuid5(NAMESPACE, f"{filename}:{chunk_index}")
```

Re-ingesting a file overwrites its own points instead of duplicating them. Combined with
`delete_by_source()` before upserting, an edited document that got *shorter* leaves no orphaned
tail chunks behind — those would still match searches and cite a version of the text that no
longer exists.

---

## CLI reference

```bash
python -m app.ingestion.processor                    # incremental — skip unchanged files
python -m app.ingestion.processor --wipe             # drop the collection and rebuild
python -m app.ingestion.processor --force            # re-ingest all, keep the collection
python -m app.ingestion.processor --dry-run          # parse + chunk only, zero API calls
python -m app.ingestion.processor --limit 2          # first two files
python -m app.ingestion.processor --file "fnx050.pdf"
python -m app.ingestion.processor DATA/subfolder
```

**Always `--dry-run` first on a new corpus.** It exercises every loader and chunker tier and writes
`processed_data/*.json`. Reading one of those files is the fastest way to catch a badly-extracted
PDF — before you have spent any quota embedding it.

`--dry-run` opens the manifest **read-only**. A dry run never indexes anything, so persisting its
results would mark every file `ok` with zero points and the next real run would skip the whole
corpus. As a second line of defence, `should_skip()` also refuses to trust an `ok` record that has
no indexed points.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Everything ingested |
| 1 | No supported files found, or the path does not exist |
| 2 | Infrastructure problem — Qdrant unreachable, dimension mismatch, no embedding backend |
| 3 | Completed, but one or more files failed |
