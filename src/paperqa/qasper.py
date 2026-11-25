"""QASPER: download, load, chunk papers, and extract gold references for evaluation.

Official data (v0.3): https://allenai.org/data/qasper
Each paper has `full_text` (sections -> paragraphs), `figures_and_tables` (captions)
and `qas` (questions, each answered by one or more annotators with evidence paragraphs).
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR

QASPER_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
SPLIT_FILES = {"train": "qasper-train-v0.3.json", "dev": "qasper-dev-v0.3.json"}
FLOAT_PREFIX = "FLOAT SELECTED"


def normalize_text(text: str) -> str:
    """Collapse whitespace. Used to match evidence strings against paragraph text."""
    return " ".join(text.split())


# --------------------------------------------------------------------------- loading


def download_qasper(data_dir: Path = DATA_DIR) -> Path:
    """Download the official train/dev JSON into data_dir/raw (skips if already there)."""
    raw_dir = data_dir / "raw"
    if all((raw_dir / f).exists() for f in SPLIT_FILES.values()):
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "qasper-train-dev-v0.3.tgz"
    if not archive.exists():
        print(f"Downloading {QASPER_URL} ...")
        urllib.request.urlretrieve(QASPER_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name in SPLIT_FILES.values() and member.isfile():
                src = tar.extractfile(member)
                assert src is not None
                (raw_dir / name).write_bytes(src.read())
    return raw_dir


def load_split(split: str = "dev", data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """Return {paper_id: paper_dict} for a split, downloading the data if needed."""
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {list(SPLIT_FILES)} (the test set is not public)")
    raw_dir = download_qasper(data_dir)
    with open(raw_dir / SPLIT_FILES[split], encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- chunking


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    kind: str  # "abstract" | "paragraph" | "caption"
    section_name: str
    section_idx: int  # -1 for abstract and captions
    paragraph_idx: int
    text: str  # raw text, exactly as in the dataset

    @property
    def document(self) -> str:
        """Text that is embedded and shown to the LLM: section name + paragraph."""
        return f"[{self.section_name}] {self.text}" if self.section_name else self.text

    def metadata(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "kind": self.kind,
            "section_name": self.section_name,
            "section_idx": self.section_idx,
            "paragraph_idx": self.paragraph_idx,
        }


def format_section_name(name: str | None) -> str:
    # QASPER encodes nested sections as "Approach ::: Pretraining".
    if not name:
        return ""
    return " > ".join(part.strip() for part in name.split(":::") if part.strip())


def paper_chunks(paper_id: str, paper: dict) -> list[Chunk]:
    """One chunk per paragraph, plus one for the abstract and one per figure/table caption."""
    chunks: list[Chunk] = []
    if paper.get("abstract", "").strip():
        chunks.append(Chunk(f"{paper_id}#abs", paper_id, "abstract", "Abstract", -1, 0, paper["abstract"]))
    for s_idx, section in enumerate(paper["full_text"]):
        name = format_section_name(section["section_name"])
        for p_idx, para in enumerate(section["paragraphs"]):
            if para.strip():
                chunks.append(Chunk(f"{paper_id}#s{s_idx}p{p_idx}", paper_id, "paragraph", name, s_idx, p_idx, para))
    for c_idx, fig in enumerate(paper.get("figures_and_tables", [])):
        if fig.get("caption", "").strip():
            chunks.append(
                Chunk(f"{paper_id}#cap{c_idx}", paper_id, "caption", "Figures and Tables", -1, c_idx, fig["caption"])
            )
    return chunks


# --------------------------------------------------------------------------- gold references


@dataclass
class Reference:
    """One annotator's answer, in the same string form as the official QASPER evaluator."""

    answer: str
    type: str  # "extractive" | "abstractive" | "boolean" | "none"
    evidence: list[str]  # normalized texts of evidence chunks (only ones that exist in the index)


@dataclass
class Question:
    question_id: str
    paper_id: str
    question: str
    references: list[Reference]

    @property
    def answerable(self) -> bool:
        """False only if every annotator marked the question unanswerable."""
        return any(r.type != "none" for r in self.references)

    @property
    def gold_evidence_sets(self) -> list[set[str]]:
        """Evidence sets of answerable references that point to at least one indexed chunk."""
        return [set(r.evidence) for r in self.references if r.type != "none" and r.evidence]


def evidence_to_chunk_text(evidence: str) -> str:
    """Evidence for tables/figures is 'FLOAT SELECTED: <caption>'; strip the marker."""
    if evidence.startswith(FLOAT_PREFIX):
        evidence = evidence[len(FLOAT_PREFIX) :].lstrip(": ")
    return normalize_text(evidence)


def parse_reference(answer: dict, chunk_texts: set[str]) -> Reference:
    # Same precedence as the official evaluator: extractive > free-form > yes/no.
    if answer["unanswerable"]:
        return Reference("Unanswerable", "none", [])
    if answer["extractive_spans"]:
        text, kind = ", ".join(answer["extractive_spans"]), "extractive"
    elif answer["free_form_answer"]:
        text, kind = answer["free_form_answer"], "abstractive"
    elif answer["yes_no"] is not None:
        text, kind = ("Yes" if answer["yes_no"] else "No"), "boolean"
    else:
        raise ValueError(f"annotation has no answer: {answer}")
    # Some evidence entries are just a section heading (e.g. "Datasets"); those can't be
    # retrieved as a chunk, so they are dropped. Everything else matches a chunk exactly.
    evidence = []
    for e in answer["evidence"]:
        t = evidence_to_chunk_text(e)
        if t in chunk_texts and t not in evidence:
            evidence.append(t)
    return Reference(text, kind, evidence)


def paper_questions(paper_id: str, paper: dict, chunks: list[Chunk] | None = None) -> list[Question]:
    chunks = chunks if chunks is not None else paper_chunks(paper_id, paper)
    chunk_texts = {normalize_text(c.text) for c in chunks}
    return [
        Question(
            question_id=qa["question_id"],
            paper_id=paper_id,
            question=qa["question"],
            references=[parse_reference(a["answer"], chunk_texts) for a in qa["answers"]],
        )
        for qa in paper["qas"]
    ]


def load_questions(papers: dict[str, dict]) -> list[Question]:
    questions = []
    for paper_id, paper in papers.items():
        questions.extend(paper_questions(paper_id, paper))
    return questions
