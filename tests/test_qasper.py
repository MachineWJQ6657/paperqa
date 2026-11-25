from paperqa.qasper import format_section_name, paper_chunks, paper_questions


def _answer(**kw):
    base = {
        "unanswerable": False,
        "extractive_spans": [],
        "yes_no": None,
        "free_form_answer": "",
        "evidence": [],
        "highlighted_evidence": [],
    }
    return {"answer": base | kw}


PAPER = {
    "title": "A Paper",
    "abstract": "We study things.",
    "full_text": [
        {"section_name": "Introduction", "paragraphs": ["Intro para one.", "", "Intro  para two."]},
        {"section_name": "Method ::: Training", "paragraphs": ["We train BERT."]},
    ],
    "figures_and_tables": [{"file": "1-Table1-1.png", "caption": "Table 1: Results on SQuAD."}],
    "qas": [
        {
            "question": "Which model?",
            "question_id": "q1",
            "answers": [
                _answer(extractive_spans=["BERT"], evidence=["We train BERT.", "Training"]),
                _answer(free_form_answer="a BERT model", evidence=["FLOAT SELECTED: Table 1: Results on SQuAD."]),
            ],
        },
        {
            "question": "Is it good?",
            "question_id": "q2",
            "answers": [_answer(yes_no=False, evidence=["Intro para two."]), _answer(unanswerable=True)],
        },
        {"question": "Why?", "question_id": "q3", "answers": [_answer(unanswerable=True)]},
    ],
}


def test_format_section_name():
    assert format_section_name("Method ::: Training") == "Method > Training"
    assert format_section_name(None) == ""


def test_paper_chunks():
    chunks = paper_chunks("p", PAPER)
    ids = [c.chunk_id for c in chunks]
    assert ids == ["p#abs", "p#s0p0", "p#s0p2", "p#s1p0", "p#cap0"]  # empty paragraph skipped
    by_id = {c.chunk_id: c for c in chunks}
    assert by_id["p#s1p0"].document == "[Method > Training] We train BERT."
    assert by_id["p#s0p2"].paragraph_idx == 2
    assert by_id["p#cap0"].kind == "caption"


def test_references_and_evidence_matching():
    q1, q2, q3 = paper_questions("p", PAPER)

    r1, r2 = q1.references
    assert (r1.answer, r1.type) == ("BERT", "extractive")
    assert r1.evidence == ["We train BERT."]  # section heading "Training" is dropped
    assert (r2.answer, r2.type) == ("a BERT model", "abstractive")
    assert r2.evidence == ["Table 1: Results on SQuAD."]  # FLOAT SELECTED prefix stripped

    assert [(r.answer, r.type) for r in q2.references] == [("No", "boolean"), ("Unanswerable", "none")]
    assert q2.references[0].evidence == ["Intro para two."]  # whitespace normalized
    assert q2.answerable and q2.gold_evidence_sets == [{"Intro para two."}]

    assert not q3.answerable
    assert q3.gold_evidence_sets == []
