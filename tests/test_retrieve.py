from paperqa.retrieve import BM25Index, Hit, rrf_fuse, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the F1-score of BERT?") == ["f1", "score", "bert"]


def test_bm25_ranks_by_term_match_and_rarity():
    docs = [
        tokenize("we train a bert model on squad"),
        tokenize("the model is evaluated on squad and newsqa"),
        tokenize("related work on parsing"),
    ]
    index = BM25Index(docs)
    scores = index.scores(tokenize("which bert model"))
    assert scores[0] > scores[1] > scores[2] == 0.0  # "bert" is rarer than "model"
    assert (index.scores(tokenize("transformers")) == 0).all()  # unknown term
    assert (scores >= 0).all()


def test_bm25_idf_is_positive_for_very_common_terms():
    docs = [["model", "a"], ["model", "b"], ["model", "c"]]  # "model" is in every doc
    assert (BM25Index(docs).scores(["model"]) > 0).all()


def _hit(cid):
    return Hit(cid, "p", "paragraph", "S", cid, 0.0)


def test_rrf_fuse():
    dense = [_hit("a"), _hit("b"), _hit("c")]
    bm25 = [_hit("b"), _hit("d")]
    fused = rrf_fuse([dense, bm25], rrf_k=60)
    assert [h.chunk_id for h in fused] == ["b", "a", "d", "c"]
    assert fused[0].score == 1 / 62 + 1 / 61  # rank 2 in dense, rank 1 in bm25
    assert fused[1].score == 1 / 61
