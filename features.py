import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import (
    nlp,
    text_to_sequence,
    pos_mats_for_texts,
    extract_verbs,
    link_to_external_knowledge,
    align_pos_to_subwords,
)


def prepare_features(
    data_df,
    vocab: dict,
    max_seq_length: int,
    kb_embeddings,
    tfidf_vectorizer=None,
    verb_vectorizer=None,
    fit_vectorizers: bool = False,
):
    """Prepare features for the custom-transformer (non-LLM) BiMind model.

    Returns:
        sequences, pos_mats, content_features, knowledge_features,
        tfidf_vectorizer, verb_vectorizer
    """
    sequences = [
        text_to_sequence(text, vocab, max_seq_length)
        for text in data_df["statement"]
    ]

    pos_mats = pos_mats_for_texts(data_df["statement"].tolist(), max_seq_length)

    if fit_vectorizers or tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(max_features=150)
        tfidf_statement = tfidf_vectorizer.fit_transform(data_df["statement"]).toarray()
    else:
        tfidf_statement = tfidf_vectorizer.transform(data_df["statement"]).toarray()

    verbs_corpus = []
    for text in data_df["statement"]:
        ex = extract_verbs(text)
        verbs_corpus.append(" ".join(ex) if ex else "no_verb")

    if fit_vectorizers or verb_vectorizer is None:
        verb_vectorizer = TfidfVectorizer(max_features=75)
        tfidf_verbs = verb_vectorizer.fit_transform(verbs_corpus).toarray()
    else:
        tfidf_verbs = verb_vectorizer.transform(verbs_corpus).toarray()

    content_features = np.hstack([tfidf_statement, tfidf_verbs])

    knowledge_features = link_to_external_knowledge(
        data_df["statement"].tolist(), kb_embeddings=kb_embeddings, top_k=3
    )

    print(
        f"POS mats: {pos_mats.shape} | TF-IDF Statement: {tfidf_statement.shape} | "
        f"TF-IDF Verbs: {tfidf_verbs.shape} | Content: {content_features.shape} | "
        f"Knowledge: {knowledge_features.shape}"
    )

    return (
        sequences,
        pos_mats,
        content_features,
        knowledge_features,
        tfidf_vectorizer,
        verb_vectorizer,
    )


def prepare_llm_features(
    texts,
    labels,
    tokenizer,
    nlp_model,
    kb_embeddings,
    max_len: int = 256,
    tfidf_vectorizer=None,
    verb_vectorizer=None,
    fit_vectorizers: bool = False,
):
    """Prepare features for the LLM-backbone BiMind model.

    Returns:
        input_ids, attention_masks, pos_feats, content_features,
        knowledge_features, tfidf_vectorizer, verb_vectorizer
    """
    print("Aligning POS tags to subword tokens...")
    input_ids, attention_masks, pos_feats = align_pos_to_subwords(
        texts, tokenizer, nlp_model, max_len=max_len
    )

    if fit_vectorizers or tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(max_features=150)
        tfidf_statement = tfidf_vectorizer.fit_transform(texts).toarray()
    else:
        tfidf_statement = tfidf_vectorizer.transform(texts).toarray()

    verbs_corpus = []
    for text in texts:
        doc = nlp_model(text)
        verbs = [t.lemma_ for t in doc if t.pos_ == "VERB"]
        verbs_corpus.append(" ".join(verbs) if verbs else "no_verb")

    if fit_vectorizers or verb_vectorizer is None:
        verb_vectorizer = TfidfVectorizer(max_features=75)
        tfidf_verbs = verb_vectorizer.fit_transform(verbs_corpus).toarray()
    else:
        tfidf_verbs = verb_vectorizer.transform(verbs_corpus).toarray()

    content_features = np.hstack([tfidf_statement, tfidf_verbs])

    knowledge_features = link_to_external_knowledge(
        texts, kb_embeddings=kb_embeddings, top_k=3
    )

    print(
        f"Input IDs: {input_ids.shape} | POS feats: {pos_feats.shape} | "
        f"Content: {content_features.shape} | Knowledge: {knowledge_features.shape}"
    )

    return (
        input_ids,
        attention_masks,
        pos_feats,
        content_features,
        knowledge_features,
        tfidf_vectorizer,
        verb_vectorizer,
    )
