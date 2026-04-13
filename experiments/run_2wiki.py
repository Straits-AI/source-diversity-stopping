"""
2WikiMultiHopQA Benchmark — AEA framework evaluation.

Addresses reviewer concern about single-benchmark evaluation by adding a second
established benchmark: 2WikiMultiHopQA, which covers bridge and comparison
question types derived from Wikipedia and Wikidata.

2WikiMultiHopQA question types:
  - bridge: answer requires chaining two facts (A→B→answer)
  - comparison: answer requires comparing attributes of two entities

Supports both real data (HuggingFace) and synthetic fallback.

Key hypothesis: Does the stopping > searching hierarchy hold on a SECOND benchmark?
  - pi_learned_stop > pi_aea_heuristic > pi_ensemble > pi_semantic

Usage
-----
    python experiments/run_2wiki.py

Results are saved to experiments/results/2wiki.json.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ──────────────────────────────────────────────────────────────
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.evaluation.metrics import (
    exact_match,
    f1_score,
    support_recall as sr_metric,
    support_precision as sp_metric,
    utility_at_budget,
    normalize_cost,
)
from experiments.aea.policies.single_substrate import (
    SemanticOnlyPolicy,
    LexicalOnlyPolicy,
)
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy
from experiments.aea.policies.learned_stopping import LearnedStoppingPolicy
from experiments.aea.answer_generator import AnswerGenerator

# ── Constants ─────────────────────────────────────────────────────────────────
N_EXAMPLES = 100
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "2wiki.json"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "stopping_classifier.pkl"

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "REPLACE_WITH_YOUR_OPENROUTER_API_KEY",
)

# Policies for E2E LLM-answer evaluation (subset to control API cost)
LLM_EVAL_POLICIES = {"pi_semantic", "pi_aea_heuristic", "pi_learned_stop"}

# Max ops for normalised cost computation
_MAX_OPS = 18


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic 2WikiMultiHopQA-style dataset
#
# 50 bridge questions (A→B→answer) and 50 comparison questions
# ("Which X, A or B?") with 10 paragraphs per question (2 gold + 8 distractors).
# Entity names drawn from different domains (science, arts, history, geography).
# ─────────────────────────────────────────────────────────────────────────────

# Bridge pool: (entity_a, relation_ab, entity_b, attribute, answer)
# entity_a --relation_ab--> entity_b --attribute--> answer
_BRIDGE_POOL = [
    ("Marie Curie",          "discovered",    "Polonium",              "named after",     "Poland"),
    ("Isaac Newton",         "attended",      "Trinity College",       "located in",      "Cambridge"),
    ("Leonardo da Vinci",    "painted",       "The Last Supper",       "commissioned by", "Ludovico Sforza"),
    ("Albert Einstein",      "developed",     "General Relativity",    "published in",    "1915"),
    ("Galileo Galilei",      "improved",      "the telescope",         "invented by",     "Hans Lippershey"),
    ("Charles Darwin",       "sailed on",     "HMS Beagle",            "captained by",    "Robert FitzRoy"),
    ("Nikola Tesla",         "worked for",    "Thomas Edison",         "founded",         "General Electric"),
    ("Ada Lovelace",         "collaborated with", "Charles Babbage",   "invented",        "the Analytical Engine"),
    ("Gregor Mendel",        "studied at",    "University of Vienna",  "located in",      "Austria"),
    ("Louis Pasteur",        "disproved",     "spontaneous generation","challenged by",   "Felix Pouchet"),
    ("Wolfgang Amadeus Mozart", "composed",   "The Magic Flute",       "premiered in",    "Vienna"),
    ("Ludwig van Beethoven", "studied under", "Joseph Haydn",          "born in",         "Rohrau"),
    ("Johann Sebastian Bach","served at",     "St. Thomas Church",     "located in",      "Leipzig"),
    ("Franz Schubert",       "admired",       "Ludwig van Beethoven",  "composed",        "Symphony No. 9"),
    ("Frederic Chopin",      "born in",       "the Duchy of Warsaw",   "capital",         "Warsaw"),
    ("Napoleon Bonaparte",   "exiled to",     "Saint Helena",          "administered by", "Britain"),
    ("Julius Caesar",        "crossed",       "the Rubicon",           "flows into",      "the Adriatic Sea"),
    ("Cleopatra",            "allied with",   "Julius Caesar",         "born in",         "Alexandria"),
    ("Alexander the Great",  "tutored by",    "Aristotle",             "born in",         "Stagira"),
    ("Genghis Khan",         "founded",       "the Mongol Empire",     "capital",         "Karakorum"),
    ("William Shakespeare",  "wrote",         "Hamlet",                "set in",          "Denmark"),
    ("Jane Austen",          "published",     "Pride and Prejudice",   "set in",          "Hertfordshire"),
    ("Charles Dickens",      "edited",        "Household Words",       "published in",    "London"),
    ("Mark Twain",           "wrote",         "Adventures of Huckleberry Finn", "set on", "Mississippi River"),
    ("Ernest Hemingway",     "lived in",      "Paris",                 "capital of",      "France"),
    ("Pablo Picasso",        "co-founded",    "Cubism",                "alongside",       "Georges Braque"),
    ("Salvador Dali",        "was part of",   "Surrealism",            "founded by",      "Andre Breton"),
    ("Vincent van Gogh",     "stayed at",     "the Yellow House",      "located in",      "Arles"),
    ("Claude Monet",         "founded",       "Impressionism",         "named after",     "Impression, Sunrise"),
    ("Rembrandt",            "painted",       "The Night Watch",       "displayed at",    "Rijksmuseum"),
    ("Mahatma Gandhi",       "led",           "the Salt March",        "protested against","British salt tax"),
    ("Nelson Mandela",       "imprisoned at", "Robben Island",         "located near",    "Cape Town"),
    ("Martin Luther King Jr","delivered",     "I Have a Dream",        "given at",        "Lincoln Memorial"),
    ("Simone de Beauvoir",   "authored",      "The Second Sex",        "published in",    "1949"),
    ("Virginia Woolf",       "founded",       "the Bloomsbury Group",  "based in",        "London"),
    ("Sigmund Freud",        "practiced in",  "Vienna",                "capital of",      "Austria"),
    ("Carl Jung",            "studied under", "Sigmund Freud",         "born in",         "Kesswil"),
    ("Karl Marx",            "co-authored",   "The Communist Manifesto","written with",   "Friedrich Engels"),
    ("Friedrich Nietzsche",  "studied at",    "University of Leipzig",  "located in",     "Germany"),
    ("Immanuel Kant",        "taught at",     "University of Königsberg","located in",    "Prussia"),
    ("Marco Polo",           "traveled to",   "China",                  "ruled by",       "Kublai Khan"),
    ("Christopher Columbus", "sailed for",    "Spain",                  "reigned by",     "Queen Isabella"),
    ("Ferdinand Magellan",   "led",           "the circumnavigation",   "completed by",   "Juan Sebastian Elcano"),
    ("Vasco da Gama",        "reached",       "Calicut",                "located in",     "India"),
    ("James Cook",           "charted",       "New Zealand",            "inhabited by",   "the Maori"),
    ("Nicolaus Copernicus",  "proposed",      "heliocentrism",          "later proved by","Galileo"),
    ("Johannes Kepler",      "served under",  "Tycho Brahe",           "based in",       "Prague"),
    ("Dmitri Mendeleev",     "created",       "the Periodic Table",    "first published in","1869"),
    ("Max Planck",           "proposed",      "quantum theory",         "extended by",    "Albert Einstein"),
    ("Werner Heisenberg",    "formulated",    "the Uncertainty Principle","basis of",     "quantum mechanics"),
]

# Comparison pool: (entity_a, entity_b, attribute, value_a, value_b, answer)
# "Which is [older/larger/earlier]?" type questions
_COMPARISON_POOL = [
    ("The Colosseum",       "The Pantheon",         "older",    "70 AD",   "125 AD",  "The Pantheon"),
    ("Mount Everest",       "K2",                   "taller",   "8849m",   "8611m",   "Mount Everest"),
    ("Amazon River",        "Nile River",            "longer",   "6400km",  "6650km",  "Nile River"),
    ("The Great Wall",      "Hadrian's Wall",        "longer",   "21196km", "118km",   "The Great Wall"),
    ("Leonardo da Vinci",   "Michelangelo",          "older",    "1452",    "1475",    "Leonardo da Vinci"),
    ("Shakespeare",         "Chaucer",               "older",    "1564",    "1343",    "Chaucer"),
    ("Isaac Newton",        "Gottfried Leibniz",     "older",    "1643",    "1646",    "Isaac Newton"),
    ("Oxford University",   "Cambridge University",  "older",    "1096",    "1209",    "Oxford University"),
    ("Notre-Dame de Paris", "Chartres Cathedral",    "taller",   "96m",     "113m",    "Chartres Cathedral"),
    ("The Eiffel Tower",    "The Statue of Liberty", "taller",   "330m",    "93m",     "The Eiffel Tower"),
    ("Albert Einstein",     "Niels Bohr",            "older",    "1879",    "1885",    "Albert Einstein"),
    ("Beethoven",           "Mozart",                "older",    "1770",    "1756",    "Mozart"),
    ("Plato",               "Aristotle",             "older",    "428 BC",  "384 BC",  "Plato"),
    ("Paris",               "London",                "older",    "52 BC",   "43 AD",   "Paris"),
    ("Rome",                "Athens",                "older",    "753 BC",  "5000 BC", "Athens"),
    ("Africa",              "Asia",                  "larger",   "30.4Mkm2","44.6Mkm2","Asia"),
    ("Pacific Ocean",       "Atlantic Ocean",        "larger",   "165Mkm2", "106Mkm2", "Pacific Ocean"),
    ("China",               "United States",         "larger",   "9.6Mkm2", "9.8Mkm2", "United States"),
    ("Brazil",              "Argentina",             "larger",   "8.5Mkm2", "2.8Mkm2", "Brazil"),
    ("Russia",              "Canada",                "larger",   "17Mkm2",  "10Mkm2",  "Russia"),
    ("Sahara Desert",       "Arabian Desert",        "larger",   "9.2Mkm2", "2.3Mkm2", "Sahara Desert"),
    ("J.K. Rowling",        "Stephen King",          "older",    "1965",    "1947",    "Stephen King"),
    ("Tolstoy",             "Dostoevsky",            "older",    "1828",    "1821",    "Dostoevsky"),
    ("Homer",               "Virgil",                "older",    "800 BC",  "70 BC",   "Homer"),
    ("Dante",               "Petrarch",              "older",    "1265",    "1304",    "Dante"),
    ("The Iliad",           "The Odyssey",           "longer",   "15693 lines","12109 lines","The Iliad"),
    ("Harvard University",  "Yale University",       "older",    "1636",    "1701",    "Harvard University"),
    ("MIT",                 "Caltech",               "older",    "1861",    "1891",    "MIT"),
    ("Stanford University", "Princeton University",  "older",    "1885",    "1746",    "Princeton University"),
    ("Tesla",               "Edison",                "older",    "1856",    "1847",    "Edison"),
    ("Pythagoras",          "Euclid",                "older",    "570 BC",  "300 BC",  "Pythagoras"),
    ("Archimedes",          "Euclid",                "older",    "287 BC",  "300 BC",  "Euclid"),
    ("Copernicus",          "Galileo",               "older",    "1473",    "1564",    "Copernicus"),
    ("Kepler",              "Newton",                "older",    "1571",    "1643",    "Kepler"),
    ("Faraday",             "Maxwell",               "older",    "1791",    "1831",    "Faraday"),
    ("Dickens",             "Austen",                "older",    "1812",    "1775",    "Austen"),
    ("Balzac",              "Flaubert",              "older",    "1799",    "1821",    "Balzac"),
    ("Voltaire",            "Rousseau",              "older",    "1694",    "1712",    "Voltaire"),
    ("Locke",               "Hume",                  "older",    "1632",    "1711",    "Locke"),
    ("Descartes",           "Spinoza",               "older",    "1596",    "1632",    "Descartes"),
    ("Monet",               "Renoir",                "older",    "1840",    "1841",    "Monet"),
    ("Picasso",             "Matisse",               "older",    "1881",    "1869",    "Matisse"),
    ("Rembrandt",           "Vermeer",               "older",    "1606",    "1632",    "Rembrandt"),
    ("Raphael",             "Titian",                "older",    "1483",    "1488",    "Raphael"),
    ("Caravaggio",          "Velazquez",             "older",    "1571",    "1599",    "Caravaggio"),
    ("Lincoln",             "Washington",            "older",    "1809",    "1732",    "Washington"),
    ("Churchill",           "Roosevelt",             "older",    "1874",    "1858",    "Roosevelt"),
    ("Gandhi",              "Nehru",                 "older",    "1869",    "1889",    "Gandhi"),
    ("Marx",                "Engels",                "older",    "1818",    "1820",    "Marx"),
    ("Lenin",               "Stalin",               "older",    "1870",    "1878",    "Lenin"),
]


def _make_bridge_example(idx: int, row: tuple, rng: random.Random) -> dict:
    """Build one bridge-type 2WikiMultiHopQA-style example."""
    entity_a, rel_ab, entity_b, attribute, answer = row

    gold_para_1 = {
        "title": f"{entity_a}",
        "paragraph_text": (
            f"{entity_a} {rel_ab} {entity_b}. "
            f"This is one of the most notable achievements associated with {entity_a}. "
            f"Historians and scholars have extensively documented this relationship."
        ),
        "is_supporting": True,
    }
    gold_para_2 = {
        "title": f"{entity_b}",
        "paragraph_text": (
            f"{entity_b} is {attribute} {answer}. "
            f"This connection has been verified through multiple primary sources. "
            f"The significance of {entity_b} in this context cannot be understated."
        ),
        "is_supporting": True,
    }

    distractors = [
        {
            "title": f"History of {entity_b}",
            "paragraph_text": (
                f"The history of {entity_b} spans many centuries. "
                f"Numerous scholars have examined its origins and development. "
                f"Various interpretations exist regarding its most significant periods."
            ),
            "is_supporting": False,
        },
        {
            "title": f"{entity_a} — Early Life",
            "paragraph_text": (
                f"{entity_a} was recognized early for exceptional talent. "
                f"Their formative years shaped the trajectory of their career. "
                f"Associates and contemporaries noted their remarkable dedication."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Related works by {entity_a}",
            "paragraph_text": (
                f"Beyond {entity_b}, {entity_a} was associated with many other notable endeavors. "
                f"These parallel contributions enriched their overall legacy. "
                f"Scholars continue to debate which work represents their greatest achievement."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Overview of {attribute} relationships",
            "paragraph_text": (
                f"Many entities share {attribute} relationships with others in their field. "
                f"Such connections often reflect deeper organizational or cultural ties. "
                f"Analysis reveals recurring patterns across different domains."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Context of {answer}",
            "paragraph_text": (
                f"{answer} plays a significant role in the broader context of related subjects. "
                f"Its importance is recognized across multiple disciplines. "
                f"Further research has expanded our understanding of its role."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Contemporaries of {entity_a}",
            "paragraph_text": (
                f"In the same era as {entity_a}, several other figures made comparable contributions. "
                f"Cross-disciplinary influences were common among peers. "
                f"Their collective work defined an important period in history."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Legacy and Impact",
            "paragraph_text": (
                f"The legacy of individuals associated with {entity_b} continues to influence modern thinking. "
                f"Educational institutions incorporate these contributions into their curricula. "
                f"Public recognition has grown steadily over the decades."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Further Research on {entity_b}",
            "paragraph_text": (
                f"Ongoing research continues to shed light on {entity_b}. "
                f"New archival discoveries periodically revise our understanding. "
                f"International collaboration has accelerated recent findings."
            ),
            "is_supporting": False,
        },
    ]

    paragraphs = [gold_para_1, gold_para_2] + distractors
    rng.shuffle(paragraphs)

    # Build question in 2Wiki bridge style
    question = f"What is the {attribute} of the {rel_ab.split()[-1]} of {entity_a}?"

    # Supporting facts: (title, sent_idx)
    supporting_facts = [
        [entity_a, 0],
        [entity_b, 0],
    ]

    return {
        "id": f"2wiki_bridge_{idx:03d}",
        "question": question,
        "answer": answer,
        "type": "bridge",
        "supporting_facts": supporting_facts,
        "paragraphs": paragraphs,
    }


def _make_comparison_example(idx: int, row: tuple, rng: random.Random) -> dict:
    """Build one comparison-type 2WikiMultiHopQA-style example."""
    entity_a, entity_b, comparator, value_a, value_b, answer = row

    gold_para_1 = {
        "title": f"{entity_a}",
        "paragraph_text": (
            f"{entity_a} is a well-known entity. "
            f"It was established or born in {value_a}. "
            f"Many records document its founding, creation, or origin date."
        ),
        "is_supporting": True,
    }
    gold_para_2 = {
        "title": f"{entity_b}",
        "paragraph_text": (
            f"{entity_b} is another notable entity in the same category. "
            f"Its origin or establishment date is recorded as {value_b}. "
            f"Comparative studies frequently include both {entity_a} and {entity_b}."
        ),
        "is_supporting": True,
    }

    distractors = [
        {
            "title": f"History of {entity_a}",
            "paragraph_text": (
                f"{entity_a} has a rich history spanning multiple eras. "
                f"Its development has been shaped by many external forces. "
                f"Scholars have traced its evolution through primary and secondary sources."
            ),
            "is_supporting": False,
        },
        {
            "title": f"History of {entity_b}",
            "paragraph_text": (
                f"The history of {entity_b} is intertwined with broader cultural movements. "
                f"Many significant events took place during its formative years. "
                f"Documentation of its early period is extensive."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Comparison of Similar Entities",
            "paragraph_text": (
                f"Comparative analyses of entities similar to {entity_a} and {entity_b} "
                f"reveal interesting patterns. Scholars use multiple criteria to evaluate such comparisons. "
                f"Results depend heavily on the selected metric."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Modern Views on {entity_a}",
            "paragraph_text": (
                f"Contemporary perspectives on {entity_a} emphasize its enduring relevance. "
                f"New research has added nuance to earlier interpretations. "
                f"Public perception has evolved considerably over time."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Legacy of {entity_b}",
            "paragraph_text": (
                f"The legacy of {entity_b} continues to be debated among experts. "
                f"Its influence extends across several generations. "
                f"Primary sources paint a complex picture of its impact."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Related Comparisons in the Field",
            "paragraph_text": (
                f"When comparing entities in this field, researchers consider many factors. "
                f"Size, age, and influence are common metrics. "
                f"Definitive rankings remain contested in academic discourse."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Influence of {entity_a} on Later Developments",
            "paragraph_text": (
                f"{entity_a} had a profound influence on what came after it. "
                f"Subsequent entities often cite it as a foundational reference. "
                f"Its impact can be traced through decades of documented work."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Criticism of {entity_b}",
            "paragraph_text": (
                f"Critics of {entity_b} have raised questions about several of its characteristics. "
                f"These critiques have prompted reassessments in the scholarly community. "
                f"Defenders have responded with counterarguments drawing on empirical evidence."
            ),
            "is_supporting": False,
        },
    ]

    paragraphs = [gold_para_1, gold_para_2] + distractors
    rng.shuffle(paragraphs)

    question = f"Which is {comparator}, {entity_a} or {entity_b}?"

    supporting_facts = [
        [entity_a, 1],  # sentence with value_a
        [entity_b, 1],  # sentence with value_b
    ]

    return {
        "id": f"2wiki_comparison_{idx:03d}",
        "question": question,
        "answer": answer,
        "type": "comparison",
        "supporting_facts": supporting_facts,
        "paragraphs": paragraphs,
    }


def build_synthetic_2wiki(n: int = N_EXAMPLES, seed: int = SEED) -> list[dict]:
    """
    Build a synthetic 2WikiMultiHopQA-style dataset.

    50 bridge questions + 50 comparison questions = 100 total.
    Each question has 10 paragraphs (2 gold + 8 distractors).

    Parameters
    ----------
    n : int
        Total number of examples (split evenly bridge/comparison).
    seed : int
        Random seed.

    Returns
    -------
    list[dict]
        List of 2Wiki-format examples.
    """
    rng = random.Random(seed)
    n_bridge = n // 2
    n_comparison = n - n_bridge

    bridge_rows = _BRIDGE_POOL[:n_bridge]
    comparison_rows = _COMPARISON_POOL[:n_comparison]

    bridge_examples = [
        _make_bridge_example(i, row, rng)
        for i, row in enumerate(bridge_rows)
    ]
    comparison_examples = [
        _make_comparison_example(i, row, rng)
        for i, row in enumerate(comparison_rows)
    ]

    all_examples = bridge_examples + comparison_examples
    rng.shuffle(all_examples)

    print(
        f"Built synthetic 2WikiMultiHopQA: "
        f"{len(bridge_examples)} bridge + {len(comparison_examples)} comparison "
        f"= {len(all_examples)} total"
    )
    return all_examples


# ─────────────────────────────────────────────────────────────────────────────
# Real 2WikiMultiHopQA loading (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

_HF_NAMES = [
    "scholarly-shadows-syndicate/2WikiMultihopQA",
    "2wikimultihopqa",
    "xanhho/2WikiMultihopQA",
    "DAMO-NLP-SG/2WikiMultiHopQA",
    "Tevatron/2wiki",
    "2wiki",
]


def _try_load_huggingface(n: int, seed: int) -> Optional[list[dict]]:
    """Try loading 2WikiMultiHopQA from HuggingFace. Returns None on failure."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets library not available")
        return None

    rng = random.Random(seed)

    for hf_name in _HF_NAMES:
        for split in ("validation", "dev", "test"):
            try:
                print(f"  Trying HF dataset '{hf_name}' split='{split}' …")
                ds = load_dataset(hf_name, split=split, trust_remote_code=True)
                raw = list(ds)
                if not raw:
                    continue
                print(f"    Loaded {len(raw)} examples from {hf_name}/{split}")

                # Filter and sample
                bridge = [e for e in raw if e.get("type") == "bridge"]
                comparison = [e for e in raw if e.get("type") == "comparison"]
                print(f"    bridge={len(bridge)}, comparison={len(comparison)}")

                n_b = min(n // 2, len(bridge))
                n_c = min(n - n_b, len(comparison))
                if n_b + n_c < 20:
                    print(f"    Too few examples ({n_b + n_c}), skipping.")
                    continue

                rng.shuffle(bridge)
                rng.shuffle(comparison)
                selected = bridge[:n_b] + comparison[:n_c]
                rng.shuffle(selected)
                print(f"    Selected {len(selected)} examples ({n_b} bridge + {n_c} comparison)")
                return selected

            except Exception as e:
                print(f"    {hf_name}/{split}: {str(e)[:100]}")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Format conversion: 2Wiki → EvaluationHarness
# ─────────────────────────────────────────────────────────────────────────────

def convert_2wiki_real(raw: dict) -> dict:
    """
    Convert a real 2WikiMultiHopQA example to EvaluationHarness format.

    Real 2Wiki context schema:
        context: [[title, [sentence0, sentence1, ...]], ...]  (like HotpotQA)
    OR:
        paragraphs: [{"title": str, "paragraph_text": str}, ...]

    Returns
    -------
    dict
        EvaluationHarness-compatible example with "content" key.
    """
    # Try HotpotQA-style context first
    context_raw = raw.get("context", [])
    paragraphs = raw.get("paragraphs", [])

    context_docs = []

    if context_raw and isinstance(context_raw[0], (list, tuple)):
        # HotpotQA-style: [[title, [sentences]], ...]
        for entry in context_raw:
            title = entry[0]
            sentences = entry[1]
            content = " ".join(s.strip() for s in sentences)
            context_docs.append({
                "id": title,
                "title": title,
                "content": content,
            })
    elif paragraphs:
        for para in paragraphs:
            title = para.get("title", "")
            text = para.get("paragraph_text", para.get("text", ""))
            context_docs.append({
                "id": title,
                "title": title,
                "content": text,
            })
    else:
        # Fallback: treat context as list of dicts
        for i, entry in enumerate(context_raw):
            if isinstance(entry, dict):
                title = entry.get("title", f"doc_{i}")
                content = entry.get("content", entry.get("text", entry.get("paragraph_text", "")))
            else:
                title = f"doc_{i}"
                content = str(entry)
            context_docs.append({
                "id": title,
                "title": title,
                "content": content,
            })

    # Gold IDs from supporting_facts
    supporting_facts = raw.get("supporting_facts", [])
    seen: set[str] = set()
    gold_ids: list[str] = []
    for sf in supporting_facts:
        if isinstance(sf, (list, tuple)) and len(sf) >= 1:
            title = sf[0]
        elif isinstance(sf, dict):
            title = sf.get("title", "")
        else:
            continue
        if title and title not in seen:
            seen.add(title)
            gold_ids.append(title)

    return {
        "id": raw.get("_id", raw.get("id", "")),
        "question": raw.get("question", ""),
        "answer": raw.get("answer", ""),
        "type": raw.get("type", "bridge"),
        "supporting_facts": supporting_facts,
        "context": context_docs,
        "gold_ids": gold_ids,
    }


def convert_2wiki_synthetic(raw: dict) -> dict:
    """
    Convert a synthetic 2Wiki example (paragraphs list) to EvaluationHarness format.
    """
    paragraphs = raw.get("paragraphs", [])

    context_docs = []
    for para in paragraphs:
        title = para.get("title", "")
        text = para.get("paragraph_text", "")
        context_docs.append({
            "id": title,
            "title": title,
            "content": text,
        })

    gold_ids = [
        para["title"]
        for para in paragraphs
        if para.get("is_supporting", False)
    ]

    return {
        "id": raw.get("id", ""),
        "question": raw.get("question", ""),
        "answer": raw.get("answer", ""),
        "type": raw.get("type", "bridge"),
        "supporting_facts": raw.get("supporting_facts", []),
        "context": context_docs,
        "gold_ids": gold_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Policy and address-space factories
# ─────────────────────────────────────────────────────────────────────────────

def make_policies(model_path: Path = MODEL_PATH) -> list:
    """
    Instantiate the five key policies for 2Wiki evaluation.
    Order matches the hierarchy we want to verify:
      pi_semantic, pi_lexical, pi_ensemble, pi_aea_heuristic, pi_learned_stop
    """
    policies = [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
    ]
    try:
        policies.append(
            LearnedStoppingPolicy(
                model_path=model_path,
                top_k=5,
                max_steps=8,
            )
        )
    except FileNotFoundError as e:
        print(f"  [WARNING] Could not load LearnedStoppingPolicy: {e}")
        print("  Skipping pi_learned_stop.")
    return policies


def make_address_spaces() -> dict:
    """Create fresh address-space instances."""
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM answer generation (E2E eval)
# ─────────────────────────────────────────────────────────────────────────────

def extract_workspace_passages(per_ex: dict, example: dict) -> list[str]:
    """
    Reconstruct passage texts from trace result_items.
    """
    id_to_content = {doc["id"]: doc["content"] for doc in example.get("context", [])}

    best_score: dict[str, float] = {}
    for step in per_ex.get("trace", []):
        for item in step.get("result_items", []):
            doc_id = item.get("id", "")
            score = float(item.get("score", 0.0))
            if doc_id and (doc_id not in best_score or best_score[doc_id] < score):
                best_score[doc_id] = score

    sorted_ids = sorted(best_score, key=lambda k: -best_score[k])
    return [id_to_content[rid] for rid in sorted_ids if rid in id_to_content]


def generate_llm_answers(
    per_example: list[dict],
    dataset: list[dict],
    generator: AnswerGenerator,
    policy_name: str,
) -> list[dict]:
    """
    Generate LLM answers for each example using retrieved workspace passages.
    """
    id_to_example = {ex["id"]: ex for ex in dataset}
    enriched: list[dict] = []
    total = len(per_example)

    for idx, ex in enumerate(per_example):
        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"    [{policy_name}] LLM answer {idx + 1}/{total} …", flush=True)

        example = id_to_example.get(ex.get("id", ""), {})
        passages = extract_workspace_passages(ex, example)
        question = ex.get("question", "")
        gold = ex.get("gold_answer", "")

        llm_answer = generator.generate_answer(question, passages)
        em = exact_match(llm_answer, gold)
        f1 = f1_score(llm_answer, gold)

        enriched_ex = dict(ex)
        enriched_ex["llm_answer"] = llm_answer
        enriched_ex["llm_em"] = em
        enriched_ex["llm_f1"] = f1

        # Compute LLM-based utility
        sp_prec = ex.get("support_precision", 0.0)
        ops = ex.get("operations_used", 0)
        norm_ops = min(1.0, ops / _MAX_OPS)
        enriched_ex["llm_utility_at_budget"] = utility_at_budget(
            answer_score=f1,
            evidence_score=sp_prec,
            cost=norm_ops,
        )
        enriched.append(enriched_ex)

    return enriched


def aggregate_llm(per_example: list[dict]) -> dict:
    """Average LLM metrics across examples."""
    def mean(key: str) -> float:
        vals = [ex[key] for ex in per_example if key in ex and isinstance(ex[key], (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "em": mean("llm_em"),
        "f1": mean("llm_f1"),
        "support_recall": mean("support_recall"),
        "support_precision": mean("support_precision"),
        "avg_ops": mean("operations_used"),
        "utility_at_budget": mean("llm_utility_at_budget"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_by_type(per_example: list[dict], qtype: str) -> dict:
    """Aggregate metrics for a question type subset."""
    subset = [r for r in per_example if r.get("type", "") == qtype]
    if not subset:
        return {}
    keys = ["support_recall", "support_precision", "operations_used",
            "utility_at_budget", "exact_match", "f1"]
    return {
        k: float(np.mean([r[k] for r in subset if k in r and isinstance(r[k], (int, float))]))
        for k in keys
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_2wiki(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
    llm_eval_policies: set = LLM_EVAL_POLICIES,
) -> dict:
    """
    Run all key policies on n_examples 2WikiMultiHopQA questions.

    Tests: does stopping > searching hierarchy hold on a SECOND benchmark?

    Returns
    -------
    dict
        Keyed by policy name; each value has "aggregated", "per_example",
        "by_type" (bridge/comparison breakdown), and optional LLM E2E metrics.
    """
    random.seed(seed)
    np.random.seed(seed)

    # ── Set API key ──────────────────────────────────────────────────────────
    if OPENROUTER_API_KEY:
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("2WikiMultiHopQA Benchmark — AEA Evaluation")
    print("=" * 60)
    print()
    print("Step 1: Loading dataset …")

    raw_examples = _try_load_huggingface(n_examples, seed)
    is_synthetic = False

    if raw_examples is None:
        print("\nAll HuggingFace sources failed. Constructing synthetic 2Wiki-style dataset …")
        raw_examples = build_synthetic_2wiki(n=n_examples, seed=seed)
        is_synthetic = True
        dataset = [convert_2wiki_synthetic(ex) for ex in raw_examples]
    else:
        dataset = [convert_2wiki_real(ex) for ex in raw_examples]

    n_actual = len(dataset)
    n_bridge = sum(1 for ex in dataset if ex.get("type") == "bridge")
    n_comparison = sum(1 for ex in dataset if ex.get("type") == "comparison")

    print(f"\nDataset ready: {n_actual} examples total")
    print(f"  bridge:     {n_bridge}")
    print(f"  comparison: {n_comparison}")
    print(f"  synthetic:  {is_synthetic}")
    print()

    # ── Policy loop ───────────────────────────────────────────────────────────
    print("Step 2: Retrieval evaluation …")
    policies = make_policies()
    generator = AnswerGenerator()
    all_results: dict = {}

    for policy in policies:
        pname = policy.name()
        print(f"\n{'─' * 60}")
        print(f"Policy: {pname}")
        print(f"{'─' * 60}")

        # Retrieval phase
        address_spaces = make_address_spaces()
        harness = EvaluationHarness(
            address_spaces=address_spaces,
            max_steps=10,
            token_budget=4000,
            seed=seed,
        )

        t_start = time.perf_counter()
        result = harness.evaluate(policy, dataset)
        elapsed = time.perf_counter() - t_start
        result["runtime_seconds"] = round(elapsed, 2)
        result["is_synthetic"] = is_synthetic

        # Inject question type into per_example results (harness doesn't pass it through)
        id_to_type = {ex["id"]: ex.get("type", "bridge") for ex in dataset}
        for ex in result["per_example"]:
            ex["type"] = id_to_type.get(ex.get("id", ""), "bridge")

        # Per-type breakdowns
        result["by_type"] = {
            "bridge": _aggregate_by_type(result["per_example"], "bridge"),
            "comparison": _aggregate_by_type(result["per_example"], "comparison"),
        }

        agg = result["aggregated"]
        print(f"  support_recall:    {agg['support_recall']:.4f}")
        print(f"  support_precision: {agg['support_precision']:.4f}")
        print(f"  avg_operations:    {agg['operations_used']:.2f}")
        print(f"  utility@budget:    {agg['utility_at_budget']:.4f}")
        print(f"  n_errors:          {result['n_errors']}")
        print(f"  runtime:           {elapsed:.1f}s")

        # LLM E2E eval for selected policies
        if pname in llm_eval_policies:
            print(f"\n  LLM E2E answer generation for {pname} …", flush=True)
            per_example = generate_llm_answers(
                result["per_example"], dataset, generator, pname
            )
            result["per_example"] = per_example
            result["llm_aggregated"] = aggregate_llm(per_example)
            la = result["llm_aggregated"]
            print(f"  LLM EM:  {la['em']:.4f}")
            print(f"  LLM F1:  {la['f1']:.4f}")
            print(f"  LLM U@B: {la['utility_at_budget']:.4f}")

        all_results[pname] = result

    # ── Print summary tables ──────────────────────────────────────────────────
    _print_summary_tables(all_results, n_actual, n_bridge, n_comparison)

    # ── Key finding ───────────────────────────────────────────────────────────
    _print_key_finding(all_results)

    # ── API usage ─────────────────────────────────────────────────────────────
    usage = generator.usage_summary()
    print(f"\nAPI usage: {usage['total_calls']} calls, "
          f"{usage['total_tokens']} tokens, "
          f"{usage['total_errors']} errors", flush=True)

    # ── Save results ─────────────────────────────────────────────────────────
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        # Slim per_example: drop trace to keep file size manageable
        slim_results = {}
        for pname, res in all_results.items():
            slim_per = [{k: v for k, v in ex.items() if k != "trace"}
                        for ex in res.get("per_example", [])]
            slim_results[pname] = {**res, "per_example": slim_per}

        output = {
            "benchmark": "2WikiMultiHopQA",
            "n_examples": n_actual,
            "n_bridge": n_bridge,
            "n_comparison": n_comparison,
            "is_synthetic": is_synthetic,
            "seed": seed,
            "results": slim_results,
            "api_usage": usage,
        }
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        print(f"\nResults saved to: {results_path}", flush=True)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary_tables(
    all_results: dict,
    n_total: int,
    n_bridge: int,
    n_comparison: int,
) -> None:
    """Print overall and per-type summary tables."""
    col = {
        "Policy": 22, "SR": 15, "SP": 18, "AvgOps": 9, "U@B": 14,
    }

    def _header() -> str:
        return (
            f"| {'Policy':<{col['Policy']}} "
            f"| {'SupportRecall':>{col['SR']}} "
            f"| {'SupportPrecision':>{col['SP']}} "
            f"| {'AvgOps':>{col['AvgOps']}} "
            f"| {'Utility@Budget':>{col['U@B']}} |"
        )

    def _sep() -> str:
        return (
            f"| {'-' * col['Policy']} "
            f"| {'-' * col['SR']} "
            f"| {'-' * col['SP']} "
            f"| {'-' * col['AvgOps']} "
            f"| {'-' * col['U@B']} |"
        )

    def _row(pname: str, agg: dict) -> str:
        return (
            f"| {pname:<{col['Policy']}} "
            f"| {agg.get('support_recall', 0):>{col['SR']}.4f} "
            f"| {agg.get('support_precision', 0):>{col['SP']}.4f} "
            f"| {agg.get('operations_used', 0):>{col['AvgOps']}.2f} "
            f"| {agg.get('utility_at_budget', 0):>{col['U@B']}.4f} |"
        )

    print(f"\n{'=' * 80}")
    print(f"=== 2WikiMultiHopQA Baselines (N={n_total}) ===")
    print(f"{'=' * 80}")

    print(f"\nOverall (bridge={n_bridge}, comparison={n_comparison}):")
    print(_header())
    print(_sep())
    for pname, result in all_results.items():
        print(_row(pname, result["aggregated"]))

    print(f"\nBridge questions (N={n_bridge}):")
    print(_header())
    print(_sep())
    for pname, result in all_results.items():
        bt = result.get("by_type", {}).get("bridge", {})
        if bt:
            print(_row(pname, bt))

    print(f"\nComparison questions (N={n_comparison}):")
    print(_header())
    print(_sep())
    for pname, result in all_results.items():
        ct = result.get("by_type", {}).get("comparison", {})
        if ct:
            print(_row(pname, ct))

    # LLM E2E table
    llm_policies = {k: v for k, v in all_results.items() if "llm_aggregated" in v}
    if llm_policies:
        lcol = {"Policy": 22, "EM": 8, "F1": 8, "SR": 15, "AvgOps": 9, "U@B": 14}
        print(f"\nE2E with LLM Answers (policies with llm_eval=True):")
        print(
            f"| {'Policy':<{lcol['Policy']}} "
            f"| {'EM':>{lcol['EM']}} "
            f"| {'F1':>{lcol['F1']}} "
            f"| {'SupportRecall':>{lcol['SR']}} "
            f"| {'AvgOps':>{lcol['AvgOps']}} "
            f"| {'Utility@Budget':>{lcol['U@B']}} |"
        )
        print(
            f"| {'-' * lcol['Policy']} "
            f"| {'-' * lcol['EM']} "
            f"| {'-' * lcol['F1']} "
            f"| {'-' * lcol['SR']} "
            f"| {'-' * lcol['AvgOps']} "
            f"| {'-' * lcol['U@B']} |"
        )
        for pname, result in llm_policies.items():
            la = result["llm_aggregated"]
            print(
                f"| {pname:<{lcol['Policy']}} "
                f"| {la['em']:>{lcol['EM']}.4f} "
                f"| {la['f1']:>{lcol['F1']}.4f} "
                f"| {la['support_recall']:>{lcol['SR']}.4f} "
                f"| {la['avg_ops']:>{lcol['AvgOps']}.2f} "
                f"| {la['utility_at_budget']:>{lcol['U@B']}.4f} |"
            )

    print()


def _print_key_finding(all_results: dict) -> None:
    """
    Print the key finding: does stopping > searching hold on 2Wiki?

    Hierarchy to verify: pi_learned_stop > pi_aea_heuristic > pi_ensemble > pi_semantic
    """
    print("─" * 60)
    print("KEY FINDING: Does stopping > searching hold on 2WikiMultiHopQA?")
    print("─" * 60)

    hierarchy = ["pi_learned_stop", "pi_aea_heuristic", "pi_ensemble", "pi_semantic", "pi_lexical"]
    present = [p for p in hierarchy if p in all_results]

    if len(present) < 2:
        print("  Not enough policies to compare.")
        return

    print("\nSupportRecall hierarchy (higher = better retrieval):")
    for pname in present:
        recall = all_results[pname]["aggregated"]["support_recall"]
        print(f"  {pname:<25} {recall:.4f}")

    # Check if hierarchy holds
    recalls = [(p, all_results[p]["aggregated"]["support_recall"]) for p in present]
    sorted_recalls = sorted(recalls, key=lambda x: -x[1])
    print(f"\n  Ranked by SupportRecall: {' > '.join(p for p, _ in sorted_recalls)}")

    # LLM E2E comparison if available
    llm_present = [p for p in hierarchy if p in all_results and "llm_aggregated" in all_results[p]]
    if len(llm_present) >= 2:
        print("\nLLM F1 hierarchy (E2E answer quality):")
        llm_f1s = [(p, all_results[p]["llm_aggregated"]["f1"]) for p in llm_present]
        llm_f1s.sort(key=lambda x: -x[1])
        for pname, f1 in llm_f1s:
            print(f"  {pname:<25} {f1:.4f}")

        # Does best stopping beat best searching?
        stopping_policies = [p for p in llm_present if "stop" in p or "heuristic" in p]
        searching_policies = [p for p in llm_present if p not in stopping_policies]
        if stopping_policies and searching_policies:
            best_stopping = max(
                all_results[p]["llm_aggregated"]["f1"] for p in stopping_policies
            )
            best_searching = max(
                all_results[p]["llm_aggregated"]["f1"] for p in searching_policies
            )
            delta = best_stopping - best_searching
            verdict = "YES" if delta >= 0 else "NO"
            print(f"\n  Stopping > Searching: {verdict}  "
                  f"(delta F1 = {delta:+.4f})")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_2wiki()
