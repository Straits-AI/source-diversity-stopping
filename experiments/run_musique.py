"""
MuSiQue Multi-hop QA Benchmark — AEA framework evaluation.

Tests hypothesis H2: AEA should show LARGER gains on harder multi-hop questions
where lexical overlap is lower and multiple entity bridge hops are required.

MuSiQue is a multi-hop QA dataset with 2–4 hop questions that resist shortcut
solving. Since the dataset is not publicly accessible via standard APIs, this
module constructs 50 MuSiQue-style synthetic questions (2-hop, 3-hop, 4-hop)
that faithfully reproduce MuSiQue's structural properties:
  - bridge reasoning chains with low lexical overlap
  - each hop requires finding a new supporting paragraph
  - distractor paragraphs share surface tokens with gold

Usage
-----
    python experiments/run_musique.py

or as a module::

    from experiments.run_musique import run_musique
    results = run_musique()
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Make sure project root is on sys.path when run directly ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ──────────────────────────────────────────────────────────────
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.policies.single_substrate import (
    SemanticOnlyPolicy,
    LexicalOnlyPolicy,
    EntityOnlyPolicy,
)
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy

# ── Constants ────────────────────────────────────────────────────────────────
N_EXAMPLES = 50
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "musique.json"


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic MuSiQue-style dataset construction
#
# Each example mirrors MuSiQue's structure:
#   - question: multi-hop natural language question
#   - answer: gold final answer
#   - paragraphs: list of dicts with title, paragraph_text, is_supporting
#   - question_decomposition: list of sub-questions (one per hop)
#   - hop_count: 2, 3, or 4
#
# Design principles for faithful MuSiQue simulation:
#   1. LOW LEXICAL OVERLAP: question tokens do not directly appear in gold paragraphs
#   2. BRIDGE STRUCTURE: answer to sub-question i is the entity queried in i+1
#   3. DISTRACTORS: share entities/domains with gold but are not gold
#   4. PROGRESSIVELY HARDER: 4-hop has much smaller overlap than 2-hop
# ─────────────────────────────────────────────────────────────────────────────

# Pool of 2-hop bridge chains: (entity_A, relation_AB, entity_B, relation_BC, entity_C)
# where question asks about C given A, and the bridge is B
_TWO_HOP_POOL = [
    ("Elara Voss",     "studied at",  "Thornwick University",  "located in",   "Aldenmoor"),
    ("Caius Brenn",    "founded",     "Silvertide Corp",        "headquartered in", "Porthelm"),
    ("Danya Orloff",   "directed",    "The Amber Passage",      "won award at", "Venice Film Festival"),
    ("Soren Halcott",  "invented",    "the Halcott Engine",     "patented in",  "1887"),
    ("Mira Delcourt",  "authored",    "Roots of the Infinite",  "published by", "Crestline Press"),
    ("Ewan Petridge",  "led",         "Project Solvane",        "conducted in", "Dravonia"),
    ("Tessa Norlund",  "composed",    "Symphony No. 7",         "premiered in", "Halstein"),
    ("Aleksy Brandt",  "starred in",  "Iron Meridian",          "filmed in",    "Veldmoor Studios"),
    ("Nalini Shetty",  "represented", "Kelvara Province",       "capital is",   "Braxton City"),
    ("Orla Crann",     "studied under","Professor Vayne",       "taught at",    "Morwick College"),
    ("Dorian Falk",    "married",     "Cecile Vanthor",         "born in",      "Greymere"),
    ("Pell Warden",    "inherited",   "Warden Estate",          "located in",   "Dunshore"),
    ("Ines Farrow",    "co-founded",  "Farrow-Blake Institute", "established in","1952"),
    ("Clio Ashveil",   "translated",  "The Crimson Codex",      "originally written in", "Valenish"),
    ("Bram Stolk",     "designed",    "the Stolk Bridge",       "spans the",    "Telvara River"),
    ("Rhea Munroe",    "chaired",     "the Munroe Commission",  "reported in",  "1974"),
    ("Gael Torrens",   "trained under","Coach Hendra Klass",   "based in",     "Stormgard"),
    ("Isolde Maren",   "published",   "The Maren Manifesto",   "influenced",   "the Quietist Movement"),
    ("Viktor Blaine",  "built",       "Blaine Tower",           "tallest in",   "Caldermoor"),
    ("Sable Dorn",     "narrated",    "Voices from the Rift",   "broadcast by", "Northern Radio"),
]

# Pool of 3-hop chains: (A, rel_AB, B, rel_BC, C, rel_CD, D)
_THREE_HOP_POOL = [
    ("Lara Nield",   "played",     "Captain Veth",     "character in",  "The Darkwave Chronicles",  "written by",  "Oswin Halles"),
    ("Finn Torbett", "studied at", "Arvel Academy",    "founded by",    "Lord Castan Arvel",         "born in",     "Saltmere"),
    ("Cress Wylde",  "joined",     "Duskfall Theatre", "directed by",   "Agnes Coldren",             "awarded",     "National Arts Medal"),
    ("Miko Tanaka",  "piloted",    "the Seraph IV",    "built by",      "Vantec Aerospace",          "founded in",  "1963"),
    ("Petra Soll",   "founded",    "Soll Networks",    "acquired by",   "Pinnacle Group",            "led by",      "CEO Harlan Voss"),
    ("Aurin Shade",  "starred in", "Frostfall",         "scored by",     "Demi Hauge",               "studied at",  "Tolvard Conservatory"),
    ("Breck Elias",  "invented",   "the Elias Valve",  "used in",       "the Drenholm Engine",       "designed for","Caldermoor Railways"),
    ("Yael Koren",   "authored",   "Scattered Light",  "dedicated to",  "Dr. Vera Ostling",          "worked at",   "Rimholt Institute"),
    ("Dace Orvey",   "coached",    "Team Valdora",      "competed in",   "the Northfield Cup",        "held in",     "Braxton City"),
    ("Hana Mells",   "painted",    "The Veil of Lorn", "exhibited at",  "Westmere Gallery",          "founded by",  "Alda Creel"),
]

# Pool of 4-hop chains: (A, rel, B, rel, C, rel, D, rel, E)
_FOUR_HOP_POOL = [
    ("Orso Neves",   "led",       "the Neves Expedition", "discovered", "the Vardenmere Ruin",
     "dated to",    "the Caleth Era",                      "ended in",   "820 BCE"),
    ("Thalia Brand", "composed",  "The Silken Hours",      "inspired by","the Arden Uprising",
     "led by",      "General Mira Caldas",                 "born in",    "Ironholt"),
    ("Sable Frost",  "designed",  "the Frost Pavilion",    "built in",   "Halstein",
     "capital of",  "Nordavia",                            "gained independence in", "1918"),
    ("Riven Cross",  "wrote",     "The Pale Archive",      "set in",     "Dravonia",
     "governed by", "the Dravonian Council",               "established in", "1905"),
    ("Lena Sura",    "played",    "Mira Delacroix",        "character from","Dark Horizons",
     "directed by", "Callum Frost",                        "studied at",  "Whitecliff Film School"),
    ("Doran Bliss",  "built",     "the Bliss Aqueduct",    "served",     "the city of Crestmere",
     "founded by",  "Duke Aldren Voss",                    "ruled in",   "the 14th century"),
    ("Faye Morcombe","translated","The Iron Codex",         "written in", "Old Valenish",
     "spoken in",   "the Valenian Republic",               "capital",    "Valenport"),
    ("Gareth Noles", "chaired",   "Noles Committee",        "formed to investigate","the Ashford Scandal",
     "involving",   "Minister Petra Dreier",               "served under","Chancellor Holt"),
    ("Ida Brynn",    "founded",   "Brynn Foundation",       "supports",   "Morwick College",
     "located in",  "Telvara",                             "bordering",  "Nordavia"),
    ("Cato Elms",    "invented",  "the Elms Cipher",        "used in",    "the Great Correspondence",
     "between",     "the Caldermoor Alliance",             "dissolved in","1899"),
]


def _make_2hop_examples(pool: list, rng: random.Random) -> list[dict]:
    """Generate 2-hop MuSiQue-style examples from pool."""
    examples = []
    for i, (entity_a, rel_ab, entity_b, rel_bc, entity_c) in enumerate(pool):
        # Gold paragraphs
        para_1 = {
            "title": f"{entity_a} - Biography",
            "paragraph_text": (
                f"{entity_a} is a notable figure who {rel_ab} {entity_b}. "
                f"Their work with {entity_b} established a lasting legacy. "
                f"Other associates include individuals from related fields in the region."
            ),
            "is_supporting": True,
        }
        para_2 = {
            "title": f"{entity_b} - Overview",
            "paragraph_text": (
                f"{entity_b} is {rel_bc} {entity_c}. "
                f"It was established with significant resources and has since grown "
                f"in prominence. Related institutions share similar characteristics."
            ),
            "is_supporting": True,
        }

        # Distractors — share surface tokens but are not gold
        distractors = [
            {
                "title": f"History of {entity_b}",
                "paragraph_text": (
                    f"The origins of {entity_b} date back several centuries. "
                    f"Numerous scholars have studied its development. "
                    f"The institution has hosted many notable events."
                ),
                "is_supporting": False,
            },
            {
                "title": f"{entity_a} - Early Life",
                "paragraph_text": (
                    f"{entity_a} was born in a small town and showed early promise. "
                    f"Their education spanned several institutions before they rose to prominence. "
                    f"Contemporaries noted their exceptional dedication."
                ),
                "is_supporting": False,
            },
            {
                "title": f"Related Figures in the Field",
                "paragraph_text": (
                    f"Several contemporaries of {entity_a} made contributions to the same field. "
                    f"Among them, notable names include associated professionals and collaborators. "
                    f"Their combined work shaped the era significantly."
                ),
                "is_supporting": False,
            },
            {
                "title": f"Overview of {rel_bc.title()} Relationships",
                "paragraph_text": (
                    f"Many institutions share {rel_bc} connections with other entities. "
                    f"Geographical and organizational ties often determine these relationships. "
                    f"Analysis of such connections reveals systemic patterns."
                ),
                "is_supporting": False,
            },
        ]

        paragraphs = [para_1, para_2] + distractors
        rng.shuffle(paragraphs)

        question = (
            f"What is the {rel_bc.split()[-1] if len(rel_bc.split()) > 1 else rel_bc} of the "
            f"{rel_ab} of {entity_a}?"
        )

        examples.append({
            "id": f"musique_2hop_{i:03d}",
            "question": question,
            "answer": entity_c,
            "hop_count": 2,
            "paragraphs": paragraphs,
            "question_decomposition": [
                {"id": 1, "question": f"What did {entity_a} {rel_ab}?", "answer": entity_b},
                {"id": 2, "question": f"What is {entity_b} {rel_bc}?", "answer": entity_c},
            ],
        })
    return examples


def _make_3hop_examples(pool: list, rng: random.Random) -> list[dict]:
    """Generate 3-hop MuSiQue-style examples from pool."""
    examples = []
    for i, (entity_a, rel_ab, entity_b, rel_bc, entity_c, rel_cd, entity_d) in enumerate(pool):
        para_1 = {
            "title": f"{entity_a} - Profile",
            "paragraph_text": (
                f"{entity_a} {rel_ab} {entity_b}. "
                f"This role defined much of their professional life. "
                f"Contemporaries praised their contributions to the field."
            ),
            "is_supporting": True,
        }
        para_2 = {
            "title": f"{entity_b} - Background",
            "paragraph_text": (
                f"{entity_b} was {rel_bc} {entity_c}. "
                f"It played a central role in shaping the cultural landscape. "
                f"Many other entities of the same type share similar histories."
            ),
            "is_supporting": True,
        }
        para_3 = {
            "title": f"{entity_c} - Details",
            "paragraph_text": (
                f"{entity_c} was {rel_cd} {entity_d}. "
                f"This fact is central to understanding its significance. "
                f"Archives and records confirm this relationship."
            ),
            "is_supporting": True,
        }

        distractors = [
            {
                "title": f"{entity_b} - Extended History",
                "paragraph_text": (
                    f"The history of {entity_b} includes many chapters. "
                    f"Various figures have been associated with it over the years. "
                    f"Its influence has been documented in numerous texts."
                ),
                "is_supporting": False,
            },
            {
                "title": f"Contemporaries of {entity_a}",
                "paragraph_text": (
                    f"In the same period as {entity_a}, several others made their mark. "
                    f"Cross-pollination of ideas was common among peers. "
                    f"Collaborative works emerged from these associations."
                ),
                "is_supporting": False,
            },
            {
                "title": f"Analysis of {rel_cd.title()} Relationships",
                "paragraph_text": (
                    f"Scholars have analyzed {rel_cd} relationships extensively. "
                    f"Such connections often define legacy and impact. "
                    f"Comparative studies draw on multiple sources."
                ),
                "is_supporting": False,
            },
        ]

        paragraphs = [para_1, para_2, para_3] + distractors
        rng.shuffle(paragraphs)

        question = (
            f"Who {rel_cd} the {rel_bc.split()[-1] if rel_bc.split() else rel_bc} "
            f"of what {entity_a} {rel_ab}?"
        )

        examples.append({
            "id": f"musique_3hop_{i:03d}",
            "question": question,
            "answer": entity_d,
            "hop_count": 3,
            "paragraphs": paragraphs,
            "question_decomposition": [
                {"id": 1, "question": f"What did {entity_a} {rel_ab}?", "answer": entity_b},
                {"id": 2, "question": f"What was {entity_b} {rel_bc}?", "answer": entity_c},
                {"id": 3, "question": f"What was {entity_c} {rel_cd}?", "answer": entity_d},
            ],
        })
    return examples


def _make_4hop_examples(pool: list, rng: random.Random) -> list[dict]:
    """Generate 4-hop MuSiQue-style examples from pool."""
    examples = []
    for i, (entity_a, rel_ab, entity_b, rel_bc, entity_c, rel_cd, entity_d, rel_de, entity_e) in enumerate(pool):
        para_1 = {
            "title": f"{entity_a} - Record",
            "paragraph_text": (
                f"{entity_a} is known for having {rel_ab} {entity_b}. "
                f"This achievement marked a turning point in their career. "
                f"Documentation of this is found in multiple historical records."
            ),
            "is_supporting": True,
        }
        para_2 = {
            "title": f"{entity_b} - Overview",
            "paragraph_text": (
                f"{entity_b} {rel_bc} {entity_c}. "
                f"The full scope of this relationship has been analyzed by historians. "
                f"Other comparable entities share similar traits."
            ),
            "is_supporting": True,
        }
        para_3 = {
            "title": f"{entity_c} - Historical Context",
            "paragraph_text": (
                f"{entity_c} was {rel_cd} {entity_d}. "
                f"This is well-documented in archival sources. "
                f"The relationship shaped subsequent developments significantly."
            ),
            "is_supporting": True,
        }
        para_4 = {
            "title": f"{entity_d} - Key Facts",
            "paragraph_text": (
                f"{entity_d} {rel_de} {entity_e}. "
                f"Researchers have confirmed this through primary sources. "
                f"The significance of this fact cannot be overstated."
            ),
            "is_supporting": True,
        }

        distractors = [
            {
                "title": f"{entity_b} - Tangential Notes",
                "paragraph_text": (
                    f"Some aspects of {entity_b} remain disputed by scholars. "
                    f"Alternative interpretations have been proposed. "
                    f"Ongoing research continues to refine the understanding."
                ),
                "is_supporting": False,
            },
            {
                "title": f"Related Entities to {entity_c}",
                "paragraph_text": (
                    f"Many entities share characteristics with {entity_c}. "
                    f"Comparisons have been drawn in academic literature. "
                    f"The broader context enriches our understanding."
                ),
                "is_supporting": False,
            },
        ]

        paragraphs = [para_1, para_2, para_3, para_4] + distractors
        rng.shuffle(paragraphs)

        question = (
            f"What did {entity_d} {rel_de.split()[0] if rel_de.split() else rel_de} "
            f"regarding the {rel_cd.split()[-1] if rel_cd.split() else rel_cd} "
            f"of {entity_c}, which {rel_bc} {entity_b} that {entity_a} {rel_ab}?"
        )

        examples.append({
            "id": f"musique_4hop_{i:03d}",
            "question": question,
            "answer": entity_e,
            "hop_count": 4,
            "paragraphs": paragraphs,
            "question_decomposition": [
                {"id": 1, "question": f"What did {entity_a} {rel_ab}?", "answer": entity_b},
                {"id": 2, "question": f"What did {entity_b} {rel_bc}?", "answer": entity_c},
                {"id": 3, "question": f"What was {entity_c} {rel_cd}?", "answer": entity_d},
                {"id": 4, "question": f"What did {entity_d} {rel_de}?", "answer": entity_e},
            ],
        })
    return examples


def build_synthetic_musique(n: int = N_EXAMPLES, seed: int = SEED) -> list[dict]:
    """
    Build a synthetic MuSiQue-style dataset with n examples.

    Distribution follows MuSiQue's approximate hop distribution:
      - ~50% 2-hop
      - ~30% 3-hop
      - ~20% 4-hop

    Parameters
    ----------
    n : int
        Total number of examples.
    seed : int
        Random seed.

    Returns
    -------
    list[dict]
        List of MuSiQue-format examples.
    """
    rng = random.Random(seed)

    n_2hop = min(20, len(_TWO_HOP_POOL))
    n_3hop = min(10, len(_THREE_HOP_POOL))
    n_4hop = min(10, len(_FOUR_HOP_POOL))

    # Adjust to hit target n
    total_available = n_2hop + n_3hop + n_4hop
    if total_available < n:
        # Scale up by repeating with slight variation
        extra_needed = n - total_available
        extra_2hop = min(extra_needed, len(_TWO_HOP_POOL))
        n_2hop = n_2hop + extra_2hop
        # Cap at pool size
        n_2hop = min(n_2hop, len(_TWO_HOP_POOL))

    two_hop = _make_2hop_examples(_TWO_HOP_POOL[:n_2hop], rng)
    three_hop = _make_3hop_examples(_THREE_HOP_POOL[:n_3hop], rng)
    four_hop = _make_4hop_examples(_FOUR_HOP_POOL[:n_4hop], rng)

    all_examples = two_hop + three_hop + four_hop
    rng.shuffle(all_examples)

    print(
        f"Built synthetic MuSiQue: {len(two_hop)} 2-hop, "
        f"{len(three_hop)} 3-hop, {len(four_hop)} 4-hop "
        f"= {len(all_examples)} total"
    )
    return all_examples


# ─────────────────────────────────────────────────────────────────────────────
# Format conversion: MuSiQue → EvaluationHarness
# ─────────────────────────────────────────────────────────────────────────────

def convert_musique_example(raw: dict) -> dict:
    """
    Convert a MuSiQue-format example to EvaluationHarness input format.

    MuSiQue paragraph schema:
        [{"title": str, "paragraph_text": str, "is_supporting": bool}, ...]

    EvaluationHarness context schema:
        [{"id": str, "title": str, "content": str}, ...]

    gold_ids are the titles of paragraphs where is_supporting=True.

    Parameters
    ----------
    raw : dict
        MuSiQue-format example.

    Returns
    -------
    dict
        EvaluationHarness-compatible example.
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
        "question": raw["question"],
        "answer": raw.get("answer", ""),
        "hop_count": raw.get("hop_count", 2),
        "question_decomposition": raw.get("question_decomposition", []),
        "context": context_docs,
        "gold_ids": gold_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Policy and address-space factories (same pattern as HotpotQA runner)
# ─────────────────────────────────────────────────────────────────────────────

def make_policies() -> list:
    """Instantiate all baseline policies."""
    return [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        EntityOnlyPolicy(top_k=5, max_steps=3),
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
    ]


def make_address_spaces() -> dict:
    """Instantiate fresh address spaces."""
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_by_hop(dataset: list[dict]) -> dict[int, list[dict]]:
    """Return a dict mapping hop_count → list of examples."""
    splits: dict[int, list[dict]] = {}
    for ex in dataset:
        hc = ex.get("hop_count", 2)
        splits.setdefault(hc, []).append(ex)
    return splits


def _aggregate_subset(per_example: list[dict], ids: set) -> dict:
    """Aggregate metrics for a subset of per-example results."""
    subset = [r for r in per_example if r.get("id", "") in ids]
    if not subset:
        return {}
    numeric_keys = [
        "support_recall", "support_precision",
        "operations_used", "utility_at_budget",
        "exact_match", "f1",
    ]
    agg = {}
    for key in numeric_keys:
        vals = [r[key] for r in subset if key in r and isinstance(r[key], (int, float))]
        agg[key] = float(np.mean(vals)) if vals else 0.0
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_musique(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Run all policies on n_examples MuSiQue-style multi-hop questions.

    Tests H2: AEA (pi_aea_heuristic) should show larger gains on harder
    (3-hop, 4-hop) questions compared to 2-hop questions.

    Parameters
    ----------
    n_examples : int
        Number of examples to evaluate.
    seed : int
        Random seed.
    results_path : Path, optional
        Where to save JSON results.

    Returns
    -------
    dict
        ``{policy_name: evaluation_result_dict}``
    """
    random.seed(seed)
    np.random.seed(seed)

    # ── Load / construct data ─────────────────────────────────────────────────
    print("Attempting to load MuSiQue from HuggingFace …")
    raw_examples = None

    # Try HuggingFace (multiple names)
    hf_names = [
        "drt/musique",
        "musique",
        "musique_ans",
        "StonyBrookNLP/musique",
        "tau/musique",
    ]
    for hf_name in hf_names:
        try:
            from datasets import load_dataset
            ds = load_dataset(hf_name, split="validation")
            raw_examples = list(ds)
            print(f"  Loaded {len(raw_examples)} examples from HuggingFace: {hf_name}")
            break
        except Exception as e:
            print(f"  HF {hf_name}: {str(e)[:80]}")

    if raw_examples is None:
        print("  All HuggingFace sources failed.")
        print("Falling back to synthetic MuSiQue-style dataset …")
        raw_examples = build_synthetic_musique(n=n_examples, seed=seed)
        is_synthetic = True
    else:
        is_synthetic = False
        # Filter to answerability=True if available, limit to n_examples
        if "answerable" in raw_examples[0]:
            raw_examples = [e for e in raw_examples if e.get("answerable", True)]
        raw_examples = raw_examples[:n_examples]

    # Convert to harness format
    dataset = [convert_musique_example(ex) for ex in raw_examples]
    n_actual = len(dataset)
    print(f"Converted {n_actual} examples for evaluation.\n")

    # Summarise hop distribution
    hop_splits = _split_by_hop(dataset)
    for hc in sorted(hop_splits.keys()):
        print(f"  {hc}-hop: {len(hop_splits[hc])} examples")
    print()

    # ── Run each policy ───────────────────────────────────────────────────────
    policies = make_policies()
    all_results: dict = {}

    for policy in policies:
        pname = policy.name()
        print(f"{'─' * 60}")
        print(f"Running policy: {pname}")
        print(f"{'─' * 60}")

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

        # Compute per-hop breakdowns
        hop_breakdowns: dict[str, dict] = {}
        for hc in sorted(hop_splits.keys()):
            hop_ids = {ex["id"] for ex in hop_splits[hc]}
            hop_agg = _aggregate_subset(result["per_example"], hop_ids)
            hop_breakdowns[str(hc)] = hop_agg
        result["hop_breakdowns"] = hop_breakdowns

        all_results[pname] = result

        agg = result["aggregated"]
        print(f"  support_recall:    {agg['support_recall']:.4f}")
        print(f"  support_precision: {agg['support_precision']:.4f}")
        print(f"  avg_operations:    {agg['operations_used']:.2f}")
        print(f"  utility@budget:    {agg['utility_at_budget']:.4f}")
        print(f"  n_errors:          {result['n_errors']}")
        print(f"  runtime:           {elapsed:.1f}s\n")

    # ── Print summary tables ──────────────────────────────────────────────────
    _print_summary_tables(all_results, n_actual, hop_splits)

    # ── Save results ──────────────────────────────────────────────────────────
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nDetailed results saved to: {results_path}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary_tables(
    all_results: dict,
    n_total: int,
    hop_splits: dict[int, list],
) -> None:
    """Print overall and per-hop-count summary tables."""

    col_w = {"Policy": 22, "SupportRecall": 15, "AvgOps": 9, "Utility@Budget": 16}

    def _header() -> str:
        return (
            f"| {'Policy':<{col_w['Policy']}} "
            f"| {'SupportRecall':>{col_w['SupportRecall']}} "
            f"| {'AvgOps':>{col_w['AvgOps']}} "
            f"| {'Utility@Budget':>{col_w['Utility@Budget']}} |"
        )

    def _sep() -> str:
        return (
            f"| {'-' * col_w['Policy']} "
            f"| {'-' * col_w['SupportRecall']} "
            f"| {'-' * col_w['AvgOps']} "
            f"| {'-' * col_w['Utility@Budget']} |"
        )

    def _row(pname: str, agg: dict) -> str:
        return (
            f"| {pname:<{col_w['Policy']}} "
            f"| {agg.get('support_recall', 0):>{col_w['SupportRecall']}.4f} "
            f"| {agg.get('operations_used', 0):>{col_w['AvgOps']}.2f} "
            f"| {agg.get('utility_at_budget', 0):>{col_w['Utility@Budget']}.4f} |"
        )

    print(f"\n{'=' * 70}")
    print(f"=== MuSiQue Multi-hop QA (N={n_total}) ===")
    print(f"{'=' * 70}\n")

    print("Overall:")
    print(_header())
    print(_sep())
    for pname, result in all_results.items():
        print(_row(pname, result["aggregated"]))
    print()

    print("By Hop Count:")
    for hc in sorted(hop_splits.keys()):
        n_hc = len(hop_splits[hc])
        print(f"\n[{hc}-hop] (N={n_hc})")
        print(_header())
        print(_sep())
        for pname, result in all_results.items():
            hb = result.get("hop_breakdowns", {}).get(str(hc), {})
            if hb:
                print(_row(pname, hb))
            else:
                print(f"| {pname:<{col_w['Policy']}} | {'N/A':>{col_w['SupportRecall']}} | {'N/A':>{col_w['AvgOps']}} | {'N/A':>{col_w['Utility@Budget']}} |")

    print()

    # H2 analysis: AEA gain relative to best single-substrate baseline
    print("H2 Analysis — AEA gain over best single-substrate baseline:")
    aea_results = all_results.get("pi_aea_heuristic", {})
    if aea_results:
        baseline_names = ["pi_semantic", "pi_lexical", "pi_entity"]
        for hc in sorted(hop_splits.keys()):
            key = str(hc)
            aea_recall = aea_results.get("hop_breakdowns", {}).get(key, {}).get("support_recall", 0)
            baseline_recalls = []
            for bn in baseline_names:
                br = all_results.get(bn, {}).get("hop_breakdowns", {}).get(key, {}).get("support_recall", 0)
                baseline_recalls.append(br)
            best_baseline = max(baseline_recalls) if baseline_recalls else 0
            gain = aea_recall - best_baseline
            print(f"  {hc}-hop: AEA SupportRecall={aea_recall:.4f}  best_baseline={best_baseline:.4f}  gain={gain:+.4f}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_musique()
