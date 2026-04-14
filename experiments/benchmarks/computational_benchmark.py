"""
Computational Benchmark for AEA Tool-Execution Experiment.

100 questions that REQUIRE computation, structured as:
  50 comparison  questions — "Which company has higher revenue, A or B?"
  50 arithmetic  questions — "What is the combined population of cities X and Y?"

Each question has 10 paragraphs:
  2 gold  — contain the numbers needed to answer
  8 distractors — plausible but non-gold text

Seed: 42 (all randomness is deterministic).
"""

from __future__ import annotations

import random
import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Raw data pools
# ─────────────────────────────────────────────────────────────────────────────

# Comparison: (entity_a, revenue_a_millions, entity_b, revenue_b_millions, answer)
_COMPARISON_POOL = [
    ("Valtec Industries",      840,   "Norbridge Corp",       620,   "Valtec Industries"),
    ("Greenfield Systems",     310,   "Apex Dynamics",        490,   "Apex Dynamics"),
    ("Crestline Holdings",    1050,   "Duneport Ventures",    780,   "Crestline Holdings"),
    ("Thornwall Enterprises",  225,   "Ironclad Solutions",   380,   "Ironclad Solutions"),
    ("SkyMarket Co",           670,   "PinePath Ltd",         530,   "SkyMarket Co"),
    ("Meridian Foundry",       415,   "Coldstream AG",        560,   "Coldstream AG"),
    ("Harborgate Inc",         990,   "Sunridge Partners",    730,   "Harborgate Inc"),
    ("Polaris Works",          145,   "Ember Capital",        210,   "Ember Capital"),
    ("Redwood Financial",      880,   "Siltstone Corp",       950,   "Siltstone Corp"),
    ("Falcrest Manufacturing", 340,   "Dawnhollow Tech",      270,   "Falcrest Manufacturing"),
    ("Cloudbend Systems",      505,   "Tidewater Group",      490,   "Cloudbend Systems"),
    ("Frostpeak Retail",       760,   "Sandgate Holdings",    820,   "Sandgate Holdings"),
    ("Ironbark Logistics",     430,   "Goldenrod Inc",        395,   "Ironbark Logistics"),
    ("Highcliff Analytics",    185,   "Ravenmore Digital",    260,   "Ravenmore Digital"),
    ("Stonecroft Chemicals",   635,   "Wavehaven Corp",       580,   "Stonecroft Chemicals"),
    ("Duskbell Telecom",       910,   "Brightfall Media",     870,   "Duskbell Telecom"),
    ("Crownfield Pharma",      445,   "Wintera Biotech",      510,   "Wintera Biotech"),
    ("Glassgate Retail",       295,   "Thornwick Goods",      320,   "Thornwick Goods"),
    ("Mooncrest Energy",       700,   "Bramblewood Power",    665,   "Mooncrest Energy"),
    ("Riverstride Tech",       550,   "Dalebrook Software",   575,   "Dalebrook Software"),
    ("Amber Point Capital",    920,   "Blue Ridge Assets",    840,   "Amber Point Capital"),
    ("Cascade Biomedics",      370,   "Delta Therapeutics",   420,   "Delta Therapeutics"),
    ("Eclipso Media",          610,   "Foxhaven Studios",     490,   "Eclipso Media"),
    ("Glenmore Steel",         875,   "Harwick Metals",       790,   "Glenmore Steel"),
    ("Indigo Networks",        255,   "Jetstream Communications", 310, "Jetstream Communications"),
    ("Kestrel Aerospace",     1150,   "Lakewood Aviation",    980,   "Kestrel Aerospace"),
    ("Marlowe Consumer Goods", 430,   "Norden Retail",        400,   "Marlowe Consumer Goods"),
    ("Oakridge Insurance",     765,   "Pinnacle Life",        720,   "Oakridge Insurance"),
    ("Quartz Digital",         290,   "Redrock Tech",         315,   "Redrock Tech"),
    ("Silvergate Finance",     640,   "Thornfield Bank",      580,   "Silvergate Finance"),
    ("Uplift Logistics",       480,   "Velocity Freight",     510,   "Velocity Freight"),
    ("Westlake Chemicals",     835,   "Xara Polymers",        790,   "Westlake Chemicals"),
    ("Yorke Mining",           355,   "Zenith Resources",     410,   "Zenith Resources"),
    ("Altera Health",          270,   "Bravada Pharma",       305,   "Bravada Pharma"),
    ("Cobalt Engineering",     920,   "Dynamo Fabrication",   860,   "Cobalt Engineering"),
    ("Electra Power",          745,   "Fusion Energy",        780,   "Fusion Energy"),
    ("Goldtree Investments",   325,   "Harvest Capital",      290,   "Goldtree Investments"),
    ("Indus Holdings",         680,   "Jupiter Conglomerates", 650,  "Indus Holdings"),
    ("Kelp Bio",               160,   "Litmus Research",      195,   "Litmus Research"),
    ("Madera Timber",          520,   "Northwood Forest Products", 490, "Madera Timber"),
    ("Omega Diagnostics",      385,   "Pathfinder Labs",      420,   "Pathfinder Labs"),
    ("Quantum Semiconductors", 970,   "Resonant Chips",       890,   "Quantum Semiconductors"),
    ("Saltflat Mining",        275,   "Terrain Extraction",   310,   "Terrain Extraction"),
    ("Unified Rail",           730,   "Vector Transit",       695,   "Unified Rail"),
    ("Waveband Radio",         195,   "Xerograph Media",      175,   "Waveband Radio"),
    ("Yonder Agriculture",     340,   "Zenova Farming",       360,   "Zenova Farming"),
    ("Arclight Studios",       585,   "Brightside Entertainment", 510, "Arclight Studios"),
    ("Cinder Robotics",        425,   "Drivetech Automation", 450,   "Drivetech Automation"),
    ("Elevate Logistics",      615,   "Freightco",            590,   "Elevate Logistics"),
    ("Grandeur Hotels",        775,   "Heritage Resorts",     820,   "Heritage Resorts"),
]

# Arithmetic (population): (city_a, pop_a_thousands, city_b, pop_b_thousands)
# Answer = pop_a + pop_b (in thousands)
_ARITHMETIC_POOL = [
    ("Crestfield",    42,    "Harrowgate",    38,    80),
    ("Velstrom",      61,    "Brindlemoor",   28,    89),
    ("Thornwich",     54,    "Aldervane",     47,    101),
    ("Duskhollow",    33,    "Porthmere",     76,    109),
    ("Wychford",      88,    "Greymount",     52,    140),
    ("Oskveld",       31,    "Castleway",     45,    76),
    ("Thornbrook",    70,    "Coldveil",      39,    109),
    ("Strandmore",    58,    "Glenholt",      64,    122),
    ("Salthaven",     82,    "Rimwick",       29,    111),
    ("Fordmere",      49,    "Ashvale",       67,    116),
    ("Crestmont",     55,    "Daleford",      48,    103),
    ("Elmswick",      72,    "Fairhollow",    35,    107),
    ("Graystone",     90,    "Halewood",      44,    134),
    ("Ironvale",      26,    "Jasper Creek",  58,    84),
    ("Kelwood",       63,    "Longmere",      77,    140),
    ("Marsden",       41,    "Northfield",    53,    94),
    ("Oakdale",       87,    "Pinehurst",     32,    119),
    ("Queensbury",    68,    "Riverton",      74,    142),
    ("Silverdale",    39,    "Thorngate",     61,    100),
    ("Underwick",     50,    "Valewood",      83,    133),
    ("Westham",       46,    "Yarrow",        57,    103),
    ("Zephyr Falls",  79,    "Aldgate",       36,    115),
    ("Brentwood",     64,    "Coldbrook",     71,    135),
    ("Dunmore",       48,    "Everton",       89,    137),
    ("Froston",       32,    "Granby",        55,    87),
    ("Harlowe",       76,    "Ivywood",       43,    119),
    ("Jackfield",     60,    "Kenwick",       84,    144),
    ("Larchmont",     37,    "Moorfield",     65,    102),
    ("Newfield",      91,    "Orchard Bay",   28,    119),
    ("Pendleton",     53,    "Queensmere",    72,    125),
    ("Ravenscroft",   45,    "Saltfield",     68,    113),
    ("Thistledown",   80,    "Upnor",         34,    114),
    ("Valgate",       59,    "Whitmore",      78,    137),
    ("Xanfield",      42,    "Yellowstone",   66,    108),
    ("Zenwick",       57,    "Ashford",       49,    106),
    ("Blackmere",     73,    "Clearwater",    61,    134),
    ("Dawnfield",     38,    "Eastwick",      85,    123),
    ("Fallhaven",     54,    "Greenfield",    47,    101),
    ("Hammerwick",    69,    "Iselwick",      33,    102),
    ("Juniper Bay",   82,    "Keldwick",      58,    140),
    ("Lowfield",      43,    "Milford",       77,    120),
    ("Northwick",     66,    "Overton",       51,    117),
    ("Pebblecroft",   29,    "Quinton",       74,    103),
    ("Redfield",      88,    "Sunwick",       40,    128),
    ("Tarnfield",     55,    "Umberwick",     63,    118),
    ("Verdant Hills",  48,   "Wakefield",     79,    127),
    ("Xyston",        34,    "Yarnwick",      67,    101),
    ("Zestfield",     71,    "Arborton",      56,    127),
    ("Birchfield",    62,    "Crossmoor",     45,    107),
    ("Driftwood Bay", 83,    "Ember Vale",    37,    120),
]

# Distractor paragraph templates — vary by index so each distractor is unique
_DISTRACTOR_TEMPLATES = [
    (
        "The {entity} was established in {year} and has since expanded its operations "
        "across multiple sectors. Its headquarters are located in the downtown district "
        "and it employs approximately {employees} staff members."
    ),
    (
        "{entity} is best known for its contributions to the regional economy. "
        "Founded in {year}, the organisation has maintained a consistent presence "
        "in both domestic and international markets."
    ),
    (
        "Industry analysts frequently cite {entity} as an example of sustained "
        "growth. The company was incorporated in {year} and currently occupies "
        "a leading position in its primary segment."
    ),
    (
        "The history of {entity} dates to {year}, when it was incorporated under "
        "local business law. Over the following decades it diversified into "
        "adjacent fields while retaining its original core identity."
    ),
    (
        "{entity} operates in a highly competitive environment. Its founding in "
        "{year} coincided with broader market shifts that shaped its current strategy. "
        "The organisation continues to invest in research and development."
    ),
    (
        "Observers of the {sector} sector often cite {entity} as a benchmark "
        "for operational excellence. The organisation was founded in {year} and "
        "has grown steadily through organic investment."
    ),
    (
        "Regional records show that {entity} was chartered in {year}. "
        "Since then, the entity has played a notable role in local commerce "
        "and contributes to the tax base and employment levels of the district."
    ),
    (
        "According to public filings, {entity} was incorporated in {year}. "
        "The organisation reports operations across several product lines, "
        "each contributing to its overall market presence."
    ),
]

_SECTORS = ["technology", "manufacturing", "financial", "retail", "energy", "healthcare"]
_EMPLOYEES_OPTS = [120, 450, 1200, 3400, 8700, 22000]


def _make_distractor(idx: int, rng: random.Random) -> dict:
    """Generate a unique non-numeric distractor paragraph."""
    template = _DISTRACTOR_TEMPLATES[idx % len(_DISTRACTOR_TEMPLATES)]
    entity_name = f"Distractor Entity {idx + 1}"
    year = rng.randint(1940, 2010)
    employees = rng.choice(_EMPLOYEES_OPTS)
    sector = rng.choice(_SECTORS)
    content = template.format(
        entity=entity_name,
        year=year,
        employees=employees,
        sector=sector,
    )
    return {
        "id": f"distractor_{idx}",
        "title": entity_name,
        "content": content,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark class
# ─────────────────────────────────────────────────────────────────────────────

class ComputationalBenchmark:
    """
    Generates 100 computation-focused questions for the tool-execution experiment.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.  Default 42.
    n_comparison : int
        Number of comparison questions.  Default 50.
    n_arithmetic : int
        Number of arithmetic (population sum) questions.  Default 50.
    n_distractors : int
        Distractor paragraphs per question (beyond the 2 gold ones).  Default 8.
    """

    def __init__(
        self,
        seed: int = 42,
        n_comparison: int = 50,
        n_arithmetic: int = 50,
        n_distractors: int = 8,
    ) -> None:
        self.seed = seed
        self.n_comparison = n_comparison
        self.n_arithmetic = n_arithmetic
        self.n_distractors = n_distractors

    def generate(self) -> list[dict]:
        """
        Return a list of example dicts ready for the EvaluationHarness.

        Each dict:
            id          : str
            question    : str
            answer      : str
            context     : list[{"id": str, "content": str, ...}]
            gold_ids    : list[str]
            task_type   : "comparison" | "arithmetic"
        """
        rng = random.Random(self.seed)
        examples: list[dict] = []

        # ── Comparison questions ─────────────────────────────────────────────
        comp_pool = list(_COMPARISON_POOL)
        rng.shuffle(comp_pool)
        for i, (ea, ra, eb, rb, answer) in enumerate(comp_pool[: self.n_comparison]):
            example = self._make_comparison(i, ea, ra, eb, rb, answer, rng)
            examples.append(example)

        # ── Arithmetic questions ─────────────────────────────────────────────
        arith_pool = list(_ARITHMETIC_POOL)
        rng.shuffle(arith_pool)
        for i, (ca, pa, cb, pb, total) in enumerate(arith_pool[: self.n_arithmetic]):
            example = self._make_arithmetic(i + self.n_comparison, ca, pa, cb, pb, total, rng)
            examples.append(example)

        return examples

    # ─────────────────────────────────────────────────────────
    # Question builders
    # ─────────────────────────────────────────────────────────

    def _make_comparison(
        self,
        idx: int,
        entity_a: str, revenue_a: int,
        entity_b: str, revenue_b: int,
        answer: str,
        rng: random.Random,
    ) -> dict:
        question = (
            f"Which company has higher annual revenue, {entity_a} or {entity_b}?"
        )
        gold_a = {
            "id": f"comp_{idx}_gold_a",
            "title": entity_a,
            "content": (
                f"{entity_a} reported annual revenue of ${revenue_a} million in its most "
                f"recent fiscal year. The company has maintained consistent growth across "
                f"its core business segments and continues to expand its market presence."
            ),
        }
        gold_b = {
            "id": f"comp_{idx}_gold_b",
            "title": entity_b,
            "content": (
                f"{entity_b} reported annual revenue of ${revenue_b} million last fiscal "
                f"year. The firm has invested heavily in operational efficiency, resulting "
                f"in improved margins and a stronger competitive position."
            ),
        }

        gold_ids = [gold_a["id"], gold_b["id"]]
        context = [gold_a, gold_b]
        for j in range(self.n_distractors):
            context.append(_make_distractor(idx * self.n_distractors + j, rng))
        rng.shuffle(context)

        return {
            "id": f"comp_{idx:04d}",
            "question": question,
            "answer": answer,
            "context": context,
            "gold_ids": gold_ids,
            "task_type": "comparison",
        }

    def _make_arithmetic(
        self,
        idx: int,
        city_a: str, pop_a: int,
        city_b: str, pop_b: int,
        total: int,
        rng: random.Random,
    ) -> dict:
        question = (
            f"What is the combined population of {city_a} and {city_b} "
            f"(in thousands of residents)?"
        )
        gold_a = {
            "id": f"arith_{idx}_gold_a",
            "title": city_a,
            "content": (
                f"{city_a} has a population of {pop_a},000 residents according to the most "
                f"recent municipal census. The city covers a geographic area of approximately "
                f"{rng.randint(30, 200)} square kilometres and serves as a regional hub."
            ),
        }
        gold_b = {
            "id": f"arith_{idx}_gold_b",
            "title": city_b,
            "content": (
                f"{city_b} is home to {pop_b},000 inhabitants as recorded in the latest "
                f"census data. The municipality has seen steady demographic growth over the "
                f"past decade, driven by economic development and infrastructure investment."
            ),
        }

        gold_ids = [gold_a["id"], gold_b["id"]]
        context = [gold_a, gold_b]
        for j in range(self.n_distractors):
            context.append(_make_distractor(idx * self.n_distractors + j, rng))
        rng.shuffle(context)

        return {
            "id": f"arith_{idx:04d}",
            "question": question,
            "answer": str(total),
            "context": context,
            "gold_ids": gold_ids,
            "task_type": "arithmetic",
        }
