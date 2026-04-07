"""
BRIGHT Benchmark — AEA Framework Evaluation.

BRIGHT (Reasoning-Intensive Retrieval Benchmark) tests retrieval where standard
semantic/keyword methods fail because queries and relevant documents share low
lexical AND low semantic overlap — correct retrieval requires REASONING to
connect query intent to document content.

Key hypothesis:
  If the structural stopping signal (stop when 2+ items from 2+ sources) is
  distribution-invariant, it should work EVEN on reasoning-intensive retrieval
  where standard retrievers struggle.  Failure here would limit the thesis to
  multi-hop QA only.

Data strategy:
  BRIGHT has massive corpora (50k–400k docs/domain), but each question maps to
  a small set of gold documents in a topic-specific folder.  We construct a
  per-question candidate pool of 20 documents:
    - Gold documents for the question (from the question's topic folder)
    - Surface-lexical distractors: docs from OTHER topic folders whose folder
      name shares keywords with the question (high surface overlap, wrong topic)
    - Random distractors: additional docs from completely unrelated topics

  This faithfully preserves the BRIGHT challenge:
    - Gold docs: correct topic folder, different vocabulary than the question
    - Distractors: wrong topic but superficially keyword-similar to query

  Domains used (4 science domains, sampled proportionally):
    biology, earth_science, economics, psychology

Usage
-----
    python experiments/run_bright.py

Results saved to experiments/results/bright.json.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

# ── Project root on sys.path ─────────────────────────────────────────────────
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
)
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy

# ── Constants ─────────────────────────────────────────────────────────────────
N_EXAMPLES = 200          # 200 reasoning-intensive questions (50 per domain)
N_PER_DOMAIN = 50
SEED = 42
CANDIDATE_POOL_SIZE = 20  # Documents per question (mirrors HotpotQA setting)
N_GOLD_MAX = 3            # Cap gold docs included (some queries have 5+)
N_DISTRACTORS = CANDIDATE_POOL_SIZE - N_GOLD_MAX  # surface + random distractors

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "bright.json"

# BRIGHT domains to use (4 science domains for diversity)
DOMAINS = ["biology", "earth_science", "economics", "psychology"]


# ─────────────────────────────────────────────────────────────────────────────
# BRIGHT data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_bright_domain(domain: str, rng: random.Random) -> tuple[list[dict], dict[str, str]]:
    """
    Load BRIGHT examples and documents for one domain.

    Returns
    -------
    examples : list[dict]
        Each BRIGHT example with query, gold_ids, reasoning, etc.
    doc_cache : dict[str, str]
        doc_id -> content mapping (sampled for tractability).
    """
    from datasets import load_dataset  # type: ignore

    print(f"  Loading {domain} examples …", end=" ", flush=True)
    examples_ds = load_dataset("xlangai/BRIGHT", "examples", split=domain)
    examples = list(examples_ds)
    print(f"{len(examples)} queries")

    # Collect all gold doc ids so we can load exactly those docs
    needed_ids: set[str] = set()
    for ex in examples:
        for gid in ex.get("gold_ids", [])[:N_GOLD_MAX]:
            needed_ids.add(gid)

    # Load documents as a streaming iterator to avoid OOM on huge corpora
    # We cache: (a) all needed gold docs, (b) a pool of "other" docs for distractors
    print(f"  Loading {domain} documents (streaming) …", end=" ", flush=True)
    docs_iter = load_dataset("xlangai/BRIGHT", "documents", split=domain, streaming=True)

    gold_docs: dict[str, str] = {}       # needed gold docs
    pool_docs: dict[str, str] = {}       # docs from different topic prefixes
    pool_limit = 2000                    # cap distractor pool to keep it tractable

    for doc in docs_iter:
        doc_id: str = doc["id"]
        content: str = doc.get("content", "") or ""

        if doc_id in needed_ids:
            gold_docs[doc_id] = content
        elif len(pool_docs) < pool_limit:
            pool_docs[doc_id] = content

        # Stop early once we have all gold docs and a reasonable pool
        if len(gold_docs) >= len(needed_ids) and len(pool_docs) >= pool_limit:
            break

    doc_cache = {**gold_docs, **pool_docs}
    print(f"{len(gold_docs)} gold + {len(pool_docs)} pool = {len(doc_cache)} total")
    return examples, doc_cache


def _topic_prefix(doc_id: str) -> str:
    """Return the topic folder of a BRIGHT doc id.  e.g. 'insects_attracted_to_light'."""
    return doc_id.split("/")[0] if "/" in doc_id else doc_id


def _surface_keywords(text: str) -> set[str]:
    """Extract content words (non-stopwords, len>3) from text for overlap check."""
    _STOP = frozenset({
        "the", "and", "for", "are", "but", "not", "this", "that", "have",
        "from", "with", "they", "what", "when", "where", "why", "how",
        "can", "was", "were", "been", "will", "would", "could", "should",
        "does", "did", "has", "had", "its", "into", "also", "than",
        "more", "some", "about", "which", "who", "there", "their",
    })
    tokens = re.findall(r"\b[a-z]{4,}\b", text.lower())
    return {t for t in tokens if t not in _STOP}


def _build_candidate_pool(
    example: dict,
    doc_cache: dict[str, str],
    all_doc_ids: list[str],
    rng: random.Random,
) -> list[dict]:
    """
    Build a 20-document candidate pool for one BRIGHT example.

    Strategy:
    1. Gold docs (up to N_GOLD_MAX) from the example's topic.
    2. Surface distractors: docs from OTHER topics that share keywords with query.
    3. Random distractors: completely unrelated docs.

    This preserves the BRIGHT challenge:
    - Gold: correct reasoning target, different vocabulary than query
    - Distractors: wrong topic but superficially query-similar (keyword traps)
    """
    query = example["query"]
    gold_ids_all = example.get("gold_ids", [])
    gold_ids = gold_ids_all[:N_GOLD_MAX]

    # Topic prefix of this question
    question_topic = _topic_prefix(gold_ids[0]) if gold_ids else ""

    # Step 1: Get gold documents
    gold_docs = []
    for gid in gold_ids:
        if gid in doc_cache:
            gold_docs.append({
                "id": gid,
                "title": gid.split("/")[-1].replace("_", " ").replace(".txt", ""),
                "content": doc_cache[gid],
                "is_gold": True,
            })

    n_gold = len(gold_docs)
    n_needed = CANDIDATE_POOL_SIZE - n_gold

    # Step 2: Build distractor pool from other topic documents
    query_keywords = _surface_keywords(query)

    other_docs = [
        did for did in all_doc_ids
        if _topic_prefix(did) != question_topic and did not in set(gold_ids_all)
        and did in doc_cache
    ]
    rng.shuffle(other_docs)

    # Score distractors by surface keyword overlap (higher overlap = better distractor)
    # These are "keyword traps" — they share surface vocab but are about wrong topic
    surface_scored = []
    for did in other_docs[:500]:  # Check up to 500 candidates
        content = doc_cache[did]
        doc_keywords = _surface_keywords(content[:300])
        overlap = len(query_keywords & doc_keywords)
        surface_scored.append((overlap, did))
    surface_scored.sort(key=lambda x: -x[0])

    # Take top surface-overlap distractors (keyword traps)
    n_surface = min(n_needed // 2, len(surface_scored))
    surface_distractors = [did for _, did in surface_scored[:n_surface]]

    # Step 3: Random distractors (completely unrelated)
    used_ids = set(gold_ids_all) | set(surface_distractors)
    random_pool = [
        did for did in other_docs if did not in used_ids
    ]
    n_random = n_needed - n_surface
    random_distractors = rng.sample(random_pool, min(n_random, len(random_pool)))

    # Combine all distractors
    distractor_ids = surface_distractors + random_distractors
    distractor_docs = []
    for did in distractor_ids:
        content = doc_cache.get(did, "")
        distractor_docs.append({
            "id": did,
            "title": did.split("/")[-1].replace("_", " ").replace(".txt", ""),
            "content": content,
            "is_gold": False,
        })

    # Combine and shuffle
    all_docs = gold_docs + distractor_docs
    rng.shuffle(all_docs)
    return all_docs


def build_bright_dataset(
    n_per_domain: int = N_PER_DOMAIN,
    seed: int = SEED,
) -> list[dict]:
    """
    Build the BRIGHT evaluation dataset.

    Loads real BRIGHT data from HuggingFace, constructs per-question
    20-document candidate pools with BRIGHT-style low lexical overlap.

    Returns
    -------
    list[dict]
        EvaluationHarness-compatible examples:
        - "id": str
        - "question": str
        - "answer": str
        - "context": list[{"id", "title", "content"}]
        - "gold_ids": list[str]
        - "domain": str
        - "bright_reasoning": str (from BRIGHT annotation)
    """
    rng = random.Random(seed)
    dataset: list[dict] = []

    for domain in DOMAINS:
        print(f"\n[{domain}]")
        examples, doc_cache = _load_bright_domain(domain, rng)

        # Sample n_per_domain questions
        rng.shuffle(examples)
        selected = examples[:n_per_domain]

        # All doc ids in this domain's cache (for distractor pool)
        all_doc_ids = list(doc_cache.keys())

        print(f"  Building candidate pools for {len(selected)} questions …", flush=True)
        domain_examples = []
        for ex in selected:
            gold_ids_all = ex.get("gold_ids", [])
            gold_ids = gold_ids_all[:N_GOLD_MAX]

            # Skip questions where we don't have any gold docs in cache
            if not any(gid in doc_cache for gid in gold_ids):
                continue

            context_docs = _build_candidate_pool(ex, doc_cache, all_doc_ids, rng)

            harness_ex = {
                "id": f"bright_{domain}_{ex['id']}",
                "question": ex["query"],
                "answer": ex.get("gold_answer", ""),
                "context": [
                    {
                        "id": doc["id"],
                        "title": doc["title"],
                        "content": doc["content"],
                    }
                    for doc in context_docs
                ],
                "gold_ids": [
                    doc["id"] for doc in context_docs if doc.get("is_gold", False)
                ],
                "domain": domain,
                "bright_reasoning": ex.get("reasoning", ""),
                "bright_question_id": ex["id"],
            }
            domain_examples.append(harness_ex)

        print(f"  {len(domain_examples)} examples with candidate pools")
        dataset.extend(domain_examples)

    rng.shuffle(dataset)
    print(f"\nTotal BRIGHT dataset: {len(dataset)} examples across {len(DOMAINS)} domains")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic BRIGHT-style fallback
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (question, gold_doc_title, gold_doc_content, answer, domain, category)
# Gold docs have LOW lexical overlap with the question (no shared content words)
# Distractors share surface keywords but discuss wrong concepts

_BRIGHT_ANALOGICAL = [
    # Biology / mechanisms
    (
        "Why do leaves change color in autumn?",
        "Chlorophyll degradation in senescing organs",
        "As chloroplast activity declines during leaf senescence, chlorophyll molecules break "
        "down faster than they are replenished. Carotenoids, which are masked during the growing "
        "season by the dominant green pigment, become visible as accessory pigments persist. "
        "Anthocyanins are newly synthesized from sugar accumulation under cool temperatures.",
        "Chlorophyll degrades, revealing carotenoids; anthocyanins are newly synthesized",
        "biology",
        "analogical",
    ),
    (
        "Why does salt melt ice?",
        "Colligative properties of solutions: freezing point depression",
        "When a solute dissolves in a liquid solvent, it disrupts the equilibrium between the "
        "liquid and solid phase by reducing the chemical potential of the liquid. This "
        "thermodynamic effect lowers the temperature at which the solid phase becomes stable, "
        "shifting the solid-liquid boundary to a lower temperature.",
        "Salt lowers the freezing point of water through colligative properties",
        "chemistry",
        "analogical",
    ),
    (
        "Why do we see lightning before hearing thunder?",
        "Propagation velocities of electromagnetic radiation versus acoustic waves",
        "Electromagnetic radiation propagates through the atmosphere at approximately 3×10⁸ m/s, "
        "while acoustic disturbances travel at roughly 343 m/s under standard conditions. "
        "The temporal offset between the two perceived events is directly proportional to "
        "the distance from the originating source.",
        "Light travels much faster than sound through air",
        "physics",
        "analogical",
    ),
    (
        "Why does bread rise when baked?",
        "Carbon dioxide production by Saccharomyces cerevisiae during fermentation",
        "Yeast ferment carbohydrates via glycolysis and alcoholic fermentation pathways, "
        "producing carbon dioxide as a metabolic byproduct. CO₂ gas becomes trapped within "
        "gluten networks formed by wheat proteins during kneading, causing expansion of the "
        "dough matrix when heat accelerates gas production.",
        "Yeast fermentation produces CO2 that expands within gluten networks",
        "biology",
        "analogical",
    ),
    (
        "Why is the sky blue?",
        "Rayleigh scattering of electromagnetic radiation in the atmosphere",
        "Molecular constituents of the atmosphere scatter incident solar radiation in a manner "
        "inversely proportional to the fourth power of the wavelength. Shorter wavelengths "
        "experience substantially greater scattering cross-sections, leading to preferential "
        "diffusion of high-frequency components across the observable sky dome.",
        "Rayleigh scattering preferentially scatters short wavelengths",
        "physics",
        "analogical",
    ),
    (
        "Why does copper turn green over time?",
        "Patina formation through atmospheric corrosion of copper alloys",
        "Prolonged exposure to atmospheric oxygen, water vapor, carbon dioxide, and sulfur "
        "compounds initiates electrochemical oxidation of copper surfaces. The resulting basic "
        "copper carbonate and copper sulfate compounds form a stable protective layer with "
        "distinct optical properties differing from the underlying metal substrate.",
        "Atmospheric oxidation forms copper carbonate/sulfate patina",
        "chemistry",
        "analogical",
    ),
    (
        "Why do we feel warmer after drinking alcohol?",
        "Peripheral vasodilation and thermoregulatory responses to ethanol",
        "Ethanol induces relaxation of vascular smooth muscle in cutaneous blood vessels, "
        "increasing blood flow to peripheral tissues. This redistribution of warm blood from "
        "core to periphery creates a sensation of warmth despite the net effect of accelerating "
        "heat dissipation and reducing core body temperature.",
        "Alcohol causes vasodilation, moving warm blood to skin surface",
        "medicine",
        "analogical",
    ),
    (
        "Why does metal feel colder than wood at the same temperature?",
        "Thermal conductivity differences and heat transfer rate perception",
        "The subjective experience of temperature upon contact with a material is determined by "
        "the rate at which heat transfers between the skin and the contacted surface. Materials "
        "with high thermal conductivity conduct heat away from the skin more rapidly, creating "
        "a sensory perception of coldness independent of the object's equilibrium temperature.",
        "Metal conducts heat away from skin faster due to higher thermal conductivity",
        "physics",
        "analogical",
    ),
    (
        "Why do stars twinkle but planets do not?",
        "Atmospheric turbulence and angular resolution of extended versus point sources",
        "Stellar objects subtend angles far below the resolving power of the human visual system, "
        "making them effectively point sources. Refractive index variations in atmospheric cells "
        "cause apparent positional shifts of point sources, perceived as scintillation. "
        "Extended objects average these perturbations across their angular diameter.",
        "Stars are point sources that scintillate; planets have resolvable disks",
        "astronomy",
        "analogical",
    ),
    (
        "Why do we age?",
        "Telomere attrition and accumulation of somatic mutations over cellular generations",
        "With each mitotic division, replication machinery fails to copy the terminal segments of "
        "linear chromosomes, leading to progressive shortening of protective telomeric sequences. "
        "Cells approaching critically short telomeres activate checkpoints leading to senescence "
        "or apoptosis. Simultaneously, imperfect DNA repair allows accumulation of mutational burden.",
        "Telomere shortening and DNA damage accumulation cause cellular senescence",
        "biology",
        "analogical",
    ),
    # Earth science / causal
    (
        "Why do earthquakes happen along the Pacific Rim?",
        "Convergent and transform plate boundary interactions at subduction zones",
        "The circum-Pacific region is characterized by convergent margins where oceanic lithosphere "
        "descends beneath overriding plates. Elastic strain accumulates along the megathrust "
        "interface between subducting and overriding plates, periodically releasing as seismic "
        "events when frictional resistance is overcome.",
        "Subduction zones accumulate elastic strain that releases as earthquakes",
        "earth_science",
        "causal",
    ),
    (
        "What causes the seasons on Earth?",
        "Axial inclination of Earth's rotational axis relative to orbital plane",
        "Earth's rotational axis maintains a fixed orientation relative to distant stars, creating "
        "an angular offset between the equatorial and orbital planes. This inclination causes "
        "the solar elevation angle and day length to vary systematically with orbital position, "
        "determining the intensity of solar irradiance received per unit area at each latitude.",
        "Earth's 23.5° axial tilt causes varying solar angle across orbital positions",
        "earth_science",
        "causal",
    ),
    (
        "Why is ocean water salty?",
        "Hydrological cycling and chemical weathering of continental rocks",
        "Precipitation interacts with exposed rock surfaces, dissolving ionic species through "
        "hydrolysis and carbonic acid reactions. Fluvial transport carries dissolved ions to "
        "marine basins where evaporation concentrates solutes while water molecules return "
        "to the atmospheric reservoir via evapotranspiration.",
        "Rain weathers rocks, rivers carry ions to ocean; evaporation concentrates salts",
        "earth_science",
        "causal",
    ),
    (
        "Why do hurricanes spin counterclockwise in the Northern Hemisphere?",
        "Coriolis acceleration from Earth's rotation on large-scale atmospheric flows",
        "In a rotating reference frame, free-moving bodies experience an apparent deflection "
        "perpendicular to their velocity. In the Northern Hemisphere, this deflection is to the "
        "right, causing air flowing toward a low-pressure center to veer clockwise around the "
        "center, producing counterclockwise cyclonic rotation when viewed from above.",
        "Coriolis effect deflects winds right in Northern Hemisphere, creating cyclonic spin",
        "earth_science",
        "causal",
    ),
    (
        "Why does the moon have craters?",
        "Impact flux history and absence of geological resurfacing processes",
        "Solid planetary bodies receive continuous bombardment from interplanetary debris "
        "throughout their history. Bodies lacking active geological processes such as volcanism, "
        "tectonism, or fluvial erosion preserve impact structures over geological timescales, "
        "while atmospherically unshielded surfaces accumulate high crater densities.",
        "Moon lacks erosion/tectonics, so craters from meteorite impacts are preserved",
        "astronomy",
        "causal",
    ),
    # Economics / inferential
    (
        "Why does inflation reduce the value of savings?",
        "Purchasing power and real interest rate theory",
        "The real return on a financial instrument equals the nominal yield adjusted for the "
        "rate of price level change. When general price indices rise faster than the interest "
        "rate credited to a deposit account, the quantity of goods and services that the "
        "accumulated balance can acquire diminishes over time.",
        "Inflation erodes purchasing power when price rises exceed nominal returns",
        "economics",
        "inferential",
    ),
    (
        "Why do minimum wage increases sometimes cause unemployment?",
        "Labor demand elasticity and marginal productivity theory of wages",
        "Profit-maximizing firms employ workers up to the point where the marginal revenue "
        "product equals the wage rate. When legally mandated wages exceed the market-clearing "
        "rate, the quantity of labor demanded falls as firms substitute capital or reduce output "
        "where marginal productivity cannot justify the higher wage cost.",
        "Wages above marginal product reduce labor demand; firms substitute capital",
        "economics",
        "inferential",
    ),
    (
        "Why does a country's currency weaken when it prints more money?",
        "Quantity theory of money and purchasing power parity",
        "The price level in an economy is proportional to the money supply relative to real "
        "output under velocity stability assumptions. An increased money supply relative to "
        "goods and services leads to higher domestic prices, reducing the international "
        "purchasing power of the currency and shifting exchange rates.",
        "Money supply growth relative to output causes inflation, weakening exchange rate",
        "economics",
        "inferential",
    ),
    (
        "Why do businesses form monopolies?",
        "Economies of scale and barriers to market entry",
        "Industries with high fixed costs and declining average total costs exhibit natural "
        "monopoly tendencies as incumbents spread overhead across larger output volumes. "
        "First-mover advantages in capital-intensive sectors create cost asymmetries that "
        "deter entry by potential competitors facing higher unit costs.",
        "High fixed costs create scale advantages that deter entry",
        "economics",
        "inferential",
    ),
    (
        "Why does foreign aid sometimes fail to develop poor countries?",
        "Institutional quality, governance, and aid-induced fiscal displacement",
        "Aid inflows can displace domestic resource mobilization efforts when governments "
        "reduce taxation and spending in anticipation of external transfers. Weak institutions "
        "allow diversion of external resources to rent-seeking rather than productive investment, "
        "while aid dependency may undermine accountability mechanisms between citizens and states.",
        "Aid reduces tax effort and enables rent-seeking in weak institutions",
        "economics",
        "inferential",
    ),
    # Psychology / causal
    (
        "Why do people believe conspiracy theories?",
        "Epistemic insecurity and proportionality bias in causal attribution",
        "Human pattern recognition systems generate explanatory narratives proportional to "
        "the perceived significance of an event. For events of high subjective magnitude, "
        "simple mechanical or random causes feel cognitively unsatisfying, driving preference "
        "for complex intentional explanations. Epistemic anxiety amplifies proportionality bias.",
        "Proportionality bias: big events require big causes; uncertainty amplifies it",
        "psychology",
        "causal",
    ),
    (
        "Why do people procrastinate?",
        "Hyperbolic temporal discounting and delay aversion in reward processing",
        "Intertemporal choice is characterized by disproportionate devaluation of delayed "
        "rewards relative to normative exponential discounting models. Immediate negative "
        "affect associated with aversive tasks activates avoidance behavior, while distant "
        "deadlines fail to activate sufficient urgency signals in limbic motivational circuitry.",
        "Steep discounting of future rewards and affect regulation drive task avoidance",
        "psychology",
        "causal",
    ),
    (
        "Why do people conform to group norms even when they disagree?",
        "Informational and normative social influence mechanisms",
        "Individuals in ambiguous situations use others' behavior as diagnostic information "
        "about the correct response, producing informational conformity. Under normative "
        "influence, individuals comply with majority behavior to achieve social approval and "
        "avoid sanctions, even when privately maintaining divergent beliefs.",
        "Social approval seeking (normative) and uncertainty reduction (informational) drive conformity",
        "psychology",
        "causal",
    ),
    (
        "Why do people remember negative events better than positive ones?",
        "Amygdala modulation of hippocampal memory consolidation",
        "Emotionally arousing events trigger noradrenergic activation of the basolateral "
        "amygdala, which enhances synaptic plasticity in hippocampal circuits during memory "
        "consolidation. Negative events typically generate higher arousal than comparable "
        "positive events, producing stronger memory traces through this neuromodulatory pathway.",
        "Negative events cause higher arousal, enhancing amygdala-hippocampal consolidation",
        "psychology",
        "causal",
    ),
    (
        "Why do children learn languages faster than adults?",
        "Critical period hypothesis and synaptic pruning in neural development",
        "The brain maintains heightened synaptic plasticity during developmental sensitive "
        "periods, allowing rapid reconfiguration of phonological processing circuits in "
        "response to linguistic input. After puberty, myelination and synaptic stabilization "
        "reduce plasticity, requiring more effortful explicit learning strategies.",
        "Neural plasticity peaks in childhood; myelination reduces it in adults",
        "psychology",
        "causal",
    ),
    # Additional high-reasoning-gap questions
    (
        "Why does exercise improve mood?",
        "Endorphin release and BDNF upregulation in aerobic exercise",
        "Physical exertion activates hypothalamic-pituitary pathways releasing β-endorphins "
        "that bind opioid receptors. Sustained aerobic activity upregulates brain-derived "
        "neurotrophic factor expression, promoting hippocampal neurogenesis and synaptic "
        "remodeling in circuits regulating hedonic tone.",
        "Exercise releases endorphins and upregulates BDNF for mood regulation",
        "neuroscience",
        "causal",
    ),
    (
        "Why do vaccines work?",
        "Adaptive immune memory and clonal expansion of antigen-specific lymphocytes",
        "Primary exposure to an antigen activates naive lymphocytes bearing complementary "
        "antigen receptors, inducing proliferation and differentiation into effector and "
        "long-lived memory cells. Subsequent antigen encounters trigger rapid clonal expansion "
        "of memory populations, achieving protective antibody titers before clinical disease.",
        "Vaccines create immunological memory for rapid secondary immune response",
        "immunology",
        "causal",
    ),
    (
        "Why do some people become addicted to drugs while others do not?",
        "Dopaminergic reward sensitivity and prefrontal inhibitory control variation",
        "Individual differences in mesolimbic dopamine system sensitivity and prefrontal "
        "cortical regulation of reward-seeking behavior determine vulnerability to compulsive "
        "use patterns. Genetic polymorphisms in dopamine receptor density and transporter "
        "function interact with environmental stress to modulate addiction trajectories.",
        "Dopamine system variation and PFC inhibitory control determine addiction vulnerability",
        "neuroscience",
        "causal",
    ),
    (
        "Why is it harder to lose weight than to gain it?",
        "Adaptive thermogenesis and neuroendocrine compensation during caloric deficit",
        "Reduced caloric intake triggers metabolic adaptation through multiple homeostatic "
        "mechanisms: resting energy expenditure declines beyond what is predicted by lean mass "
        "loss, leptin secretion falls causing increased appetite, and ghrelin levels rise "
        "chronically, creating persistent hunger signals that resist the negative energy balance.",
        "Leptin drop and adaptive thermogenesis compensate against caloric deficit",
        "physiology",
        "causal",
    ),
    (
        "Why do antibiotics not work against viruses?",
        "Mechanistic target specificity of antibacterial agents",
        "Antibacterial agents target structures or metabolic processes specific to prokaryotic "
        "organisms: cell wall peptidoglycan synthesis, 70S ribosomal subunits, or prokaryotic "
        "DNA gyrase. Viruses lack these structures entirely, using host cell ribosomes and "
        "replication machinery, rendering antibacterial agents mechanistically ineffective.",
        "Antibiotics target bacterial-specific structures absent in virus replication",
        "microbiology",
        "analogical",
    ),
    (
        "Why do we need sleep?",
        "Glymphatic waste clearance and synaptic homeostasis during sleep",
        "The glymphatic system, comprising perivascular channels around cerebral blood vessels, "
        "shows dramatically increased convective flow during sleep, facilitating clearance of "
        "metabolic waste products including amyloid-beta. Separately, sleep-dependent synaptic "
        "downscaling renormalizes potentiated synapses, restoring dynamic range for plasticity.",
        "Sleep enables glymphatic waste clearance and synaptic homeostasis",
        "neuroscience",
        "causal",
    ),
    (
        "Why do rivers form meanders?",
        "Helical flow dynamics and lateral erosion in alluvial channels",
        "Secondary circulation develops in curved channel reaches as centrifugal forces create "
        "a cross-stream pressure gradient, driving flow from the outer bank toward the inner "
        "bank along the bed. This helical flow transports sediment preferentially from outer "
        "cut banks to inner point bars, amplifying initial curvature through positive feedback.",
        "Helical secondary flows erode outer banks and deposit at inner bends, amplifying curves",
        "earth_science",
        "causal",
    ),
    (
        "Why does the stock market fluctuate daily?",
        "Information aggregation and expectation revisions in efficient markets",
        "Competitive trading among participants with heterogeneous beliefs and information "
        "causes asset prices to continuously incorporate new information about future cash flows "
        "and discount rates. Even small revisions to earnings expectations or macroeconomic "
        "indicators propagate through derivative positions and algorithmic strategies.",
        "New information and expectation revisions aggregate through competitive trading",
        "economics",
        "inferential",
    ),
    (
        "Why are diamonds so hard?",
        "Covalent bonding network geometry in tetrahedral carbon crystal structure",
        "Carbon atoms in the diamond cubic structure form sp³ hybrid orbitals, creating four "
        "equivalent directional covalent bonds arranged tetrahedrally. The three-dimensional "
        "network of strong, directional bonds creates extremely high resistance to both "
        "deformation and cleavage across all crystallographic planes.",
        "3D tetrahedral covalent network resists deformation in all directions",
        "chemistry",
        "analogical",
    ),
    (
        "Why do birds migrate?",
        "Photoperiodism and circannual rhythms in Neotropical migrants",
        "Declining photoperiod in boreal latitudes triggers hormonal cascades mediated by "
        "melatonin and gonadotropins that initiate pre-migratory fattening and orientation "
        "behavior. Internal circannual oscillators entrained by photoperiod drive seasonal "
        "behavioral transitions independent of immediate environmental conditions.",
        "Photoperiod changes trigger hormonal cascades initiating migratory behavior",
        "biology",
        "causal",
    ),
    (
        "Why is dark chocolate considered healthier than milk chocolate?",
        "Flavanol content and endothelial function in polyphenol-rich foods",
        "Cacao-derived flavanols activate endothelial nitric oxide synthase, promoting "
        "vasodilation and improving flow-mediated dilation. Processing steps that increase "
        "milk content dilute cacao mass concentration, reducing the absolute quantity of "
        "bioactive flavanols per gram while increasing saturated fat.",
        "Higher cacao flavanols activate nitric oxide synthase for vascular benefits",
        "nutrition",
        "analogical",
    ),
    (
        "Why do humans have fingerprints?",
        "Mechanoreceptor enhancement through epidermal ridge amplification",
        "Dermal ridges create mechanical coupling between external tactile stimuli and "
        "Meissner and Merkel corpuscles positioned at dermal papillae. Ridge geometry "
        "amplifies transverse skin deformations, enhancing vibrotactile discrimination "
        "and grip performance through increased friction coefficients on curved surfaces.",
        "Ridges amplify mechanoreceptor signals and increase grip friction",
        "biology",
        "analogical",
    ),
    (
        "Why do we yawn when we see someone else yawn?",
        "Mirror neuron system and social mimicry in human motor behavior",
        "Observation of motor acts performed by conspecifics activates the motor cortex "
        "through an action observation-execution matching system. Yawn contagion represents "
        "a highly automatized form of social mimicry, with contagion rates correlating with "
        "trait empathy measures and social closeness.",
        "Mirror neuron action-observation matching drives social motor mimicry",
        "neuroscience",
        "analogical",
    ),
    (
        "Why does sugar make us feel happy?",
        "Dopaminergic reward circuitry activation by glucose and sucrose",
        "Glucose and sucrose activate sweet taste receptors that signal via the nucleus "
        "tractus solitarius to mesolimbic structures, triggering dopamine release in nucleus "
        "accumbens. Post-ingestive signals from intestinal sweet receptors provide a "
        "secondary reward pathway independent of oral taste sensation.",
        "Sugar activates taste receptors and mesolimbic dopamine release",
        "neuroscience",
        "causal",
    ),
    (
        "Why do magnets attract iron?",
        "Exchange interaction and magnetic domain alignment in ferromagnetic materials",
        "Ferromagnetic materials contain atomic-scale magnetic moments arising from unpaired "
        "electron spins that quantum mechanical exchange interactions align in parallel within "
        "local domains. External fields cause domain wall movement and magnetization rotation "
        "that produces a net field, attracting unmagnetized ferromagnetic objects.",
        "Quantum exchange interactions align electron spins; domains align in external fields",
        "physics",
        "analogical",
    ),
    (
        "Why is the Amazon important for global climate?",
        "Transpirational water flux and regional energy balance in tropical forests",
        "Tropical forest transpiration recycles large quantities of water to the atmosphere, "
        "sustaining a continental-scale moisture conveyor that determines precipitation "
        "patterns far downwind. Forest canopy also stores substantial carbon in biomass, "
        "acting as a net carbon sink that modulates atmospheric CO₂ accumulation rates.",
        "Forest transpiration drives moisture recycling and carbon sequestration",
        "ecology",
        "inferential",
    ),
    (
        "Why do humans get goosebumps when cold or scared?",
        "Arrector pili muscle contraction and vestigial thermoregulatory response",
        "Catecholamine release during sympathetic nervous system activation stimulates "
        "smooth muscle cells attached to hair follicles. The resulting piloerection reflex "
        "is homologous to fur erection in other mammals, which creates an insulating air "
        "layer; the response persists as a phylogenetic vestige in humans.",
        "Sympathetic catecholamines activate arrector pili; vestigial mammalian thermoregulation",
        "biology",
        "analogical",
    ),
    (
        "Why do we get headaches?",
        "Trigeminovascular sensitization and cortical spreading depression",
        "Migraine involves spreading waves of neuronal depolarization followed by prolonged "
        "suppression across cortical tissue. This activity activates trigeminal afferents "
        "surrounding meningeal blood vessels, releasing neuropeptides that promote dilation "
        "and neurogenic inflammation, transmitting pain signals to higher cortical centers.",
        "Cortical spreading depression activates trigeminal pain pathways",
        "neuroscience",
        "causal",
    ),
    (
        "Why does the economy go through boom and bust cycles?",
        "Credit creation dynamics and Minsky instability hypothesis",
        "During expansions, rising asset valuations encourage leveraged investment as lenders "
        "reduce margins and borrowers extend position sizes. As speculative and Ponzi financing "
        "units accumulate, any interruption to income or credit availability forces asset sales "
        "that depress prices, triggering a cascade of deleveraging.",
        "Credit leverage amplifies expansions until debt service failures trigger deleveraging",
        "economics",
        "inferential",
    ),
    (
        "Why do some materials conduct electricity while others do not?",
        "Band gap theory and electron mobility in crystalline solids",
        "Quantum mechanical treatment of electron states in periodic crystal lattices produces "
        "allowed energy bands separated by forbidden gaps. In conductors, partially filled "
        "conduction bands allow electrons to respond to electric fields; insulators have "
        "fully occupied valence bands separated from empty conduction bands by large gaps.",
        "Band structure determines electron mobility: conductors have partially filled bands",
        "physics",
        "analogical",
    ),
    (
        "Why does the body produce fever when sick?",
        "Cytokine-mediated hypothalamic thermoregulatory set-point elevation",
        "Pyrogens including interleukin-1, interleukin-6, and tumor necrosis factor released "
        "by activated immune cells reach the organum vasculosum of the lamina terminalis, "
        "inducing prostaglandin E₂ synthesis that shifts the hypothalamic temperature "
        "set-point upward, activating heat conservation and generation mechanisms.",
        "Cytokines induce PGE2 which raises hypothalamic temperature set-point",
        "immunology",
        "causal",
    ),
    (
        "Why is biodiversity important for ecosystems?",
        "Functional redundancy, complementarity effects, and stability-diversity relationship",
        "Species with overlapping functional roles provide redundancy against local extinctions. "
        "Niche complementarity allows more complete resource use across spatial and temporal "
        "gradients. Mathematically, diversity reduces variance in community functioning "
        "through portfolio effects when species respond differently to perturbations.",
        "Functional redundancy and complementarity provide stability through portfolio effects",
        "ecology",
        "inferential",
    ),
    (
        "Why do people get bored?",
        "Attentional regulation and tonic alertness deficits in understimulating environments",
        "Boredom arises when available stimulation fails to match attentional engagement "
        "requirements: too little novelty for automatic processing and insufficient cognitive "
        "demand for effortful attention. This creates a dissociation between time perception "
        "and task engagement, generating aversive arousal that motivates novelty-seeking.",
        "Mismatch between stimulation supply and attentional demand creates aversive state",
        "psychology",
        "causal",
    ),
    (
        "Why do fish die when taken out of water?",
        "Gill function and aqueous oxygen extraction versus atmospheric respiration",
        "Teleost gills function by countercurrent exchange of dissolved oxygen across thin "
        "epithelial lamellae perfused by blood. Atmospheric air collapses the gill lamellae "
        "through surface tension, abolishing the large exchange surface area while providing "
        "oxygen in a form the gill epithelium cannot efficiently absorb.",
        "Gill lamellae collapse in air, preventing oxygen extraction despite ambient O2",
        "biology",
        "analogical",
    ),
    (
        "Why does climate change cause more extreme weather?",
        "Thermodynamic intensification of the hydrological cycle under forcing",
        "Elevated atmospheric temperatures increase boundary layer water vapor capacity "
        "following the Clausius-Clapeyron relation, providing more latent energy available "
        "for convective systems. The increased moisture also steepens horizontal temperature "
        "gradients under some forcing scenarios, energizing extratropical storm tracks.",
        "Higher SST increases atmospheric moisture following Clausius-Clapeyron relation",
        "earth_science",
        "inferential",
    ),
    (
        "Why do economies struggle after financial crises?",
        "Balance sheet recession and debt deflation dynamics",
        "When private sector balance sheets are impaired, firms and households direct "
        "cash flow toward debt repayment rather than consumption or investment, regardless "
        "of interest rates. This aggregate demand shortfall depresses income, which worsens "
        "balance sheets further in a self-reinforcing deflationary spiral.",
        "Balance sheet repair suppresses aggregate demand in self-reinforcing deflation",
        "economics",
        "inferential",
    ),
    (
        "Why do people have different personality types?",
        "Five-factor model heritability and gene-environment correlation",
        "Twin studies indicate that approximately 40-60% of variance in broad personality "
        "dimensions is attributable to additive genetic effects. Environmental influences "
        "operate largely through non-shared mechanisms, with gene-environment correlation "
        "meaning individuals differentially select and create environments matching their "
        "heritable dispositions.",
        "Personality variance is 40-60% heritable via additive genetic effects",
        "psychology",
        "causal",
    ),
    (
        "Why is space a vacuum?",
        "Gravitational confinement limits of planetary atmospheres in interplanetary medium",
        "Gases in planetary atmospheres are gravitationally bound when thermal velocities of "
        "molecules remain below escape velocity at the exobase altitude. In interplanetary "
        "space, no massive body provides sufficient gravitational potential to confine gas, "
        "allowing molecular diffusion to maintain extremely low number densities.",
        "No gravity to confine gas; molecules diffuse to extremely low densities",
        "physics",
        "analogical",
    ),
    (
        "Why do some people suffer from seasonal affective disorder?",
        "Circadian rhythm disruption and serotonergic dysregulation under reduced photoperiod",
        "Reduced light exposure in winter months diminishes retinal stimulation of the "
        "suprachiasmatic nucleus, disrupting circadian entrainment and serotonin turnover. "
        "The SERT gene polymorphism that reduces serotonin reuptake efficiency may amplify "
        "vulnerability to light-induced changes in mood regulation.",
        "Reduced photoperiod disrupts SCN entrainment and serotonin regulation",
        "neuroscience",
        "causal",
    ),
    (
        "Why does glass shatter but rubber bend?",
        "Brittle versus viscoelastic fracture mechanics and polymer chain mobility",
        "Amorphous silicate glasses lack dislocation mobility, forcing crack propagation "
        "when stress exceeds the theoretical fracture strength. Elastomers consist of "
        "cross-linked polymer chains with high segmental mobility, allowing reversible "
        "conformational changes that dissipate strain energy without crack nucleation.",
        "Glass has no ductile deformation mode; rubber chains absorb energy conformationally",
        "materials",
        "analogical",
    ),
    (
        "Why do some animals live longer than others?",
        "Oxidative stress theory and rate of living hypothesis in comparative biology",
        "Metabolic rate determines the flux of reactive oxygen species produced as byproducts "
        "of mitochondrial electron transport. Species with higher mass-specific metabolic rates "
        "accumulate oxidative macromolecular damage faster, while those with more robust "
        "antioxidant and DNA repair systems extend healthy lifespan.",
        "ROS production rate and antioxidant/repair capacity determine oxidative damage accumulation",
        "biology",
        "causal",
    ),
    (
        "Why does depression make it hard to enjoy previously enjoyable activities?",
        "Anhedonia and reduced dopaminergic reward prediction in the mesolimbic system",
        "In major depressive disorder, blunted activation of the ventral striatum and nucleus "
        "accumbens reduces the subjective valence assigned to anticipated rewards. Diminished "
        "dopamine signaling in reward circuits impairs both the wanting and liking components "
        "of incentive salience, flattening the motivational response to previously valued stimuli.",
        "Blunted striatal dopamine signaling reduces incentive salience for rewards",
        "psychology",
        "causal",
    ),
]


def build_synthetic_bright(n: int = N_EXAMPLES, seed: int = SEED) -> list[dict]:
    """
    Build synthetic BRIGHT-style dataset.

    Uses pre-defined question/answer pairs where:
    - Questions use everyday language
    - Gold docs use scientific/technical vocabulary with low lexical overlap
    - Distractors share surface keywords with the question but discuss wrong topics
    """
    rng = random.Random(seed)

    entries = _BRIGHT_ANALOGICAL.copy()
    rng.shuffle(entries)

    if len(entries) < n:
        # Repeat with variations if needed
        repeated = (entries * ((n // len(entries)) + 2))[:n]
        rng.shuffle(repeated)
        entries = repeated
    else:
        entries = entries[:n]

    # Build a distractor pool from all gold doc contents (shuffled)
    all_gold_docs = [(e[1], e[2], e[5]) for e in entries]  # (title, content, category)

    dataset = []
    for idx, entry in enumerate(entries):
        question, gold_title, gold_content, answer, domain, category = entry

        # Gold document
        gold_doc = {
            "id": f"bright_syn_{idx:03d}_gold",
            "title": gold_title,
            "content": gold_content,
        }

        # Surface distractors: docs from the same domain pool but wrong topic
        # Use other gold doc contents that share surface keywords with question
        q_kw = _surface_keywords(question)
        scored_distractors = []
        for jdx, (dtitle, dcontent, _) in enumerate(all_gold_docs):
            if jdx == idx:
                continue
            d_kw = _surface_keywords(dcontent[:200])
            overlap = len(q_kw & d_kw)
            scored_distractors.append((overlap, jdx, dtitle, dcontent))
        scored_distractors.sort(key=lambda x: -x[0])

        # Take top 5 surface distractors (keyword traps)
        context_docs = [gold_doc]
        used_jdx = set()
        for _, jdx, dtitle, dcontent in scored_distractors[:5]:
            context_docs.append({
                "id": f"bright_syn_{jdx:03d}_dist",
                "title": dtitle,
                "content": dcontent,
            })
            used_jdx.add(jdx)

        # Fill remaining with random docs (totally unrelated)
        random_pool = [
            (jdx, dtitle, dcontent) for jdx, (dtitle, dcontent, _) in enumerate(all_gold_docs)
            if jdx != idx and jdx not in used_jdx
        ]
        rng.shuffle(random_pool)
        for jdx, dtitle, dcontent in random_pool[:CANDIDATE_POOL_SIZE - len(context_docs)]:
            context_docs.append({
                "id": f"bright_syn_{jdx:03d}_dist",
                "title": dtitle,
                "content": dcontent,
            })

        rng.shuffle(context_docs)

        dataset.append({
            "id": f"bright_{domain}_{idx:03d}",
            "question": question,
            "answer": answer,
            "context": context_docs,
            "gold_ids": [gold_doc["id"]],
            "domain": domain,
            "category": category,
            "is_synthetic": True,
        })

    print(f"Built synthetic BRIGHT-style: {len(dataset)} examples "
          f"({sum(1 for e in dataset if e['category']=='analogical')} analogical, "
          f"{sum(1 for e in dataset if e['category']=='causal')} causal, "
          f"{sum(1 for e in dataset if e['category']=='inferential')} inferential)")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Policy and address-space factories
# ─────────────────────────────────────────────────────────────────────────────

def make_policies() -> list:
    """
    Create the four core policies for BRIGHT evaluation.

    Same policies as HotpotQA and 2WikiMultiHopQA experiments:
      pi_semantic, pi_lexical, pi_ensemble, pi_aea_heuristic
    """
    return [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
    ]


def make_address_spaces() -> dict:
    """Create fresh address-space instances."""
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired t-test; returns (t_stat, p_value)."""
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan")
    a_arr = np.array(a[:n])
    b_arr = np.array(b[:n])
    result = stats.ttest_rel(a_arr, b_arr)
    return float(result.statistic), float(result.pvalue)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size between two paired samples."""
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    diffs = np.array(a[:n]) - np.array(b[:n])
    return float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))


def extract_metric(per_example: list[dict], key: str) -> list[float]:
    """Extract a metric from per_example list."""
    return [ex[key] for ex in per_example if key in ex and isinstance(ex[key], (int, float))]


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_bright(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Run all policies on BRIGHT reasoning-intensive retrieval benchmark.

    Tests: Does the structural stopping heuristic generalize to a genuinely
    different benchmark family?

    Returns
    -------
    dict
        All policy results with aggregated metrics and statistical tests.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("BRIGHT Benchmark — AEA Evaluation")
    print("Reasoning-Intensive Retrieval (Genuine Cross-Benchmark Validation)")
    print("=" * 70)
    print()

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("Step 1: Loading BRIGHT dataset …")
    dataset: list[dict] = []
    is_synthetic = False

    try:
        from datasets import load_dataset  # type: ignore  # noqa: F401
        print("  HuggingFace datasets available. Attempting real BRIGHT data …")
        dataset = build_bright_dataset(n_per_domain=n_examples // len(DOMAINS), seed=seed)
        if len(dataset) < 50:
            print(f"  Only {len(dataset)} real examples loaded (< 50). Falling back to synthetic.")
            raise RuntimeError("Too few real examples")
    except Exception as exc:
        print(f"  Real BRIGHT loading failed: {exc}")
        print("  Constructing synthetic BRIGHT-style dataset …")
        dataset = build_synthetic_bright(n=n_examples, seed=seed)
        is_synthetic = True

    n_actual = len(dataset)
    print(f"\nDataset ready: {n_actual} examples")
    print(f"  Synthetic: {is_synthetic}")
    if not is_synthetic:
        domain_counts = {}
        for ex in dataset:
            d = ex.get("domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1
        for d, c in sorted(domain_counts.items()):
            print(f"  {d}: {c}")

    # ── Step 2: Retrieval evaluation ──────────────────────────────────────────
    print(f"\nStep 2: Retrieval evaluation on {n_actual} examples …")
    policies = make_policies()
    all_results: dict = {}

    for policy in policies:
        pname = policy.name()
        print(f"\n{'─' * 60}")
        print(f"Policy: {pname}")
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

        # Inject domain into per_example for breakdowns
        id_to_domain = {ex["id"]: ex.get("domain", "unknown") for ex in dataset}
        id_to_category = {ex["id"]: ex.get("category", "unknown") for ex in dataset}
        for ex in result["per_example"]:
            ex["domain"] = id_to_domain.get(ex.get("id", ""), "unknown")
            ex["category"] = id_to_category.get(ex.get("id", ""), "unknown")

        agg = result["aggregated"]
        print(f"  support_recall:    {agg['support_recall']:.4f}")
        print(f"  support_precision: {agg['support_precision']:.4f}")
        print(f"  utility@budget:    {agg['utility_at_budget']:.4f}")
        print(f"  avg_operations:    {agg['operations_used']:.2f}")
        print(f"  n_errors:          {result['n_errors']}")
        print(f"  runtime:           {elapsed:.1f}s")

        all_results[pname] = result

    # ── Step 3: Statistical analysis ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Step 3: Statistical Analysis")
    print(f"{'=' * 70}")

    stats_results = _compute_statistics(all_results)
    _print_results_table(all_results, n_actual)
    _print_statistics(stats_results)
    _print_key_finding(all_results, stats_results)

    # ── Save results ──────────────────────────────────────────────────────────
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # Slim per_example (drop trace)
        slim_results = {}
        for pname, res in all_results.items():
            slim_per = [
                {k: v for k, v in ex.items() if k != "trace"}
                for ex in res.get("per_example", [])
            ]
            slim_results[pname] = {**res, "per_example": slim_per}

        output = {
            "benchmark": "BRIGHT",
            "benchmark_description": (
                "Reasoning-Intensive Retrieval Benchmark — "
                "queries and gold docs have low lexical AND semantic overlap"
            ),
            "n_examples": n_actual,
            "is_synthetic": is_synthetic,
            "domains": DOMAINS,
            "seed": seed,
            "results": slim_results,
            "statistics": stats_results,
        }
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        print(f"\nResults saved to: {results_path}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_statistics(all_results: dict) -> dict:
    """Compute paired t-tests and effect sizes comparing heuristic vs baselines."""
    stat_results = {}

    if "pi_aea_heuristic" not in all_results:
        return stat_results

    heuristic_sr = extract_metric(
        all_results["pi_aea_heuristic"]["per_example"], "support_recall"
    )
    heuristic_ub = extract_metric(
        all_results["pi_aea_heuristic"]["per_example"], "utility_at_budget"
    )

    for pname, result in all_results.items():
        if pname == "pi_aea_heuristic":
            continue

        baseline_sr = extract_metric(result["per_example"], "support_recall")
        baseline_ub = extract_metric(result["per_example"], "utility_at_budget")

        t_sr, p_sr = paired_ttest(heuristic_sr, baseline_sr)
        d_sr = cohens_d(heuristic_sr, baseline_sr)

        t_ub, p_ub = paired_ttest(heuristic_ub, baseline_ub)
        d_ub = cohens_d(heuristic_ub, baseline_ub)

        stat_results[f"heuristic_vs_{pname}"] = {
            "support_recall": {
                "heuristic_mean": float(np.mean(heuristic_sr)),
                "baseline_mean": float(np.mean(baseline_sr)),
                "delta": float(np.mean(heuristic_sr)) - float(np.mean(baseline_sr)),
                "t_stat": t_sr,
                "p_value": p_sr,
                "cohens_d": d_sr,
                "n": min(len(heuristic_sr), len(baseline_sr)),
            },
            "utility_at_budget": {
                "heuristic_mean": float(np.mean(heuristic_ub)),
                "baseline_mean": float(np.mean(baseline_ub)),
                "delta": float(np.mean(heuristic_ub)) - float(np.mean(baseline_ub)),
                "t_stat": t_ub,
                "p_value": p_ub,
                "cohens_d": d_ub,
                "n": min(len(heuristic_ub), len(baseline_ub)),
            },
        }

    return stat_results


def _print_results_table(all_results: dict, n: int) -> None:
    """Print main results table."""
    col = {"Policy": 20, "SR": 14, "SP": 16, "AvgOps": 9, "U@B": 14}

    def header() -> str:
        return (
            f"| {'Policy':<{col['Policy']}} "
            f"| {'SupportRecall':>{col['SR']}} "
            f"| {'SupportPrec':>{col['SP']}} "
            f"| {'AvgOps':>{col['AvgOps']}} "
            f"| {'Utility@B':>{col['U@B']}} |"
        )

    def sep() -> str:
        return f"| {'-'*col['Policy']} | {'-'*col['SR']} | {'-'*col['SP']} | {'-'*col['AvgOps']} | {'-'*col['U@B']} |"

    def row(pname: str, agg: dict) -> str:
        return (
            f"| {pname:<{col['Policy']}} "
            f"| {agg.get('support_recall', 0):>{col['SR']}.4f} "
            f"| {agg.get('support_precision', 0):>{col['SP']}.4f} "
            f"| {agg.get('operations_used', 0):>{col['AvgOps']}.2f} "
            f"| {agg.get('utility_at_budget', 0):>{col['U@B']}.4f} |"
        )

    print(f"\n{'='*80}")
    print(f"=== BRIGHT Retrieval Results (N={n}) ===")
    print(f"{'='*80}")
    print(header())
    print(sep())
    for pname, result in all_results.items():
        print(row(pname, result["aggregated"]))
    print()


def _print_statistics(stat_results: dict) -> None:
    """Print statistical test results."""
    print(f"\n{'─'*70}")
    print("Statistical Tests: pi_aea_heuristic vs baselines")
    print(f"{'─'*70}")
    print(f"\n{'Comparison':<35} {'Metric':<20} {'Delta':>8} {'p':>10} {'d':>8}")
    print(f"{'-'*35} {'-'*20} {'-'*8} {'-'*10} {'-'*8}")

    for comp, metrics in stat_results.items():
        baseline = comp.replace("heuristic_vs_", "")
        for metric, vals in metrics.items():
            delta = vals["delta"]
            p = vals["p_value"]
            d = vals["cohens_d"]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(
                f"{baseline:<35} {metric:<20} {delta:>+8.4f} {p:>10.4f} {d:>+8.3f} {sig}"
            )
    print()


def _print_key_finding(all_results: dict, stat_results: dict) -> None:
    """Print the key finding: does stopping > searching hold on BRIGHT?"""
    print(f"\n{'='*70}")
    print("KEY FINDING: Does structural stopping generalize to BRIGHT?")
    print(f"{'='*70}")

    hierarchy = ["pi_aea_heuristic", "pi_ensemble", "pi_semantic", "pi_lexical"]
    present = [p for p in hierarchy if p in all_results]

    print("\nSupportRecall ranking on BRIGHT (higher = better reasoning retrieval):")
    recalls = [(p, all_results[p]["aggregated"]["support_recall"]) for p in present]
    recalls.sort(key=lambda x: -x[1])
    for pname, r in recalls:
        print(f"  {pname:<25} {r:.4f}")

    print(f"\n  Ranked: {' > '.join(p for p, _ in recalls)}")

    # Check if heuristic is #1
    if recalls and recalls[0][0] == "pi_aea_heuristic":
        print("\n  RESULT: Heuristic WINS on BRIGHT (generalization confirmed)")
    elif recalls and recalls[0][0] != "pi_aea_heuristic":
        heuristic_rank = next(
            (i + 1 for i, (p, _) in enumerate(recalls) if p == "pi_aea_heuristic"), None
        )
        print(f"\n  RESULT: Heuristic ranks #{heuristic_rank} on BRIGHT")

    # Report key stat
    key = "heuristic_vs_pi_semantic"
    if key in stat_results:
        sr_stats = stat_results[key]["support_recall"]
        print(
            f"\n  vs pi_semantic: delta={sr_stats['delta']:+.4f}, "
            f"p={sr_stats['p_value']:.4f}, d={sr_stats['cohens_d']:+.3f} "
            f"(N={sr_stats['n']})"
        )

    print()
    print("Interpretation:")
    if all_results:
        heuristic_sr = all_results.get("pi_aea_heuristic", {}).get("aggregated", {}).get("support_recall", 0)
        semantic_sr = all_results.get("pi_semantic", {}).get("aggregated", {}).get("support_recall", 0)
        lexical_sr = all_results.get("pi_lexical", {}).get("aggregated", {}).get("support_recall", 0)

        if heuristic_sr > semantic_sr and heuristic_sr > lexical_sr:
            print("  The structural diversity signal transfers to reasoning-intensive retrieval.")
            print("  Source diversity is distribution-invariant — the thesis generalizes.")
        elif heuristic_sr > lexical_sr:
            print("  Heuristic beats BM25 but semantic density matters more on BRIGHT.")
            print("  BRIGHT's low lexical overlap makes semantic retrieval harder to stop early.")
        else:
            print("  Heuristic does not clearly win on BRIGHT.")
            print("  The structural signal may be specific to multi-hop factoid QA.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = run_bright(
        n_examples=N_EXAMPLES,
        seed=SEED,
        results_path=RESULTS_FILE,
    )
    print("Done.")
