"""
Heterogeneous Benchmark v2 for AEA Framework.

100 questions across 6 task types, with design flaws from v1 corrected:

v1 design flaws fixed
---------------------
1. Entity Bridge:   the SECOND gold paragraph no longer shares entity names with
   the question.  The bridge entity (EntityB) appears only in P1; the question
   only names EntityA.

2. Implicit Bridge: the director/bridge entity name is NOT in the question.
   P2 vocabulary is orthogonal to the question's vocabulary.

3. Low Lexical Overlap:  drug names no longer appear verbatim in the question.
   Questions describe symptoms/conditions in lay language; gold paragraphs use
   clinical terminology without repeating the lay words.

4. Multi-Hop Chain:  simplified to strict 2-hop chains (P1 → P2 → answer)
   instead of 3-hop.  The question only names the entity in P1.

Types 3 and 6 (Semantic+Computation, Discovery+Extraction) are unchanged from
v1 because they performed well.

Entity Isolation Rule
---------------------
For entity_bridge and implicit_bridge tasks the question text ONLY shares
entities with the FIRST gold paragraph.  Subsequent gold paragraphs are
reachable ONLY through entities discovered in previous paragraphs.

Lexical Isolation Rule
----------------------
For low_lexical_overlap tasks: word overlap between question and gold paragraph
must be < 15 % of content words (excluding stop words).

Validation
----------
After construction, ``HeterogeneousBenchmarkV2.validate()`` checks:
  * entity_bridge / implicit_bridge: bridge entity not in question
  * low_lexical_overlap:             overlap < 15 %
and prints a report.

Task types
----------
1. entity_bridge      (20 q)  — person → birthplace → attribute (entity hop)
2. implicit_bridge    (20 q)  — work → creator (implicit) → fact
3. semantic_computation (20 q) — compare two companies' revenues
4. low_lexical_overlap (20 q) — lay-language question, clinical gold
5. multi_hop_chain    (10 q)  — two-hop A → B → answer
6. discovery_extraction (10 q) — index → dept budget
"""

from __future__ import annotations

import re
import random
from typing import Optional

from experiments.aea.types import AddressSpaceType


# ─────────────────────────────────────────────────────────────────────────────
# Stop-word set for overlap checking
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "in", "on", "at", "by", "for", "with", "from", "to", "of", "about",
    "above", "after", "before", "between", "into", "through", "during",
    "and", "or", "but", "so", "yet", "nor", "not", "no",
    "this", "that", "these", "those", "it", "its", "its",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "hers", "they", "them", "their",
    "each", "every", "both", "all", "any", "few", "more", "most",
    "other", "some", "such", "only", "same", "than", "then", "too",
    "very", "s", "t", "just", "as", "if",
})


def _content_words(text: str) -> set[str]:
    """Return lower-cased non-stop-word tokens from *text*."""
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _word_overlap_fraction(text_a: str, text_b: str) -> float:
    """
    Jaccard-style fraction: |intersection| / |union| on content word sets.
    Returns 0.0 if union is empty.
    """
    cw_a = _content_words(text_a)
    cw_b = _content_words(text_b)
    if not cw_a or not cw_b:
        return 0.0
    intersection = cw_a & cw_b
    union = cw_a | cw_b
    return len(intersection) / len(union)


# ─────────────────────────────────────────────────────────────────────────────
# Template vocabulary pools
# ─────────────────────────────────────────────────────────────────────────────

# TYPE 1 — ENTITY BRIDGE
# (person_name, birthplace_city, attribute_label, attribute_value)
# Question only mentions person_name; P1 reveals birthplace; P2 reveals attribute.
_ENTITY_BRIDGE_V2_POOL = [
    ("Marcus Thornwell",   "Crestfield",      "population",       "42,000"),
    ("Sylvie Dunmore",     "Harrowgate",      "founded_year",     "1683"),
    ("Odell Krasner",      "Velstrom",        "elevation_metres", "340"),
    ("Fenella Gust",       "Brindlemoor",     "population",       "18,500"),
    ("Caspar Weld",        "Thornwich",       "area_sq_km",       "210"),
    ("Nadia Solberg",      "Aldervane",       "river",            "the Calven"),
    ("Idris Pembroke",     "Duskhollow",      "county",           "West Valmere"),
    ("Rosalind Kray",      "Porthmere",       "nickname",         "the Amber City"),
    ("Bram Elstowe",       "Wychford",        "cathedral_name",   "Saint Aldric's Cathedral"),
    ("Thea Calvert",       "Greymount",       "university",       "Greymount Polytechnic"),
    ("Soren Blund",        "Oskveld",         "population",       "31,000"),
    ("Mirela Voss",        "Castleway",       "founded_year",     "1421"),
    ("Hakan Freyr",        "Thornbrook",      "elevation_metres", "520"),
    ("Lena Sparre",        "Coldveil",        "area_sq_km",       "95"),
    ("Rook Aldgate",       "Strandmore",      "river",            "the Holvern"),
    ("Petra Ulfar",        "Glenholt",        "county",           "North Dunmere"),
    ("Evander Kesh",       "Salthaven",       "nickname",         "the Pearl of the Coast"),
    ("Yara Moltke",        "Rimwick",         "cathedral_name",   "Saint Petra's Basilica"),
    ("Gideon Strand",      "Fordmere",        "university",       "Fordmere College of Sciences"),
    ("Elin Crowe",         "Ashvale",         "population",       "67,000"),
]

# TYPE 2 — IMPLICIT BRIDGE
# (work_title, creator_name, fact_about_creator, fact_value)
# Question asks about a fact of the creator of the work, but does NOT name the creator.
# P1: about the work, mentions the creator's name.
# P2: about the creator, contains the fact — no shared words with question.
_IMPLICIT_BRIDGE_V2_POOL = [
    ("Midnight Crossing",          "Diana Shelbourne",    "attended_university",   "Ashworth University"),
    ("The Ironwood Codex",         "Piers Callon",        "birth_year",            "1962"),
    ("Songs of the Pale Shore",    "Maren Frost",         "prize_won",             "the Lindell Prize for Literature"),
    ("The Obsidian Cartographer",  "Tomas Quellar",       "home_country",          "Valdoria"),
    ("Last Train to Elbronn",      "Sarai Bek",           "attended_university",   "Delmoor Institute"),
    ("The Second Moon",            "Arild Vance",         "birth_year",            "1971"),
    ("Echoes of the Interior",     "Nneka Osei",          "prize_won",             "the Crestfield Award"),
    ("Glass Meridian",             "Pascal Dreier",       "home_country",          "Halvenia"),
    ("Corridor of Hours",          "Faye Moreau",         "attended_university",   "Thornwick Academy of Arts"),
    ("The Amber Insurgent",        "Brennan Wolfe",       "birth_year",            "1957"),
    ("Fractures in Blue",          "Kezia Strand",        "prize_won",             "the Veluris Cultural Medal"),
    ("The Porcelain Cartographer", "Omar Hadidi",         "home_country",          "Zarathon"),
    ("Hollow Ground",              "Astrid Bjork",        "attended_university",   "Peloria School of Cinema"),
    ("Woven in Fire",              "Remy Chalcott",       "birth_year",            "1965"),
    ("After the Vanishing",        "Linh Nguyen",         "prize_won",             "the Saoirse Grand Jury Award"),
    ("The Quiet Insurgent",        "Evander Mills",       "home_country",          "Aquinda"),
    ("Beneath Still Waters",       "Nadine Blum",         "attended_university",   "Gondurex University"),
    ("Iron Meridian",              "Priya Rajan",         "birth_year",            "1980"),
    ("Starfall Protocol",          "Matias Calderon",     "prize_won",             "the Athens Narrative Award"),
    ("The Last Cartographer",      "Saoirse Kern",        "home_country",          "Duvenmark"),
]

# TYPE 3 — SEMANTIC + COMPUTATION (unchanged from v1)
_SEMANTIC_COMPUTATION_POOL = [
    ("Valtec Industries",     840,   "Norbridge Corp",    620,   "Valtec Industries"),
    ("Greenfield Systems",    310,   "Apex Dynamics",     490,   "Apex Dynamics"),
    ("Crestline Holdings",    1050,  "Duneport Ventures", 780,   "Crestline Holdings"),
    ("Thornwall Enterprises", 225,   "Ironclad Solutions",380,   "Ironclad Solutions"),
    ("SkyMarket Co",          670,   "PinePath Ltd",      530,   "SkyMarket Co"),
    ("Meridian Foundry",      415,   "Coldstream AG",     560,   "Coldstream AG"),
    ("Harborgate Inc",        990,   "Sunridge Partners", 730,   "Harborgate Inc"),
    ("Polaris Works",         145,   "Ember Capital",     210,   "Ember Capital"),
    ("Redwood Financial",     880,   "Siltstone Corp",    950,   "Siltstone Corp"),
    ("Falcrest Manufacturing",340,   "Dawnhollow Tech",   270,   "Falcrest Manufacturing"),
    ("Cloudbend Systems",     505,   "Tidewater Group",   490,   "Cloudbend Systems"),
    ("Frostpeak Retail",      760,   "Sandgate Holdings", 820,   "Sandgate Holdings"),
    ("Ironbark Logistics",    430,   "Goldenrod Inc",     395,   "Ironbark Logistics"),
    ("Highcliff Analytics",   185,   "Ravenmore Digital", 260,   "Ravenmore Digital"),
    ("Stonecroft Chemicals",  635,   "Wavehaven Corp",    580,   "Stonecroft Chemicals"),
    ("Duskbell Telecom",      910,   "Brightfall Media",  870,   "Duskbell Telecom"),
    ("Crownfield Pharma",     445,   "Wintera Biotech",   510,   "Wintera Biotech"),
    ("Glassgate Retail",      295,   "Thornwick Goods",   320,   "Thornwick Goods"),
    ("Mooncrest Energy",      700,   "Bramblewood Power", 665,   "Mooncrest Energy"),
    ("Riverstride Tech",      550,   "Dalebrook Software",575,   "Dalebrook Software"),
]

# TYPE 4 — LOW LEXICAL OVERLAP (v2: question uses pure lay language, no drug name)
# (question_lay, gold_drug_name, gold_disease_term, gold_paragraph)
# The question must NOT contain the drug name or the clinical disease term.
# The gold paragraph must NOT contain the lay words from the question (beyond 1–2 bridging words).
_LOW_LEXICAL_V2_POOL = [
    # 1 — insomnia
    (
        "What medication helps people who cannot fall asleep at night?",
        "Somniplex",
        "chronic insomnia",
        "Somniplex is a pharmaceutical compound indicated for chronic insomnia, a persistent sleep-onset disorder. "
        "It potentiates GABA-A receptor activity, reducing sleep latency and nocturnal awakenings. "
        "Clinical trials demonstrated Somniplex reduced time-to-sleep-onset by 62 percent versus placebo.",
    ),
    # 2 — hypertension
    (
        "Which drug lowers dangerously high blood pressure in patients at risk of heart attack?",
        "Artevalin",
        "hypertension",
        "Artevalin is a calcium-channel antagonist prescribed for essential hypertension and hypertensive crisis. "
        "It dilates peripheral arterioles, reducing systemic vascular resistance and afterload. "
        "Cardiologists administer Artevalin when systolic readings exceed 180 mmHg.",
    ),
    # 3 — depression
    (
        "What pill helps someone who feels extremely sad, hopeless, and cannot enjoy anything?",
        "Luminorel",
        "major depressive disorder",
        "Luminorel is a selective serotonin-norepinephrine reuptake inhibitor approved for major depressive disorder. "
        "It elevates synaptic monoamine concentrations to alleviate anhedonia and dysphoria. "
        "Psychiatrists prescribe Luminorel as a first-line pharmacotherapy for recurrent affective illness.",
    ),
    # 4 — bacterial pneumonia
    (
        "What medicine treats a severe chest infection that makes breathing very painful?",
        "Corvacin",
        "bacterial pneumonia",
        "Corvacin is an aminopenicillin antibiotic active against Streptococcus pneumoniae and Haemophilus influenzae. "
        "It achieves high pulmonary tissue concentrations suitable for lobar consolidation. "
        "Infectious disease specialists administer Corvacin intravenously for hospitalised respiratory failure.",
    ),
    # 5 — Crohn's disease
    (
        "What treatment reduces intestinal pain and frequent toilet visits in young adults with a gut condition?",
        "Durafen",
        "Crohn's disease",
        "Durafen is a biologic agent targeting tumour necrosis factor-alpha, indicated for moderate-to-severe Crohn's disease. "
        "It suppresses granulomatous inflammation in the terminal ileum and colon. "
        "Gastroenterologists initiate Durafen when conventional immunosuppressants have failed to achieve mucosal healing.",
    ),
    # 6 — epilepsy
    (
        "Which drug stops people from shaking uncontrollably and losing consciousness suddenly?",
        "Nexalin",
        "epilepsy",
        "Nexalin is an anticonvulsant approved as adjunctive therapy for focal and generalised epilepsy. "
        "It stabilises neuronal membranes by blocking voltage-sensitive sodium channels. "
        "Neurologists titrate Nexalin slowly to minimise dizziness and ataxia during initiation.",
    ),
    # 7 — autoimmune hepatitis
    (
        "What medicine protects the liver in people whose immune system keeps attacking it?",
        "Provatex",
        "autoimmune hepatitis",
        "Provatex suppresses hepatic T-cell infiltration and autoantibody production in autoimmune hepatitis. "
        "It modulates calcineurin-dependent lymphocyte activation pathways. "
        "Hepatologists initiate Provatex in patients with elevated transaminases unresponsive to corticosteroids.",
    ),
    # 8 — muscular dystrophy
    (
        "What drug helps children whose muscles become progressively weaker with no apparent injury?",
        "Solvex",
        "muscular dystrophy",
        "Solvex promotes dystrophin production via exon-skipping in Duchenne muscular dystrophy patients. "
        "It restores partial reading-frame correction of the dystrophin gene. "
        "Paediatric neurologists administer Solvex as an intrathecal infusion during early disease stages.",
    ),
    # 9 — hypercholesterolaemia
    (
        "Which pill helps people whose arteries are blocked by too much fatty material?",
        "Talviren",
        "hypercholesterolaemia",
        "Talviren is a third-generation statin that inhibits HMG-CoA reductase in hepatocytes. "
        "It substantially reduces LDL cholesterol and slows atherosclerotic plaque progression. "
        "Cardiologists prescribe Talviren to patients at elevated cardiovascular risk with dyslipidaemia.",
    ),
    # 10 — osteoporosis
    (
        "What treatment makes fragile bones stronger in elderly women to prevent fractures?",
        "Venodrix",
        "osteoporosis",
        "Venodrix is a bisphosphonate compound that inhibits osteoclast-mediated bone resorption. "
        "It increases bone mineral density at the lumbar spine and femoral neck. "
        "Rheumatologists prescribe Venodrix weekly to postmenopausal patients with dual-energy X-ray densitometry T-scores below −2.5.",
    ),
    # 11 — type 2 diabetes
    (
        "Which medication regulates dangerously high sugar levels in people who are overweight?",
        "Caldrex",
        "type 2 diabetes mellitus",
        "Caldrex is a biguanide that reduces hepatic gluconeogenesis and improves peripheral insulin sensitivity. "
        "It is the pharmacological cornerstone for glycaemic control in type 2 diabetes mellitus. "
        "Endocrinologists titrate Caldrex alongside dietary modification to achieve target HbA1c values.",
    ),
    # 12 — tinea corporis
    (
        "What cream cures a circular itchy rash that spreads outward on the skin?",
        "Morvane",
        "tinea corporis",
        "Morvane is a topical azole antifungal that disrupts ergosterol synthesis in dermatophyte cell membranes. "
        "It is indicated for tinea corporis and tinea cruris caused by Trichophyton species. "
        "Dermatologists recommend Morvane twice daily for four weeks to achieve mycological cure.",
    ),
    # 13 — atrial fibrillation
    (
        "What drug stops the heart from beating irregularly and prevents stroke risk?",
        "Bevalux",
        "atrial fibrillation",
        "Bevalux is a class-III antiarrhythmic that prolongs the cardiac action potential by blocking potassium channels. "
        "It restores sinus rhythm in persistent atrial fibrillation and reduces thromboembolic events. "
        "Electrophysiologists initiate Bevalux under continuous ECG monitoring due to proarrhythmic potential.",
    ),
    # 14 — allergic rhinitis
    (
        "Which pill reduces runny nose and constant sneezing after touching animals or pollen?",
        "Zeltramine",
        "allergic rhinitis",
        "Zeltramine is a second-generation H1 antihistamine with minimal sedating properties. "
        "It blocks histamine H1 receptors in nasal mucosa, reducing rhinorrhoea and nasal pruritus. "
        "Allergists prescribe Zeltramine for seasonal and perennial allergic rhinitis and allergic conjunctivitis.",
    ),
    # 15 — hypothyroidism
    (
        "What medicine speeds up the body of people who feel constantly cold and tired with slow metabolism?",
        "Corilene",
        "hypothyroidism",
        "Corilene is a synthetic levothyroxine formulation that replaces deficient endogenous thyroid hormone. "
        "It restores normal basal metabolic rate and alleviates hypothyroid sequelae. "
        "Endocrinologists titrate Corilene by monitoring serum thyroid-stimulating hormone concentrations.",
    ),
    # 16 — pulmonary fibrosis
    (
        "What drug slows down the hardening of lung tissue in patients who struggle to breathe?",
        "Fibrexol",
        "idiopathic pulmonary fibrosis",
        "Fibrexol is an anti-fibrotic tyrosine-kinase inhibitor that reduces collagen deposition in alveolar tissue. "
        "It slows the decline in forced vital capacity in idiopathic pulmonary fibrosis. "
        "Pulmonologists initiate Fibrexol when high-resolution CT confirms honeycombing and traction bronchiectasis.",
    ),
    # 17 — peptic ulcer disease
    (
        "Which medicine heals open sores inside the stomach that cause burning pain after eating?",
        "Rantivex",
        "peptic ulcer disease",
        "Rantivex is a proton-pump inhibitor that irreversibly binds the H+/K+-ATPase enzyme in gastric parietal cells. "
        "It achieves near-complete acid suppression, enabling mucosal healing in peptic ulcer disease. "
        "Gastroenterologists prescribe Rantivex for eight weeks to achieve endoscopic ulcer closure.",
    ),
    # 18 — clinical depression / generalised anxiety (dual)
    (
        "What treatment calms overwhelming worry and persistent low mood at the same time?",
        "Deltraze",
        "generalised anxiety disorder and major depressive disorder",
        "Deltraze is a serotonin-norepinephrine reuptake inhibitor indicated for co-morbid generalised anxiety disorder and major depressive disorder. "
        "It normalises limbic hyper-reactivity and prefrontal executive dysfunction simultaneously. "
        "Psychiatrists favour Deltraze when anxious dysphoria fails to respond to SSRI monotherapy.",
    ),
    # 19 — narcolepsy
    (
        "Which drug helps people stay awake during the day when they fall asleep uncontrollably?",
        "Quarvolide",
        "narcolepsy",
        "Quarvolide is a wakefulness-promoting agent that enhances dopaminergic signalling in the hypothalamic arousal circuit. "
        "It is indicated for narcolepsy with cataplexy and reduces excessive daytime somnolence. "
        "Neurologists prescribe Quarvolide as an alternative to amphetamine-class stimulants.",
    ),
    # 20 — neuropathic pain
    (
        "What medicine relieves constant burning or electric-shock sensations in the hands and feet?",
        "Norplexin",
        "neuropathic pain",
        "Norplexin modulates voltage-gated calcium channels in dorsal root ganglia to attenuate ectopic neural discharges. "
        "It is indicated for peripheral neuropathic pain, including diabetic polyneuropathy and post-herpetic neuralgia. "
        "Pain specialists titrate Norplexin gradually to minimise dizziness and peripheral oedema.",
    ),
]

# TYPE 5 — MULTI-HOP CHAIN (v2: 2-hop only, A → B → answer)
# (a_entity, a_link_phrase, b_entity, b_answer_phrase, final_answer)
# Question ONLY names entity A; answer lives in B.
_MULTI_HOP_V2_POOL = [
    (
        "Thornwick Station",
        "located in the borough of Halgrave",
        "Halgrave",
        "administered by Director Everard Voss",
        "Everard Voss",
    ),
    (
        "Blue Pine Lake",
        "fed by the Kallos River",
        "Kallos River",
        "originates in the glaciers of Stormgard",
        "Stormgard",
    ),
    (
        "Alveron Museum",
        "founded by Ida Solten",
        "Ida Solten",
        "born in the year 1831",
        "1831",
    ),
    (
        "Corvax Tower",
        "designed by architect Bryce Naldor",
        "Bryce Naldor",
        "trained at the Halden School of Design",
        "Halden School of Design",
    ),
    (
        "Mendover Bridge",
        "commissioned by the Stelvane Company",
        "Stelvane Company",
        "headquartered in the port of Orvath",
        "Orvath",
    ),
    (
        "Clearwater Accord",
        "signed by envoy Dara Lindqvist",
        "Dara Lindqvist",
        "represented the nation of Valdris",
        "Valdris",
    ),
    (
        "Pellar Institute",
        "headed by Professor Gale Astor",
        "Gale Astor",
        "holds a doctorate from Crestmoor University",
        "Crestmoor University",
    ),
    (
        "Sandreth Canal",
        "built by the Ironside Company",
        "Ironside Company",
        "registered under Delvorne trade law",
        "Delvorne",
    ),
    (
        "Vorde Festival",
        "held annually in the town of Brynwick",
        "Brynwick",
        "seat of the Brynwick Heritage Council, founded in 1903",
        "1903",
    ),
    (
        "Glassholm Lighthouse",
        "operated by Maritime Board Severn",
        "Maritime Board Severn",
        "reports to Harbour Master Elara Keld",
        "Elara Keld",
    ),
]

# TYPE 6 — DISCOVERY + EXTRACTION (unchanged from v1)
_DISCOVERY_EXTRACTION_POOL = [
    ("Infrastructure",        "road maintenance",              42),
    ("Public Health",         "disease prevention",            78),
    ("Education",             "school curriculum",             115),
    ("Emergency Services",    "first responder coordination",  63),
    ("Urban Planning",        "zoning and land use",           31),
    ("Environmental Affairs", "waste management",              55),
    ("Housing Authority",     "affordable housing",            89),
    ("Transport",             "public transit systems",        97),
    ("Social Services",       "welfare support",               48),
    ("Economic Development",  "business licensing",            37),
]


# ─────────────────────────────────────────────────────────────────────────────
# Distractor vocabulary
# ─────────────────────────────────────────────────────────────────────────────

_DISTRACTOR_PEOPLE = [
    "Aldric Pemble", "Bess Falconer", "Caius Wren", "Dora Stonehaven",
    "Elara Vance", "Forrest Pell", "Gemma Dunmore", "Harlan Breck",
    "Iris Coldwell", "Jonas Stead", "Kira Holm", "Leander Crowe",
    "Mira Solt", "Noel Carver", "Ondine Marsh", "Pierce Aldwick",
    "Rosalind Thorn", "Silas Greer", "Thea Morten", "Ulric Brand",
]

_DISTRACTOR_PLACES = [
    "Keldmoor", "Brannport", "Arvelund", "Silvast", "Forsheim",
    "Stormgard", "Ironholt", "Whitecliff", "Stormbeck", "Emberton",
    "Grantholm", "Blackridge", "Castlewick", "Dunmore Falls", "Eldenvale",
    "Foxmere", "Greystone Crossing", "Hawthorn Bay", "Invermere", "Jeskwall",
]

_DISTRACTOR_COMPANIES = [
    "Aldervane Corp", "Blackthorn Holdings", "Cascadia Industries",
    "Driftwood Systems", "Elarion Ltd", "Felspar Partners", "Gravelside AG",
    "Hillcrest Dynamics", "Irontide Ventures", "Jackdaw Solutions",
    "Keystone Analytics", "Larchmont Manufacturing", "Moorfield Inc",
    "Northgate Capital", "Oakhaven Digital", "Pinecrest Retail",
    "Quarryside Energy", "Ridgeline Pharma", "Saltmarsh Telecom", "Timberfield Co",
]


def _make_distractor_para(rng: random.Random) -> str:
    """Generate a plausible distractor paragraph unrelated to the answer."""
    person  = rng.choice(_DISTRACTOR_PEOPLE)
    place   = rng.choice(_DISTRACTOR_PLACES)
    company = rng.choice(_DISTRACTOR_COMPANIES)
    year    = rng.randint(1890, 2010)
    topics  = [
        (
            f"{person} was born in {place} and later moved to {company}'s headquarters in {year}. "
            f"During that period, {place} underwent significant urban renewal. "
            f"The city's archives record {person} as a prominent figure in local history."
        ),
        (
            f"The {company} was incorporated in {year} in the district of {place}. "
            f"Its founding charter was drafted by {person}, a local administrator. "
            f"The company expanded rapidly across the region within five years."
        ),
        (
            f"In {year}, {place} hosted the International Trade Summit. "
            f"{company} was among the exhibitors, represented by delegate {person}. "
            f"The summit concluded with several bilateral agreements on tariffs."
        ),
        (
            f"{person} published a historical account of {place} in {year}. "
            f"The monograph detailed the city's industrial rise under {company}'s patronage. "
            f"Academic reviewers praised the work for its archival depth."
        ),
        (
            f"The cultural heritage of {place} is preserved by the {place} Heritage Foundation. "
            f"{person} served as the foundation's chairperson for over a decade. "
            f"Annual grants from {company} sustain the foundation's restoration projects."
        ),
    ]
    return rng.choice(topics)


# ─────────────────────────────────────────────────────────────────────────────
# Individual task-type generators
# ─────────────────────────────────────────────────────────────────────────────

def _gen_entity_bridge_v2(rng: random.Random, idx: int, pool_entry: tuple) -> dict:
    """
    Entity Bridge (v2):
    Question: "What is the [attribute] of the birthplace of [PersonName]?"
      — ONLY mentions PersonName; does NOT mention the birthplace city.
    P1: About PersonName; REVEALS the birthplace city.
    P2: About the birthplace city; contains the attribute value.
    BM25 on question → finds P1 (matches PersonName).
    Entity hop: P1 → extract city → find P2.
    """
    person_name, birthplace, attribute_label, attribute_value = pool_entry
    qid = f"entity_bridge_{idx:02d}"

    # Attribute labels map to natural-language question phrasing
    attr_q_map = {
        "population":       "What is the population of the birthplace of",
        "founded_year":     "In what year was the birthplace of",
        "elevation_metres": "What is the elevation in metres of the birthplace of",
        "area_sq_km":       "What is the area in square kilometres of the birthplace of",
        "river":            "Which river runs through the birthplace of",
        "county":           "Which county contains the birthplace of",
        "nickname":         "What is the nickname of the birthplace of",
        "cathedral_name":   "What cathedral is located in the birthplace of",
        "university":       "Which university is located in the birthplace of",
    }
    attr_q_suffix_map = {
        "population":       f"{person_name}?",
        "founded_year":     f"{person_name} first settled?",
        "elevation_metres": f"{person_name}?",
        "area_sq_km":       f"{person_name}?",
        "river":            f"{person_name}?",
        "county":           f"{person_name}?",
        "nickname":         f"{person_name}?",
        "cathedral_name":   f"{person_name}?",
        "university":       f"{person_name}?",
    }
    question = (
        attr_q_map.get(attribute_label, f"What is the {attribute_label} of the birthplace of")
        + " "
        + attr_q_suffix_map.get(attribute_label, f"{person_name}?")
    )

    # Gold P1: mentions person, reveals birthplace city name — nothing else about city attribute
    person_para = (
        f"{person_name} is a notable figure in the fields of science and culture. "
        f"Born and raised in {birthplace}, {person_name.split()[0]} left for advanced studies abroad "
        f"at the age of seventeen. "
        f"Colleagues describe {person_name.split()[0]} as a dedicated researcher with broad interests."
    )

    # Gold P2: about the birthplace city — contains the attribute; does NOT mention person_name
    attr_sentence_map = {
        "population":       f"{birthplace} is a small city with a population of {attribute_value}.",
        "founded_year":     f"{birthplace} was first settled in {attribute_value}, making it one of the older towns in the region.",
        "elevation_metres": f"{birthplace} sits at an elevation of {attribute_value} metres above sea level.",
        "area_sq_km":       f"{birthplace} covers an area of {attribute_value} square kilometres.",
        "river":            f"{birthplace} straddles {attribute_value}, which supplies fresh water to the municipality.",
        "county":           f"{birthplace} is situated in {attribute_value}, in the western administrative zone.",
        "nickname":         f"{birthplace} is popularly referred to as {attribute_value} due to its amber-coloured stone buildings.",
        "cathedral_name":   f"The most prominent building in {birthplace} is {attribute_value}, a Gothic structure dating to the fifteenth century.",
        "university":       f"{birthplace} is home to {attribute_value}, a well-regarded institution of higher learning.",
    }
    city_para = (
        f"{birthplace} is a settlement in the central province. "
        + attr_sentence_map.get(attribute_label, f"Its {attribute_label} is {attribute_value}.")
        + f" Locals take pride in {birthplace}'s heritage and annual civic festivals."
    )

    # Distractors
    distractors = []
    # Distractor 0: mentions person but says nothing about birthplace attribute
    distractors.append({
        "id": f"{qid}_dist_0",
        "title": f"{person_name} Academic Works",
        "content": (
            f"{person_name} authored several monographs on comparative linguistics. "
            f"The publications were widely cited in academic journals across Europe. "
            f"Colleagues praised the rigor and originality of the research."
        ),
    })
    # Distractor 1: mentions the city but not the specific attribute value
    distractors.append({
        "id": f"{qid}_dist_1",
        "title": f"{birthplace} Tourism",
        "content": (
            f"Visitors to {birthplace} enjoy its well-preserved market square. "
            f"The city offers a range of cultural exhibitions and seasonal festivals. "
            f"Accommodation options include boutique hotels and bed-and-breakfast establishments."
        ),
    })
    for di in range(6):
        distractors.append({
            "id": f"{qid}_dist_{di + 2}",
            "title": f"Distractor {qid} {di}",
            "content": _make_distractor_para(rng),
        })

    context = [
        {"id": f"{qid}_person", "title": person_name,  "content": person_para},
        {"id": f"{qid}_city",   "title": birthplace,   "content": city_para},
    ] + distractors
    rng.shuffle(context)

    return {
        "id": qid,
        "question": question,
        "answer": attribute_value,
        "task_type": "entity_bridge",
        "bridge_entity": birthplace,   # must NOT appear in question
        "supporting_facts": [f"{qid}_person", f"{qid}_city"],
        "gold_ids":         [f"{qid}_person", f"{qid}_city"],
        "context": context,
        "required_substrates": [AddressSpaceType.SEMANTIC, AddressSpaceType.ENTITY],
        "num_hops": 2,
    }


def _gen_implicit_bridge_v2(rng: random.Random, idx: int, pool_entry: tuple) -> dict:
    """
    Implicit Bridge (v2):
    Question: "What is the [fact] of the [role] of [Work]?"
      — mentions Work but NOT the creator's name.
    P1: About the Work; REVEALS the creator's name.
    P2: About the creator; contains the fact — vocabulary orthogonal to question.
    BM25 → finds P1 (matches Work title). Misses P2 (no overlap).
    Semantic search on question → also finds P1 (Work + role context).
    Entity hop from P1: extract creator name → find P2.
    """
    work_title, creator_name, fact_label, fact_value = pool_entry
    qid = f"implicit_bridge_{idx:02d}"

    # Role descriptions per fact type — question phrasing
    fact_q_map = {
        "attended_university": f"What university did the author of \"{work_title}\" attend?",
        "birth_year":          f"In what year was the creator of \"{work_title}\" born?",
        "prize_won":           f"What prize did the writer of \"{work_title}\" receive?",
        "home_country":        f"Which country is the director of \"{work_title}\" from?",
    }
    question = fact_q_map.get(fact_label, f"What is the {fact_label} of the creator of \"{work_title}\"?")

    # Gold P1: about the work; reveals creator name; shares lexical content with question
    work_para = (
        f"\"{work_title}\" is a celebrated work that received wide critical attention. "
        f"It was created by {creator_name}, whose meticulous craft is evident throughout. "
        f"The work has been translated into fourteen languages and remains widely studied."
    )

    # Gold P2: about the creator — does NOT mention the work title; vocabulary different from question
    fact_sentence_map = {
        "attended_university": (
            f"{creator_name} pursued higher education at {fact_value}, graduating with distinction in the arts. "
            f"The institution's interdisciplinary programme shaped the creator's formative intellectual approach."
        ),
        "birth_year": (
            f"{creator_name} was born in {fact_value} in a coastal region known for its literary culture. "
            f"Early childhood experiences profoundly influenced the thematic preoccupations of subsequent output."
        ),
        "prize_won": (
            f"{creator_name} received {fact_value} in recognition of sustained contributions to contemporary culture. "
            f"The award jury cited the creator's innovative use of structure and voice as primary reasons for selection."
        ),
        "home_country": (
            f"{creator_name} is a native of {fact_value}, where formative cultural influences shaped a distinctive aesthetic sensibility. "
            f"The national tradition of storytelling pervades the creator's body of work."
        ),
    }
    creator_para = (
        f"{creator_name} is an internationally recognised creative figure. "
        + fact_sentence_map.get(fact_label, f"Their {fact_label} is {fact_value}.")
        + f" {creator_name.split()[0]} continues to be cited as an influence by emerging voices in the field."
    )

    # Distractors
    distractors = []
    # Distractor 0: about the work but irrelevant to the fact
    distractors.append({
        "id": f"{qid}_dist_0",
        "title": f"{work_title} Reception",
        "content": (
            f"\"{work_title}\" debuted to strong reviews at several prominent cultural festivals. "
            f"Audiences responded warmly to its unconventional narrative approach. "
            f"Home distribution rights were acquired by a major international platform."
        ),
    })
    # Distractor 1: about a different creator (role-adjacent distractor)
    other_creator = rng.choice(_DISTRACTOR_PEOPLE)
    distractors.append({
        "id": f"{qid}_dist_1",
        "title": f"{other_creator} Profile",
        "content": (
            f"{other_creator} is a well-regarded creative professional with a broad portfolio. "
            f"Their work spans multiple genres and has attracted international festival recognition. "
            f"Critics frequently cite their contribution to the renewal of contemporary form."
        ),
    })
    for di in range(6):
        distractors.append({
            "id": f"{qid}_dist_{di + 2}",
            "title": f"Distractor {qid} {di}",
            "content": _make_distractor_para(rng),
        })

    context = [
        {"id": f"{qid}_work",    "title": work_title,    "content": work_para},
        {"id": f"{qid}_creator", "title": creator_name,  "content": creator_para},
    ] + distractors
    rng.shuffle(context)

    return {
        "id": qid,
        "question": question,
        "answer": fact_value,
        "task_type": "implicit_bridge",
        "bridge_entity": creator_name,   # must NOT appear in question
        "supporting_facts": [f"{qid}_work", f"{qid}_creator"],
        "gold_ids":         [f"{qid}_work", f"{qid}_creator"],
        "context": context,
        "required_substrates": [AddressSpaceType.SEMANTIC, AddressSpaceType.ENTITY],
        "num_hops": 2,
    }


def _gen_semantic_computation(rng: random.Random, idx: int, pool_entry: tuple) -> dict:
    """Semantic + Computation (unchanged from v1)."""
    comp_a, rev_a, comp_b, rev_b, higher = pool_entry
    qid = f"semantic_computation_{idx:02d}"

    para_a = (
        f"{comp_a} is a diversified industrial group headquartered in Valden City. "
        f"In its most recent fiscal year, the company reported annual revenue of ${rev_a} million. "
        f"The firm operates across logistics, manufacturing, and consumer goods segments."
    )
    para_b = (
        f"{comp_b} is a technology-focused enterprise listed on the Caldor Stock Exchange. "
        f"The company posted revenue of ${rev_b} million in the last reported period. "
        f"{comp_b} invests heavily in research and development, allocating 12 percent of sales."
    )

    distractors = []
    distractors.append({
        "id": f"{qid}_dist_0",
        "title": f"{comp_a} Leadership",
        "content": (
            f"{comp_a} appointed a new chief executive following its annual general meeting. "
            f"The incoming CEO brings twenty years of operational experience. "
            f"Analysts expect strategic restructuring across the firm's major divisions."
        ),
    })
    distractors.append({
        "id": f"{qid}_dist_1",
        "title": f"{comp_b} Acquisition",
        "content": (
            f"{comp_b} announced the acquisition of a regional competitor for $40 million. "
            f"The deal is expected to close pending regulatory review. "
            f"Integration plans will be announced in the fourth quarter."
        ),
    })
    for di in range(6):
        distractors.append({
            "id": f"{qid}_dist_{di + 2}",
            "title": f"Distractor {qid} {di}",
            "content": _make_distractor_para(rng),
        })

    context = [
        {"id": f"{qid}_comp_a", "title": comp_a, "content": para_a},
        {"id": f"{qid}_comp_b", "title": comp_b, "content": para_b},
    ] + distractors
    rng.shuffle(context)

    return {
        "id": qid,
        "question": f"Which company had higher revenue, {comp_a} or {comp_b}?",
        "answer": higher,
        "task_type": "semantic_computation",
        "supporting_facts": [f"{qid}_comp_a", f"{qid}_comp_b"],
        "gold_ids":         [f"{qid}_comp_a", f"{qid}_comp_b"],
        "context": context,
        "required_substrates": [AddressSpaceType.SEMANTIC, AddressSpaceType.EXECUTABLE],
        "num_hops": 1,
    }


def _gen_low_lexical_overlap_v2(rng: random.Random, idx: int, pool_entry: tuple) -> dict:
    """
    Low Lexical Overlap (v2):
    Question uses pure everyday language; no drug name; no clinical disease term.
    Gold paragraph uses pharmaceutical terminology; drug name is present; lay words absent.
    Distractors include sleep-/pain-/health-themed passages that don't contain the answer.
    """
    question, drug_name, disease_term, gold_para = pool_entry
    qid = f"low_lexical_overlap_{idx:02d}"

    distractors = []
    # Distractor 0: clinical-sounding but about a different compound, wrong indication
    alt_drug = "Placevex"
    distractors.append({
        "id": f"{qid}_dist_0",
        "title": f"{alt_drug} Overview",
        "content": (
            f"{alt_drug} is a broad-spectrum pharmacological agent under investigation for several metabolic disorders. "
            f"Phase-II trials suggest acceptable tolerability at therapeutic doses. "
            f"The compound's mechanism involves allosteric modulation of receptor subtypes."
        ),
    })
    # Distractor 1: lay-language health article that matches question semantics but gives no answer
    distractors.append({
        "id": f"{qid}_dist_1",
        "title": "Healthy Living Advice",
        "content": (
            "Experts recommend regular physical activity and a balanced diet to maintain good health. "
            "Adequate sleep, hydration, and stress management are key pillars of wellbeing. "
            "Consulting a healthcare professional is advisable before starting any new treatment."
        ),
    })
    for di in range(6):
        distractors.append({
            "id": f"{qid}_dist_{di + 2}",
            "title": f"Distractor {qid} {di}",
            "content": _make_distractor_para(rng),
        })

    gold_id = f"{qid}_gold"
    context = [
        {"id": gold_id, "title": f"{drug_name} Prescribing Information", "content": gold_para},
    ] + distractors
    rng.shuffle(context)

    return {
        "id": qid,
        "question": question,
        "answer": drug_name,
        "task_type": "low_lexical_overlap",
        "supporting_facts": [gold_id],
        "gold_ids":         [gold_id],
        "context": context,
        "required_substrates": [AddressSpaceType.SEMANTIC],
        "num_hops": 1,
    }


def _gen_multi_hop_chain_v2(rng: random.Random, idx: int, pool_entry: tuple) -> dict:
    """
    Multi-Hop Chain (v2 — 2 hops only):
    Question: "Starting from [EntityA], what is the [attribute]?"
      — names EntityA only.
    P1: About EntityA; reveals the link to EntityB.
    P2: About EntityB; contains the final answer.
    gold_ids = [P1, P2]; num_hops = 2.
    """
    a_entity, a_link_phrase, b_entity, b_answer_phrase, final_answer = pool_entry
    qid = f"multi_hop_chain_{idx:02d}"

    para_a = (
        f"{a_entity} is a well-known landmark in the region. "
        f"It is {a_link_phrase}. "
        f"Visitors to {a_entity} often remark on its historical significance."
    )
    para_b = (
        f"{b_entity} is an important administrative and cultural centre. "
        f"Records confirm that it is {b_answer_phrase}. "
        f"{b_entity} has maintained close ties with neighbouring districts for centuries."
    )

    distractors = []
    distractors.append({
        "id": f"{qid}_dist_0",
        "title": f"{a_entity} Tourism",
        "content": (
            f"Tourism at {a_entity} has grown steadily over the past decade. "
            f"The site attracts scholars and casual visitors alike. "
            f"Local guides offer tours covering its architectural and cultural heritage."
        ),
    })
    distractors.append({
        "id": f"{qid}_dist_1",
        "title": f"{b_entity} Infrastructure",
        "content": (
            f"{b_entity} recently completed a major infrastructure upgrade. "
            f"The project included road widening and new public transit links. "
            f"Residents expressed broad satisfaction with the improvements."
        ),
    })
    for di in range(6):
        distractors.append({
            "id": f"{qid}_dist_{di + 2}",
            "title": f"Distractor {qid} {di}",
            "content": _make_distractor_para(rng),
        })

    context = [
        {"id": f"{qid}_a", "title": a_entity,  "content": para_a},
        {"id": f"{qid}_b", "title": b_entity,  "content": para_b},
    ] + distractors
    rng.shuffle(context)

    return {
        "id": qid,
        "question": f"Starting from {a_entity}, what can be found by following the links?",
        "answer": final_answer,
        "task_type": "multi_hop_chain",
        "supporting_facts": [f"{qid}_a", f"{qid}_b"],
        "gold_ids":         [f"{qid}_a", f"{qid}_b"],
        "context": context,
        "required_substrates": [AddressSpaceType.ENTITY, AddressSpaceType.SEMANTIC],
        "num_hops": 2,
    }


def _gen_discovery_extraction(
    rng: random.Random,
    idx: int,
    pool_entry: tuple,
    all_dept_entries: list,
) -> dict:
    """Discovery + Extraction (unchanged from v1)."""
    dept_name, responsibility, budget_m = pool_entry
    qid = f"discovery_extraction_{idx:02d}"

    dept_lines = []
    for d_name, d_resp, _ in all_dept_entries:
        dept_lines.append(f"- {d_name} Department: handles {d_resp}.")
    index_content = (
        "The Municipal Services Directory lists the following departments and their primary responsibilities:\n"
        + "\n".join(dept_lines)
        + "\nEach department publishes an annual budget report available on request."
    )

    dept_para = (
        f"The {dept_name} Department oversees {responsibility} across the municipality. "
        f"According to the latest annual report, the department's total budget stands at ${budget_m} million. "
        f"Funding is allocated primarily to staff salaries, equipment, and operational expenses."
    )

    distractors = []
    other_entries = [e for e in all_dept_entries if e[0] != dept_name]
    if other_entries:
        other_dept, other_resp, other_budget = rng.choice(other_entries)
        distractors.append({
            "id": f"{qid}_dist_0",
            "title": f"{other_dept} Department Budget",
            "content": (
                f"The {other_dept} Department is responsible for {other_resp} in the city. "
                f"Its approved annual budget is ${other_budget} million. "
                f"The department recently modernised several key service delivery workflows."
            ),
        })

    for di in range(7):
        distractors.append({
            "id": f"{qid}_dist_{di + 1}",
            "title": f"Distractor {qid} {di}",
            "content": _make_distractor_para(rng),
        })

    context = [
        {"id": f"{qid}_index", "title": "Municipal Services Directory", "content": index_content},
        {"id": f"{qid}_dept",  "title": f"{dept_name} Department",      "content": dept_para},
    ] + distractors
    rng.shuffle(context)

    return {
        "id": qid,
        "question": (
            f"Which department handles {responsibility}? "
            f"What is their budget?"
        ),
        "answer": f"{dept_name}, ${budget_m} million",
        "task_type": "discovery_extraction",
        "supporting_facts": [f"{qid}_index", f"{qid}_dept"],
        "gold_ids":         [f"{qid}_index", f"{qid}_dept"],
        "context": context,
        "required_substrates": [AddressSpaceType.SEMANTIC, AddressSpaceType.LEXICAL],
        "num_hops": 2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_bridge_isolation(example: dict) -> tuple[bool, str]:
    """
    Return (pass, message).
    For entity_bridge and implicit_bridge:
      the bridge_entity must NOT appear (case-insensitive) in the question text.
    """
    bridge = example.get("bridge_entity", "")
    question = example["question"]
    if not bridge:
        return True, "no bridge_entity key — skipped"
    if bridge.lower() in question.lower():
        return False, f"bridge_entity '{bridge}' found in question: '{question}'"
    return True, f"OK — '{bridge}' not in question"


def _validate_lexical_overlap(example: dict) -> tuple[bool, str]:
    """
    Return (pass, message).
    For low_lexical_overlap tasks:
      Jaccard overlap between question and gold paragraph must be < 0.15.
    """
    question = example["question"]
    gold_id  = example["gold_ids"][0] if example.get("gold_ids") else None
    if gold_id is None:
        return False, "no gold_ids"
    gold_doc = next(
        (d for d in example.get("context", []) if d["id"] == gold_id),
        None,
    )
    if gold_doc is None:
        return False, f"gold doc '{gold_id}' not found in context"
    overlap = _word_overlap_fraction(question, gold_doc["content"])
    passed  = overlap < 0.15
    msg = f"overlap={overlap:.3f} {'< 0.15 OK' if passed else '>= 0.15 FAIL'}"
    return passed, msg


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class HeterogeneousBenchmarkV2:
    """
    Synthetic heterogeneous benchmark v2 — 100 questions across 6 task types.

    Compared with v1:
    - entity_bridge:      question no longer names the bridge entity (city)
    - implicit_bridge:    question no longer names the creator/director
    - low_lexical_overlap: question uses lay language; no drug name in question
    - multi_hop_chain:    simplified to 2-hop chains

    Parameters
    ----------
    seed : int
        Random seed for reproducible generation.  Default 42.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed  = seed
        self.examples: list[dict] = []

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def generate(self) -> list[dict]:
        """
        Generate all 100 examples deterministically from ``self.seed``.

        Returns
        -------
        list[dict]
            Each dict has keys:
            ``id``, ``question``, ``answer``, ``task_type``,
            ``supporting_facts``, ``gold_ids``, ``context``,
            ``required_substrates``, ``num_hops``.
        """
        rng = random.Random(self.seed)
        examples: list[dict] = []

        # 1. Entity Bridge (20)
        for i, entry in enumerate(_ENTITY_BRIDGE_V2_POOL):
            examples.append(_gen_entity_bridge_v2(rng, i, entry))

        # 2. Implicit Bridge (20)
        for i, entry in enumerate(_IMPLICIT_BRIDGE_V2_POOL):
            examples.append(_gen_implicit_bridge_v2(rng, i, entry))

        # 3. Semantic + Computation (20) — unchanged
        for i, entry in enumerate(_SEMANTIC_COMPUTATION_POOL):
            examples.append(_gen_semantic_computation(rng, i, entry))

        # 4. Low Lexical Overlap (20)
        for i, entry in enumerate(_LOW_LEXICAL_V2_POOL):
            examples.append(_gen_low_lexical_overlap_v2(rng, i, entry))

        # 5. Multi-hop Chain (10) — 2-hop
        for i, entry in enumerate(_MULTI_HOP_V2_POOL):
            examples.append(_gen_multi_hop_chain_v2(rng, i, entry))

        # 6. Discovery + Extraction (10) — unchanged
        for i, entry in enumerate(_DISCOVERY_EXTRACTION_POOL):
            examples.append(
                _gen_discovery_extraction(rng, i, entry, _DISCOVERY_EXTRACTION_POOL)
            )

        self.examples = examples
        return examples

    def get_by_type(self, task_type: str) -> list[dict]:
        """Return all examples of a specific task type."""
        if not self.examples:
            self.generate()
        return [ex for ex in self.examples if ex["task_type"] == task_type]

    # ─────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────

    def validate(self) -> dict:
        """
        Run design-rule checks on the generated benchmark.

        Returns
        -------
        dict
            Keys: ``passed``, ``failed``, ``total``, ``details``
        """
        if not self.examples:
            self.generate()

        details = []
        passed_count = 0
        failed_count = 0

        print("\n=== Benchmark V2 Validation ===\n")

        # Entity bridge isolation
        print("[ Entity Bridge — bridge entity isolation ]")
        for ex in self.examples:
            if ex["task_type"] == "entity_bridge":
                ok, msg = _validate_bridge_isolation(ex)
                status = "PASS" if ok else "FAIL"
                if ok:
                    passed_count += 1
                else:
                    failed_count += 1
                details.append({"id": ex["id"], "check": "bridge_isolation", "status": status, "msg": msg})
                print(f"  [{status}] {ex['id']}: {msg}")

        # Implicit bridge isolation
        print("\n[ Implicit Bridge — bridge entity isolation ]")
        for ex in self.examples:
            if ex["task_type"] == "implicit_bridge":
                ok, msg = _validate_bridge_isolation(ex)
                status = "PASS" if ok else "FAIL"
                if ok:
                    passed_count += 1
                else:
                    failed_count += 1
                details.append({"id": ex["id"], "check": "bridge_isolation", "status": status, "msg": msg})
                print(f"  [{status}] {ex['id']}: {msg}")

        # Low lexical overlap
        print("\n[ Low Lexical Overlap — word overlap < 15% ]")
        for ex in self.examples:
            if ex["task_type"] == "low_lexical_overlap":
                ok, msg = _validate_lexical_overlap(ex)
                status = "PASS" if ok else "FAIL"
                if ok:
                    passed_count += 1
                else:
                    failed_count += 1
                details.append({"id": ex["id"], "check": "lexical_overlap", "status": status, "msg": msg})
                print(f"  [{status}] {ex['id']}: {msg}")

        total = passed_count + failed_count
        print(f"\nValidation summary: {passed_count}/{total} checks passed, {failed_count} failed.\n")

        return {
            "passed":  passed_count,
            "failed":  failed_count,
            "total":   total,
            "details": details,
        }
