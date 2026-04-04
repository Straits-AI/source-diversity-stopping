"""
Heterogeneous Benchmark for AEA Framework.

100 questions across 6 task types, each designed so that no single address
space can solve it alone.  All generation is deterministic from a fixed seed.

Task types
----------
1. entity_bridge      (20 q)  — person → country → capital (entity hop required)
2. implicit_bridge    (20 q)  — movie → director (implicit, not in question) → award
3. semantic_computation (20 q)  — compare two companies' revenues (semantic + arithmetic)
4. low_lexical_overlap (20 q)  — synonyms/paraphrases defeat BM25; semantic wins
5. multi_hop_chain    (10 q)  — three-hop A → B → C → answer
6. discovery_extraction (10 q)  — find department from index, then extract budget

For every question the corpus has:
  * 2–3 gold supporting docs
  * 7–8 distractor docs (plausible, share entities/domain, but NOT the answer)
  * Total: 10 docs per question
"""

from __future__ import annotations

import random
from typing import Optional

from experiments.aea.types import AddressSpaceType


# ─────────────────────────────────────────────────────────────────────────────
# Template vocabulary pools
# ─────────────────────────────────────────────────────────────────────────────

# (person_first, person_last, birth_country, capital_of_that_country)
_ENTITY_BRIDGE_POOL = [
    ("Alaric", "Voss",      "Nordavia",      "Halstein"),
    ("Brenna", "Solberg",   "Valdoria",      "Keldmoor"),
    ("Caspian", "Wren",     "Threskia",      "Arvelund"),
    ("Delia",  "Holt",      "Murenia",       "Cadrith"),
    ("Erwin",  "Farquhar",  "Lestonia",      "Brannport"),
    ("Fenella","Crowe",     "Oshavia",       "Dramholt"),
    ("Gideon", "Marsh",     "Peloria",       "Silvast"),
    ("Hana",   "Trent",     "Veluris",       "Ondenmere"),
    ("Idris",  "Kemble",    "Zarathon",      "Forsheim"),
    ("Jora",   "Stein",     "Duvenmark",     "Elbronn"),
    ("Kaspar", "Bryne",     "Torelia",       "Stormgard"),
    ("Lyra",   "Fenn",      "Aquinda",       "Portmere"),
    ("Magnus", "Thall",     "Clevoria",      "Sundvale"),
    ("Nessa",  "Greer",     "Dravonia",      "Ironholt"),
    ("Oswin",  "Platt",     "Felvara",       "Crestmoore"),
    ("Petra",  "Dorn",      "Gondurex",      "Whitecliff"),
    ("Quinn",  "Aldric",    "Halvenia",      "Stormbeck"),
    ("Rook",   "Sable",     "Indreth",       "Marborough"),
    ("Soren",  "Vale",      "Juntera",       "Ashford"),
    ("Talia",  "Brand",     "Kolvara",       "Emberton"),
]

# (movie_title, director_name, award_name)
_IMPLICIT_BRIDGE_POOL = [
    ("Shadows of the Deep",      "Cormac Finley",      "Golden Lens Award"),
    ("The Amber Horizon",        "Vera Stahl",          "Silver Reel Prize"),
    ("Echoes of Nowhere",        "Desmond Kray",        "Palme d'Or equivalent (fictional)"),
    ("The Last Cartographer",    "Ingrid Holst",        "Crystal Bear Award"),
    ("Iron Meridian",            "Tomas Quellar",       "Sundance Grand Jury Prize"),
    ("Beneath Still Waters",     "Adela Vance",         "Berlin Golden Eagle"),
    ("The Quiet Insurgent",      "Brennan Wolfe",       "Critics' Lantern Award"),
    ("Starfall Protocol",        "Linh Nguyen",         "Academy Horizon Award"),
    ("The Cartographer's Ghost", "Remy Chalcott",       "Toronto Peoples Choice"),
    ("Fractures in Blue",        "Saoirse Kern",        "Venice Golden Lion Award"),
    ("After the Vanishing",      "Hakeem Osei",         "BAFTA Outstanding Debut"),
    ("The Obsidian Shore",       "Nadine Blum",         "FIPRESCI Discovery Prize"),
    ("Corridors of Amber",       "Pascal Dreier",       "Locarno Golden Leopard"),
    ("Hollow Ground",            "Faye Moreau",         "Rotterdam Tiger Award"),
    ("The Second Silence",       "Matias Calderon",     "Tribeca Jury Award"),
    ("Woven in Fire",            "Kezia Strand",        "Sarajevo Best Director"),
    ("The Porcelain Architect",  "Omar Hadidi",         "Warsaw Grand Prix"),
    ("Last Light Over Moren",    "Astrid Bjork",        "Austin Narrative Feature Award"),
    ("The Cartesian Doubt",      "Evander Mills",       "Fantasia Jury Prize"),
    ("Glass Roots",              "Priya Rajan",         "Athens International Award"),
]

# (company_a, revenue_a_m, company_b, revenue_b_m, higher_company)
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

# (question_paraphrase, medication_synonym, condition_synonym,
#  drug_term, disease_term)  — low lexical overlap design
_LOW_LEXICAL_POOL = [
    ("What medical condition does the medication Zolparin treat?",
     "Zolparin",    "chronic joint inflammation",
     "Zolparin",    "rheumatoid arthritis",
     "The drug Zolparin is prescribed for patients suffering from rheumatoid arthritis, a chronic autoimmune disease. "
     "Clinical trials showed significant reduction in joint swelling. "
     "Patients on Zolparin reported improved mobility within eight weeks."),
    ("Which illness is the pharmaceutical Veranex used to remedy?",
     "Veranex",     "elevated blood pressure",
     "Veranex",     "hypertension",
     "Veranex is administered to individuals diagnosed with hypertension, colloquially known as high blood pressure. "
     "The compound reduces arterial resistance and stabilises heart rate. "
     "Physicians recommend Veranex as a first-line treatment for stage-two blood pressure disorders."),
    ("What ailment is the drug Lumivex intended for?",
     "Lumivex",     "depressive disorder",
     "Lumivex",     "major depression",
     "The pharmaceutical Lumivex targets major depression by modulating serotonin reuptake inhibitors. "
     "Studies confirm its efficacy in patients with severe mood disorders. "
     "Lumivex is classified as a selective reuptake compound in clinical literature."),
    ("For what disease is the remedy Corvacin utilised?",
     "Corvacin",    "respiratory infection",
     "Corvacin",    "bacterial pneumonia",
     "Corvacin is employed against bacterial pneumonia, a severe lung infection caused by Streptococcus strains. "
     "The antibiotic interrupts cell wall synthesis in pathogenic bacteria. "
     "Corvacin is prescribed for patients hospitalised with acute respiratory failure."),
    ("What health problem does the treatment Durafen address?",
     "Durafen",     "inflammatory bowel disease",
     "Durafen",     "Crohn's disease",
     "Durafen is a biologic agent targeting Crohn's disease, a chronic gastrointestinal illness. "
     "It suppresses tumour necrosis factor pathways responsible for bowel inflammation. "
     "Gastroenterologists prescribe Durafen when conventional therapies have failed."),
    ("What disorder is the compound Nexalin formulated to manage?",
     "Nexalin",     "seizure disorder",
     "Nexalin",     "epilepsy",
     "Nexalin was developed specifically for epilepsy management in adult patients. "
     "The drug stabilises neuronal membranes and reduces cortical excitability. "
     "Long-term studies show Nexalin reduces seizure frequency by up to sixty percent."),
    ("What sickness is the pharmaceutical Provatex prescribed for?",
     "Provatex",    "immune system overreaction",
     "Provatex",    "autoimmune hepatitis",
     "Provatex is prescribed for patients presenting with autoimmune hepatitis, a liver condition in which the immune system attacks hepatic cells. "
     "The compound suppresses T-cell proliferation effectively. "
     "Treatment with Provatex typically lasts eighteen months."),
    ("Which pathology is the remedy Solvex indicated for?",
     "Solvex",      "muscle wasting condition",
     "Solvex",      "muscular dystrophy",
     "Solvex was developed to slow the progression of muscular dystrophy in paediatric patients. "
     "It promotes dystrophin synthesis through exon skipping. "
     "Clinical outcomes with Solvex show preserved muscle mass at eighteen-month follow-up."),
    ("What condition is the medicine Talverin meant to cure?",
     "Talviren",    "elevated blood lipids",
     "Talviren",    "hypercholesterolaemia",
     "Talviren is a statin-class drug used in hypercholesterolaemia management. "
     "It inhibits HMG-CoA reductase and substantially lowers LDL cholesterol. "
     "Cardiologists rely on Talviren to reduce atherosclerotic plaque formation."),
    ("For which disorder is the therapeutic Venodrix prescribed?",
     "Venodrix",    "bone density loss",
     "Venodrix",    "osteoporosis",
     "Venodrix is a bisphosphonate compound used to treat osteoporosis in postmenopausal women. "
     "It inhibits osteoclast activity and reduces fracture risk. "
     "Patients take Venodrix weekly to maintain adequate bone mineral density."),
    ("What disease is the agent Caldrex formulated to combat?",
     "Caldrex",     "excessive blood glucose",
     "Caldrex",     "type 2 diabetes",
     "Caldrex improves insulin sensitivity in patients with type 2 diabetes. "
     "The drug reduces hepatic glucose output and promotes peripheral uptake. "
     "Endocrinologists recommend Caldrex as an adjunct to lifestyle modification."),
    ("Which affliction is the substance Morvane targeted at?",
     "Morvane",     "fungal skin condition",
     "Morvane",     "tinea corporis",
     "Morvane is an antifungal cream prescribed for tinea corporis, a ringworm infection affecting the skin. "
     "It disrupts ergosterol synthesis in fungal cell membranes. "
     "Most patients see resolution within four weeks of topical Morvane therapy."),
    ("What health issue is the pharmaceutical Bevalux used to resolve?",
     "Bevalux",     "abnormal heart rhythm",
     "Bevalux",     "atrial fibrillation",
     "Bevalux is a class-three antiarrhythmic agent indicated for atrial fibrillation management. "
     "It prolongs the cardiac action potential to restore sinus rhythm. "
     "Cardiologists prescribe Bevalux for persistent AF unresponsive to cardioversion."),
    ("Which malady is the drug Zeltramine designed to handle?",
     "Zeltramine",  "allergic nasal symptoms",
     "Zeltramine",  "allergic rhinitis",
     "Zeltramine is an antihistamine compound used in the treatment of allergic rhinitis and hay fever. "
     "It blocks H1 receptors and reduces nasal inflammation and sneezing. "
     "Unlike older antihistamines, Zeltramine is non-sedating at therapeutic doses."),
    ("What malfunction does the treatment Corilene address?",
     "Corilene",    "thyroid underactivity",
     "Corilene",    "hypothyroidism",
     "Corilene is a synthetic levothyroxine formulation for treating hypothyroidism. "
     "It replaces the deficient thyroid hormone produced by the gland. "
     "Patients take Corilene once daily to normalise metabolic function."),
    ("What illness is the compound Fibrexol prescribed to treat?",
     "Fibrexol",    "lung scarring disorder",
     "Fibrexol",    "pulmonary fibrosis",
     "Fibrexol is an anti-fibrotic agent used in idiopathic pulmonary fibrosis. "
     "It slows the deposition of collagen in the lung parenchyma. "
     "Pulmonologists prescribe Fibrexol to delay respiratory decline."),
    ("Which disorder does the medicine Rantivex remedy?",
     "Rantivex",    "stomach acid overproduction",
     "Rantivex",    "peptic ulcer disease",
     "Rantivex is a proton pump inhibitor prescribed for peptic ulcer disease and gastric acid hypersecretion. "
     "It irreversibly blocks the gastric acid pump. "
     "Gastroenterologists use Rantivex for both healing and prophylaxis of peptic ulcers."),
    ("What condition is the pharmaceutical Deltraze indicated for?",
     "Deltraze",    "persistent low mood",
     "Deltraze",    "clinical depression",
     "Deltraze is a dual-action antidepressant for clinical depression and generalised anxiety. "
     "It modulates norepinephrine and serotonin reuptake simultaneously. "
     "Psychiatrists use Deltraze when patients fail to respond to SSRI monotherapy."),
    ("For which disease is the agent Quarvolide recommended?",
     "Quarvolide",  "excessive daytime sleepiness",
     "Quarvolide",  "narcolepsy",
     "Quarvolide is a wakefulness-promoting agent indicated for narcolepsy, a neurological sleep disorder. "
     "It enhances dopaminergic and noradrenergic signalling in the brain. "
     "Neurologists prescribe Quarvolide to reduce cataplexy and sudden sleep attacks."),
    ("What ailment does the remedy Norplexin target?",
     "Norplexin",   "nerve pain syndrome",
     "Norplexin",   "neuropathic pain",
     "Norplexin is an anticonvulsant medication repurposed for neuropathic pain management. "
     "It modulates voltage-gated calcium channels to reduce ectopic neural firing. "
     "Pain specialists prescribe Norplexin for diabetic neuropathy and post-herpetic neuralgia."),
]

# (hop-A entity, hop-A content summary, hop-B entity, hop-B content,
#  hop-C entity, hop-C content, final answer)
_MULTI_HOP_POOL = [
    ("Thornwick Station", "located in the borough of Halgrave", "Halgrave", "administrated by the Halgrave Council, chaired by Director Voss", "Director Voss", "full name Everard Voss", "Everard Voss"),
    ("Blue Pine Lake",    "fed by the Kallos River",           "Kallos River","flows through the town of Mendrecht", "Mendrecht",    "officially incorporated in 1847",              "1847"),
    ("Alveron Museum",    "founded by Ida Solten",             "Ida Solten",  "born in the city of Cramford",          "Cramford",     "capital of Westmere Province",                "Westmere Province"),
    ("Corvax Tower",      "designed by architect Bryce Naldor","Bryce Naldor","trained at the Halden School of Design", "Halden School","located on Pryswick Avenue",                  "Pryswick Avenue"),
    ("Mendover Bridge",   "commissioned by the Stelvane Company","Stelvane Company","headquartered in the port of Orvath","Orvath","chief export is processed amber",         "processed amber"),
    ("Clearwater Accord", "signed in the city of Delvorne",   "Delvorne",    "situated on the eastern coast of Valdris","Valdris",    "national bird is the grey heron",             "grey heron"),
    ("Pellar Institute",  "headed by Professor Gale Astor",   "Gale Astor",  "holds a doctorate from Crestmoor University","Crestmoor University","ranked 14th in Nordavia",         "Nordavia"),
    ("Sandreth Canal",    "built by the Ironside Company",    "Ironside Company","registered under Delvorne trade law", "Delvorne",    "currency is the Silver Mark",                 "Silver Mark"),
    ("Vorde Festival",    "held annually in the town of Brynwick","Brynwick", "seat of the Brynwick Heritage Council",  "Brynwick Heritage Council","founded in 1903",              "1903"),
    ("Glassholm Lighthouse","operated by Maritime Board Severn","Maritime Board Severn","reports to Harbour Master Elara Keld","Elara Keld","holds the rank of Commodore",            "Commodore"),
]

# (dept_name, responsibility_keyword, budget_m)
_DISCOVERY_EXTRACTION_POOL = [
    ("Infrastructure",  "road maintenance",       42),
    ("Public Health",   "disease prevention",     78),
    ("Education",       "school curriculum",      115),
    ("Emergency Services","first responder coordination", 63),
    ("Urban Planning",  "zoning and land use",    31),
    ("Environmental Affairs","waste management",  55),
    ("Housing Authority","affordable housing",    89),
    ("Transport",       "public transit systems", 97),
    ("Social Services", "welfare support",        48),
    ("Economic Development","business licensing", 37),
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


def _make_distractor_para(rng: random.Random, domain_hint: str = "") -> str:
    """Generate a plausible distractor paragraph unrelated to the answer."""
    person = rng.choice(_DISTRACTOR_PEOPLE)
    place  = rng.choice(_DISTRACTOR_PLACES)
    company = rng.choice(_DISTRACTOR_COMPANIES)
    year   = rng.randint(1890, 2010)
    topics = [
        f"{person} was born in {place} and later moved to {company}'s headquarters in {year}. "
        f"During that period, {place} underwent significant urban renewal. "
        f"The city's archives record {person} as a prominent figure in local history.",

        f"The {company} was incorporated in {year} in the district of {place}. "
        f"Its founding charter was drafted by {person}, a local administrator. "
        f"The company expanded rapidly across the region within five years.",

        f"In {year}, {place} hosted the International Trade Summit. "
        f"{company} was among the exhibitors, represented by delegate {person}. "
        f"The summit concluded with several bilateral agreements on tariffs.",

        f"{person} published a historical account of {place} in {year}. "
        f"The monograph detailed the city's industrial rise under {company}'s patronage. "
        f"Academic reviewers praised the work for its archival depth.",

        f"The cultural heritage of {place} is preserved by the {place} Heritage Foundation. "
        f"{person} served as the foundation's chairperson for over a decade. "
        f"Annual grants from {company} sustain the foundation's restoration projects.",
    ]
    return rng.choice(topics)


# ─────────────────────────────────────────────────────────────────────────────
# Individual task-type generators
# ─────────────────────────────────────────────────────────────────────────────

def _gen_entity_bridge(rng: random.Random, idx: int, pool_entry: tuple) -> dict:
    first, last, country, capital = pool_entry
    full_name = f"{first} {last}"
    qid = f"entity_bridge_{idx:02d}"

    # Gold doc 1: about the person (mentions country but NOT capital)
    person_para = (
        f"{full_name} is a renowned scholar who was born in {country}. "
        f"Growing up in {country}, {first} developed a passion for linguistics. "
        f"{full_name} later relocated to pursue doctoral studies abroad."
    )

    # Gold doc 2: about the country (mentions capital)
    country_para = (
        f"{country} is a sovereign nation in the northern continent. "
        f"Its capital city is {capital}, a cultural and administrative hub. "
        f"{country} has a population of approximately three million people."
    )

    # Distractors (8 total): share person name or country name but no capital link
    distractors = []
    # Distractor mentioning the person but not the country's capital
    distractors.append({
        "id": f"{qid}_dist_0",
        "title": f"{full_name} Academic Career",
        "content": (
            f"{full_name} completed a fellowship at the Halvenia Institute of Higher Studies. "
            f"His research on comparative syntax earned international recognition. "
            f"He later joined the editorial board of the Journal of Theoretical Linguistics."
        ),
    })
    # Distractor mentioning country but not capital
    distractors.append({
        "id": f"{qid}_dist_1",
        "title": f"{country} Economy",
        "content": (
            f"{country}'s economy relies heavily on maritime trade and light manufacturing. "
            f"The annual GDP growth rate averaged 3.2 percent over the past decade. "
            f"Foreign direct investment into {country} has risen steadily since market reforms."
        ),
    })
    for di in range(6):
        distractors.append({
            "id": f"{qid}_dist_{di+2}",
            "title": f"Distractor {qid} {di}",
            "content": _make_distractor_para(rng),
        })

    context = [
        {"id": f"{qid}_person", "title": full_name, "content": person_para},
        {"id": f"{qid}_country", "title": country, "content": country_para},
    ] + distractors

    rng.shuffle(context)

    return {
        "id": qid,
        "question": f"What is the capital city of the country where {full_name} was born?",
        "answer": capital,
        "task_type": "entity_bridge",
        "supporting_facts": [f"{qid}_person", f"{qid}_country"],
        "gold_ids":         [f"{qid}_person", f"{qid}_country"],
        "context": context,
        "required_substrates": [AddressSpaceType.SEMANTIC, AddressSpaceType.ENTITY],
        "num_hops": 2,
    }


def _gen_implicit_bridge(rng: random.Random, idx: int, pool_entry: tuple) -> dict:
    movie, director, award = pool_entry
    qid = f"implicit_bridge_{idx:02d}"

    # Gold doc 1: movie paragraph (mentions director name, NOT the award)
    movie_para = (
        f"\"{movie}\" is a critically acclaimed film released in the early 2000s. "
        f"The film was directed by {director}, known for his distinctive visual style. "
        f"The production was shot on location across three continents."
    )

    # Gold doc 2: director paragraph (mentions award, NOT the movie title)
    director_para = (
        f"{director} is an internationally recognised filmmaker. "
        f"His contributions to world cinema earned him the {award} in 2009. "
        f"{director} is known for his meticulous approach to cinematography."
    )

    distractors = []
    distractors.append({
        "id": f"{qid}_dist_0",
        "title": f"{movie} Box Office",
        "content": (
            f"\"{movie}\" grossed over $85 million worldwide during its theatrical run. "
            f"The film opened to strong reviews in North America and Europe. "
            f"Home video rights were sold to a major streaming platform."
        ),
    })
    distractors.append({
        "id": f"{qid}_dist_1",
        "title": f"Film Awards Ceremony",
        "content": (
            f"The annual cinema awards ceremony celebrated achievements across multiple genres. "
            f"The Best Director category featured nominees from six countries. "
            f"A standing ovation greeted the announcement of the winner."
        ),
    })
    for di in range(6):
        distractors.append({
            "id": f"{qid}_dist_{di+2}",
            "title": f"Distractor {qid} {di}",
            "content": _make_distractor_para(rng),
        })

    context = [
        {"id": f"{qid}_movie",    "title": movie,    "content": movie_para},
        {"id": f"{qid}_director", "title": director, "content": director_para},
    ] + distractors

    rng.shuffle(context)

    return {
        "id": qid,
        "question": f"What award did the director of \"{movie}\" win?",
        "answer": award,
        "task_type": "implicit_bridge",
        "supporting_facts": [f"{qid}_movie", f"{qid}_director"],
        "gold_ids":         [f"{qid}_movie", f"{qid}_director"],
        "context": context,
        "required_substrates": [AddressSpaceType.SEMANTIC, AddressSpaceType.ENTITY],
        "num_hops": 2,
    }


def _gen_semantic_computation(rng: random.Random, idx: int, pool_entry: tuple) -> dict:
    comp_a, rev_a, comp_b, rev_b, higher = pool_entry
    qid = f"semantic_computation_{idx:02d}"

    # Gold doc 1: company A
    para_a = (
        f"{comp_a} is a diversified industrial group headquartered in Valden City. "
        f"In its most recent fiscal year, the company reported annual revenue of ${rev_a} million. "
        f"The firm operates across logistics, manufacturing, and consumer goods segments."
    )

    # Gold doc 2: company B
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
            "id": f"{qid}_dist_{di+2}",
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


def _gen_low_lexical_overlap(rng: random.Random, idx: int, pool_entry: tuple) -> dict:
    question, med_q, cond_q, drug_doc, disease_doc, gold_para = pool_entry
    qid = f"low_lexical_overlap_{idx:02d}"

    distractors = []
    # One distractor that mentions the drug but not the disease
    distractors.append({
        "id": f"{qid}_dist_0",
        "title": f"{drug_doc} Clinical Trial",
        "content": (
            f"A phase-III clinical trial of {drug_doc} enrolled 850 participants across twelve centres. "
            f"The primary endpoint was reduction in symptom severity at twelve weeks. "
            f"Adverse effects were mild and consistent with the drug class profile."
        ),
    })
    # One distractor about a different drug for the same disease area (domain-adjacent but wrong drug)
    alt_drug = "Placevex"
    distractors.append({
        "id": f"{qid}_dist_1",
        "title": f"{alt_drug} Overview",
        "content": (
            f"{alt_drug} is a well-established pharmaceutical used in chronic disease management. "
            f"It is administered orally and reaches peak plasma concentration within two hours. "
            f"Long-term safety data for {alt_drug} show a favourable tolerability profile."
        ),
    })
    for di in range(6):
        distractors.append({
            "id": f"{qid}_dist_{di+2}",
            "title": f"Distractor {qid} {di}",
            "content": _make_distractor_para(rng),
        })

    gold_doc_id = f"{qid}_gold"
    context = [
        {"id": gold_doc_id, "title": f"{drug_doc} Prescribing Information", "content": gold_para},
    ] + distractors

    rng.shuffle(context)

    return {
        "id": qid,
        "question": question,
        "answer": disease_doc,
        "task_type": "low_lexical_overlap",
        "supporting_facts": [gold_doc_id],
        "gold_ids":         [gold_doc_id],
        "context": context,
        "required_substrates": [AddressSpaceType.SEMANTIC],
        "num_hops": 1,
    }


def _gen_multi_hop_chain(rng: random.Random, idx: int, pool_entry: tuple) -> dict:
    (a_entity, a_link_phrase, b_entity, b_link_phrase,
     c_entity, c_answer_phrase, final_answer) = pool_entry
    qid = f"multi_hop_chain_{idx:02d}"

    # Gold doc A: mentions entity A and contains the link to entity B
    para_a = (
        f"{a_entity} is a well-known landmark in the region. "
        f"It is {a_link_phrase}. "
        f"Visitors to {a_entity} often remark on its historical significance."
    )

    # Gold doc B: mentions entity B and contains the link to entity C
    para_b = (
        f"{b_entity} is an important administrative and cultural centre. "
        f"The area is {b_link_phrase}. "
        f"{b_entity} has maintained close ties with neighbouring districts for centuries."
    )

    # Gold doc C: contains the final answer
    para_c = (
        f"{c_entity} is a notable institution with deep historical roots. "
        f"Records indicate that its {c_answer_phrase}. "
        f"It continues to play a central role in regional affairs."
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
            "id": f"{qid}_dist_{di+2}",
            "title": f"Distractor {qid} {di}",
            "content": _make_distractor_para(rng),
        })

    context = [
        {"id": f"{qid}_a", "title": a_entity, "content": para_a},
        {"id": f"{qid}_b", "title": b_entity, "content": para_b},
        {"id": f"{qid}_c", "title": c_entity, "content": para_c},
    ] + distractors

    rng.shuffle(context)

    return {
        "id": qid,
        "question": f"Following the chain of links from {a_entity}, what is the final answer?",
        "answer": final_answer,
        "task_type": "multi_hop_chain",
        "supporting_facts": [f"{qid}_a", f"{qid}_b", f"{qid}_c"],
        "gold_ids":         [f"{qid}_a", f"{qid}_b", f"{qid}_c"],
        "context": context,
        "required_substrates": [
            AddressSpaceType.ENTITY,
            AddressSpaceType.SEMANTIC,
        ],
        "num_hops": 3,
    }


def _gen_discovery_extraction(rng: random.Random, idx: int, pool_entry: tuple,
                               all_dept_entries: list) -> dict:
    dept_name, responsibility, budget_m = pool_entry
    qid = f"discovery_extraction_{idx:02d}"

    # Build the index paragraph listing ALL departments and their responsibilities
    # This forces discovery first: you must find the right dept from the index,
    # then look up the budget in the dept-specific paragraph.
    dept_lines = []
    for d_name, d_resp, _ in all_dept_entries:
        dept_lines.append(f"- {d_name} Department: handles {d_resp}.")
    index_content = (
        "The Municipal Services Directory lists the following departments and their primary responsibilities:\n"
        + "\n".join(dept_lines)
        + "\nEach department publishes an annual budget report available on request."
    )

    # Gold doc 2: specific department budget page
    dept_para = (
        f"The {dept_name} Department oversees {responsibility} across the municipality. "
        f"According to the latest annual report, the department's total budget stands at ${budget_m} million. "
        f"Funding is allocated primarily to staff salaries, equipment, and operational expenses."
    )

    distractors = []
    # Pick a different dept as a distractor to make it confusing
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
            "id": f"{qid}_dist_{di+1}",
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
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class HeterogeneousBenchmark:
    """
    Synthetic heterogeneous benchmark with 100 questions across 6 task types.

    Parameters
    ----------
    seed : int
        Random seed for reproducible generation.  Default 42.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
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
        for i, entry in enumerate(_ENTITY_BRIDGE_POOL):
            examples.append(_gen_entity_bridge(rng, i, entry))

        # 2. Implicit Bridge (20)
        for i, entry in enumerate(_IMPLICIT_BRIDGE_POOL):
            examples.append(_gen_implicit_bridge(rng, i, entry))

        # 3. Semantic + Computation (20)
        for i, entry in enumerate(_SEMANTIC_COMPUTATION_POOL):
            examples.append(_gen_semantic_computation(rng, i, entry))

        # 4. Low Lexical Overlap (20)
        for i, entry in enumerate(_LOW_LEXICAL_POOL):
            examples.append(_gen_low_lexical_overlap(rng, i, entry))

        # 5. Multi-hop Chain (10)
        for i, entry in enumerate(_MULTI_HOP_POOL):
            examples.append(_gen_multi_hop_chain(rng, i, entry))

        # 6. Discovery + Extraction (10)
        for i, entry in enumerate(_DISCOVERY_EXTRACTION_POOL):
            examples.append(
                _gen_discovery_extraction(rng, i, entry, _DISCOVERY_EXTRACTION_POOL)
            )

        self.examples = examples
        return examples

    def get_by_type(self, task_type: str) -> list[dict]:
        """
        Return all examples of a specific task type.

        Parameters
        ----------
        task_type : str
            One of: ``entity_bridge``, ``implicit_bridge``,
            ``semantic_computation``, ``low_lexical_overlap``,
            ``multi_hop_chain``, ``discovery_extraction``.

        Returns
        -------
        list[dict]
        """
        if not self.examples:
            self.generate()
        return [ex for ex in self.examples if ex["task_type"] == task_type]
