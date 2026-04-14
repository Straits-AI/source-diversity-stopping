"""
FEVER Fact Verification Benchmark — AEA Framework Evaluation.

Tests whether source-diversity stopping generalises beyond QA tasks.
FEVER is a fact-checking benchmark where the task is to classify claims as
SUPPORTED, REFUTED, or NOT ENOUGH INFO based on retrieved Wikipedia evidence.

Key question: Does pi_heuristic (source-diversity stopping) beat pi_ensemble
on a NON-QA task?

Design
------
- 200 FEVER-style examples: 100 SUPPORTED + 100 REFUTED claims
- Each example: 10 context paragraphs (1-2 gold evidence + 8-9 distractors)
- Evaluation: SupportRecall (did we find the gold evidence?), AvgOps, U@B
- Stopping is retrieval-only — no LLM answer generation needed
- Seed: 42

Usage
-----
    python experiments/run_fever.py

Results saved to experiments/results/fever.json.
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
from scipy import stats as scipy_stats

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ───────────────────────────────────────────────────────────────
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.evaluation.metrics import (
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

# ── Constants ─────────────────────────────────────────────────────────────────
N_EXAMPLES = 200          # 100 SUPPORTED + 100 REFUTED
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "fever.json"

# Max ops for normalised cost
_MAX_OPS = 18


# ─────────────────────────────────────────────────────────────────────────────
# FEVER-style synthetic dataset
#
# 100 SUPPORTED claims: evidence clearly supports the claim
# 100 REFUTED claims:   evidence clearly contradicts the claim
#
# Each example has 10 paragraphs: 1-2 gold evidence + 8-9 distractors
# ─────────────────────────────────────────────────────────────────────────────

# SUPPORTED claims pool: (claim, evidence_title, evidence_text, distractor_entity)
_SUPPORTED_POOL = [
    ("Barack Obama was born in Hawaii.",
     "Barack Obama",
     "Barack Obama was born on August 4, 1961, in Honolulu, Hawaii. He served as the 44th President of the United States from 2009 to 2017.",
     "Hawaii"),
    ("The Eiffel Tower is located in Paris.",
     "Eiffel Tower",
     "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed in 1889 as the entrance arch for the 1889 World's Fair.",
     "Paris"),
    ("Albert Einstein developed the theory of relativity.",
     "Albert Einstein",
     "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics. His 1905 paper on special relativity introduced the famous equation E=mc².",
     "Relativity"),
    ("The Great Wall of China was built to protect against invasions.",
     "Great Wall of China",
     "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states and Imperial China to protect against invasions and raids.",
     "China"),
    ("William Shakespeare was born in Stratford-upon-Avon.",
     "William Shakespeare",
     "William Shakespeare was an English playwright, poet, and actor. He was born and raised in Stratford-upon-Avon, Warwickshire, England.",
     "Stratford-upon-Avon"),
    ("The Amazon River is the largest river by discharge volume.",
     "Amazon River",
     "The Amazon River in South America is the largest river by discharge volume of water in the world. It flows through Brazil, Peru, and Colombia.",
     "Amazon"),
    ("Isaac Newton formulated the laws of motion.",
     "Isaac Newton",
     "Isaac Newton formulated the three laws of motion, which laid the foundation for classical mechanics. His seminal work 'Principia Mathematica' was published in 1687.",
     "Newton"),
    ("The Moon is Earth's only natural satellite.",
     "Moon",
     "The Moon is Earth's only natural satellite. It orbits Earth at an average distance of 384,400 km and is the fifth-largest satellite in the Solar System.",
     "Earth"),
    ("Leonardo da Vinci painted the Mona Lisa.",
     "Mona Lisa",
     "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. It is considered an archetypal masterpiece of the Italian Renaissance.",
     "Leonardo"),
    ("The speed of light in a vacuum is approximately 299,792 kilometers per second.",
     "Speed of light",
     "The speed of light in vacuum, commonly denoted c, is a universal physical constant equal to exactly 299,792,458 metres per second.",
     "Light"),
    ("Marie Curie was the first woman to win a Nobel Prize.",
     "Marie Curie",
     "Marie Curie was the first woman to win a Nobel Prize, and the only person to win Nobel Prizes in two different sciences (Physics in 1903 and Chemistry in 1911).",
     "Nobel Prize"),
    ("The Berlin Wall fell in 1989.",
     "Berlin Wall",
     "The Berlin Wall was a guarded concrete barrier that physically and ideologically divided Berlin from 1961 to 1989. It fell on November 9, 1989.",
     "Berlin"),
    ("Charles Darwin proposed the theory of evolution by natural selection.",
     "Charles Darwin",
     "Charles Darwin proposed the theory of evolution by means of natural selection in his 1859 work 'On the Origin of Species'.",
     "Evolution"),
    ("The human body has 206 bones.",
     "Human skeleton",
     "The adult human skeleton consists of 206 bones. It provides structure, protects organs, anchors muscles, and stores calcium.",
     "Skeleton"),
    ("The Sahara is the world's largest hot desert.",
     "Sahara Desert",
     "The Sahara is the world's largest hot desert, covering much of North Africa. It covers an area of about 9.2 million square kilometres.",
     "Desert"),
    ("Mount Everest is the highest mountain on Earth.",
     "Mount Everest",
     "Mount Everest, known in Nepali as Sagarmatha and in Tibetan as Chomolungma, is Earth's highest mountain above sea level, located in the Himalayas at 8,848.86 m.",
     "Himalayas"),
    ("The French Revolution began in 1789.",
     "French Revolution",
     "The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789 and ended with the formation of the French Consulate in November 1799.",
     "France"),
    ("DNA has a double helix structure.",
     "DNA",
     "DNA (deoxyribonucleic acid) has a double helix structure, first described by James Watson and Francis Crick in 1953, based on X-ray crystallography data from Rosalind Franklin.",
     "Genetics"),
    ("The Titanic sank after hitting an iceberg.",
     "RMS Titanic",
     "The RMS Titanic sank in the North Atlantic Ocean on April 15, 1912, after striking an iceberg during her maiden voyage from Southampton to New York City.",
     "Iceberg"),
    ("The capital of Australia is Canberra.",
     "Canberra",
     "Canberra is the capital city of Australia. It is located in the Australian Capital Territory and is the largest inland city in Australia.",
     "Australia"),
    ("Penicillin was discovered by Alexander Fleming.",
     "Penicillin",
     "Penicillin was discovered by Alexander Fleming in 1928. He observed that a Penicillium mould had contaminated a bacterial culture and was killing the bacteria around it.",
     "Fleming"),
    ("The Pacific Ocean is the largest ocean on Earth.",
     "Pacific Ocean",
     "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions. It covers an area of about 165.25 million km², more than twice the area of any other ocean.",
     "Ocean"),
    ("Galileo Galilei supported the heliocentric model of the solar system.",
     "Galileo Galilei",
     "Galileo Galilei was an Italian astronomer who supported the heliocentric model proposed by Copernicus, in which the Earth orbits the Sun. He was tried by the Inquisition for this view.",
     "Heliocentrism"),
    ("The Roman Colosseum was used for gladiatorial contests.",
     "Colosseum",
     "The Colosseum is an oval amphitheatre in the centre of Rome. It was used for gladiatorial contests and public spectacles such as animal hunts and executions.",
     "Rome"),
    ("Beethoven composed nine symphonies.",
     "Ludwig van Beethoven",
     "Ludwig van Beethoven was a German composer who composed nine symphonies, considered among the greatest works in the orchestral repertoire.",
     "Symphony"),
    ("The Wright Brothers made the first successful powered airplane flight.",
     "Wright Brothers",
     "The Wright Brothers, Orville and Wilbur Wright, made the first successful sustained, controlled powered heavier-than-air flight on December 17, 1903, at Kitty Hawk, North Carolina.",
     "Airplane"),
    ("Shakespeare wrote Hamlet.",
     "Hamlet",
     "Hamlet is a tragedy written by William Shakespeare, believed to have been written between 1599 and 1601. It is one of his most famous plays.",
     "Shakespeare"),
    ("The Nile River flows through Egypt.",
     "Nile River",
     "The Nile is a major north-flowing river in northeastern Africa. It flows through eleven countries including Egypt and is historically associated with Egyptian civilization.",
     "Egypt"),
    ("The atom bomb was first used in warfare in 1945.",
     "Atomic bombings of Hiroshima and Nagasaki",
     "The United States detonated two atomic bombs over the Japanese cities of Hiroshima and Nagasaki in August 1945, the first and only use of nuclear weapons in warfare.",
     "World War II"),
    ("The human brain weighs approximately 1.4 kilograms.",
     "Human brain",
     "The adult human brain weighs on average about 1.2 to 1.4 kilograms. It contains approximately 86 billion neurons and is the center of the nervous system.",
     "Brain"),
    ("The Great Fire of London occurred in 1666.",
     "Great Fire of London",
     "The Great Fire of London swept through the central parts of the English city of London from Sunday 2 September to Thursday 6 September 1666.",
     "London"),
    ("Pluto was reclassified as a dwarf planet in 2006.",
     "Pluto",
     "Pluto was reclassified as a dwarf planet by the International Astronomical Union in 2006. Prior to this, it had been considered the ninth planet in the Solar System since its discovery in 1930.",
     "Solar System"),
    ("The Suez Canal connects the Red Sea and the Mediterranean Sea.",
     "Suez Canal",
     "The Suez Canal is an artificial waterway in Egypt connecting the Mediterranean Sea to the Red Sea. It was opened in 1869 and significantly shortened the maritime route between Europe and Asia.",
     "Egypt"),
    ("Nelson Mandela was imprisoned on Robben Island.",
     "Nelson Mandela",
     "Nelson Mandela was imprisoned for 27 years, from 1964 to 1990. Most of this time was served on Robben Island, off the coast of Cape Town, South Africa.",
     "South Africa"),
    ("The telephone was invented by Alexander Graham Bell.",
     "Telephone",
     "The telephone was invented by Alexander Graham Bell, who was awarded the first patent for the electric telephone in 1876.",
     "Bell"),
    ("Homer wrote the Iliad and the Odyssey.",
     "Homer",
     "Homer was an ancient Greek poet traditionally said to be the author of the epic poems the Iliad and the Odyssey, the foundational works of ancient Greek literature.",
     "Greek literature"),
    ("Carbon dioxide is a greenhouse gas.",
     "Carbon dioxide",
     "Carbon dioxide (CO₂) is a greenhouse gas that traps heat in the Earth's atmosphere. It is the primary driver of climate change and is emitted through burning fossil fuels.",
     "Climate change"),
    ("The printing press was invented by Johannes Gutenberg.",
     "Printing press",
     "The printing press was invented by Johannes Gutenberg around 1440. It revolutionised the production of books and contributed to the spread of literacy and the Reformation.",
     "Gutenberg"),
    ("Water boils at 100 degrees Celsius at sea level.",
     "Boiling point of water",
     "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (1 atm) at sea level.",
     "Water"),
    ("The Renaissance originated in Italy.",
     "Renaissance",
     "The Renaissance was a period in European history from the 14th to the 17th century. It began in Italy, in the Late Middle Ages, before spreading to the rest of Europe.",
     "Italy"),
    ("Antarctica is the coldest continent.",
     "Antarctica",
     "Antarctica is Earth's southernmost continent. It is the coldest, driest, and windiest continent, and has the highest average elevation of all the continents.",
     "Continent"),
    ("The human heart has four chambers.",
     "Human heart",
     "The human heart is a muscular organ that pumps blood through the body. It has four chambers: two upper chambers called atria and two lower chambers called ventricles.",
     "Heart"),
    ("Pythagoras' theorem relates the sides of a right triangle.",
     "Pythagorean theorem",
     "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c².",
     "Mathematics"),
    ("Mozart was a child prodigy.",
     "Wolfgang Amadeus Mozart",
     "Wolfgang Amadeus Mozart was an Austrian composer who showed prodigious ability from his earliest childhood. He began composing at age five and performing publicly at age six.",
     "Music"),
    ("The First World War ended in 1918.",
     "World War I",
     "World War I, also known as the First World War, lasted from 1914 to 1918. It ended with the Armistice of November 11, 1918.",
     "Armistice"),
    ("Gold is a chemical element with the symbol Au.",
     "Gold",
     "Gold is a chemical element with the symbol Au (from Latin: aurum) and atomic number 79. It is a bright, slightly orange-yellow, dense, soft, malleable, and ductile metal.",
     "Chemistry"),
    ("The Mississippi River is the longest river in North America.",
     "Mississippi River",
     "The Mississippi River is the second-longest river in North America, flowing 3,730 km from Lake Itasca in Minnesota to the Gulf of Mexico. The Missouri River is actually the longest.",
     "River"),
    ("The first moon landing occurred in 1969.",
     "Apollo 11",
     "Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed on July 20, 1969.",
     "NASA"),
    ("Picasso was a co-founder of Cubism.",
     "Pablo Picasso",
     "Pablo Picasso was a Spanish painter and sculptor who, along with Georges Braque, co-founded Cubism, an avant-garde art movement that began in the early 20th century.",
     "Art"),
    ("The Treaty of Versailles ended World War I.",
     "Treaty of Versailles",
     "The Treaty of Versailles was the most important peace treaty that ended World War I. It was signed on 28 June 1919 between Germany and the Allied Powers.",
     "Peace treaty"),
    ("The human eye can distinguish about 10 million colors.",
     "Color vision",
     "The human eye can distinguish approximately 10 million different colors. This is due to three types of cone cells in the retina that are sensitive to different wavelengths of light.",
     "Vision"),
    ("The United Nations was founded in 1945.",
     "United Nations",
     "The United Nations (UN) is an intergovernmental organization founded on 24 October 1945. It was established after World War II to maintain international peace and security.",
     "International organization"),
    ("Oxygen makes up about 21% of Earth's atmosphere.",
     "Atmosphere of Earth",
     "Earth's atmosphere is composed of about 78% nitrogen, 21% oxygen, and 1% argon and other gases. Oxygen is essential for respiration in most living organisms.",
     "Atmosphere"),
    ("The Hubble Space Telescope was launched in 1990.",
     "Hubble Space Telescope",
     "The Hubble Space Telescope is a space telescope that was launched into low Earth orbit in 1990 and remains in operation. It was named after astronomer Edwin Hubble.",
     "Space telescope"),
    ("Rome was not built in a day (figurative — evidence: it took centuries).",
     "History of Rome",
     "The history of Rome spans over 28 centuries. The city was traditionally founded in 753 BC and grew from a small village to become the centre of one of the world's largest empires.",
     "Roman Empire"),
    ("The Magna Carta was signed in 1215.",
     "Magna Carta",
     "Magna Carta is a royal charter of rights agreed to by King John of England at Runnymede, near Windsor, on 15 June 1215.",
     "England"),
    ("Photosynthesis converts sunlight into energy for plants.",
     "Photosynthesis",
     "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy stored in glucose. It occurs primarily in the chloroplasts.",
     "Plants"),
    ("The Black Death killed a large proportion of Europe's population.",
     "Black Death",
     "The Black Death was a bubonic plague pandemic that occurred in Afro-Eurasia from 1346 to 1353. It killed 30–60% of the population of Europe.",
     "Plague"),
    ("The speed of sound in air is approximately 343 m/s.",
     "Speed of sound",
     "The speed of sound in dry air at 20 degrees Celsius is approximately 343 metres per second (1,235 km/h).",
     "Sound"),
    ("Abraham Lincoln was the 16th President of the United States.",
     "Abraham Lincoln",
     "Abraham Lincoln served as the 16th President of the United States from March 1861 until his assassination in April 1865.",
     "US President"),
    ("The Great Barrier Reef is off the coast of Australia.",
     "Great Barrier Reef",
     "The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs. It is located in the Coral Sea, off the coast of Queensland, Australia.",
     "Reef"),
    ("Nickel has the chemical symbol Ni.",
     "Nickel",
     "Nickel is a chemical element with the symbol Ni and atomic number 28. It is a silvery-white lustrous metal with a slight golden tinge.",
     "Element"),
    ("The Renaissance painter Michelangelo painted the Sistine Chapel ceiling.",
     "Sistine Chapel ceiling",
     "The Sistine Chapel ceiling was painted by Michelangelo between 1508 and 1512, at the commission of Pope Julius II.",
     "Vatican"),
    ("The Pacific Ring of Fire is an area with many earthquakes and volcanoes.",
     "Ring of Fire",
     "The Ring of Fire is a region around much of the rim of the Pacific Ocean where many volcanic eruptions and earthquakes occur.",
     "Pacific"),
    ("Gandhi led nonviolent resistance against British rule in India.",
     "Mahatma Gandhi",
     "Mahatma Gandhi led nonviolent resistance movements against British rule in India, inspiring civil rights and freedom movements across the world.",
     "India"),
    ("The Milky Way is a barred spiral galaxy.",
     "Milky Way",
     "The Milky Way is the galaxy that includes the Solar System. It is a barred spiral galaxy with an estimated diameter of 100,000 light-years.",
     "Galaxy"),
    ("The Renaissance occurred from the 14th to the 17th century.",
     "Renaissance",
     "The Renaissance was a cultural and intellectual movement that began in Italy in the 14th century and spread to the rest of Europe by the 17th century.",
     "History"),
    ("The Taj Mahal was built by Emperor Shah Jahan.",
     "Taj Mahal",
     "The Taj Mahal is an ivory-white marble mausoleum on the south bank of the Yamuna river in Agra, India. It was commissioned in 1632 by Mughal emperor Shah Jahan to house the tomb of his favourite wife.",
     "Mughal Empire"),
    ("Copernicus proposed that the Earth revolves around the Sun.",
     "Nicolaus Copernicus",
     "Nicolaus Copernicus was a Renaissance-era mathematician and astronomer who formulated a model of the universe that placed the Sun rather than Earth at its centre.",
     "Astronomy"),
    ("DNA replication produces two identical copies of a DNA molecule.",
     "DNA replication",
     "DNA replication is the process by which a DNA molecule is copied. This process produces two identical DNA molecules from one original molecule.",
     "Molecular biology"),
    ("The Battle of Waterloo took place in 1815.",
     "Battle of Waterloo",
     "The Battle of Waterloo was fought on 18 June 1815, near Waterloo in Belgium. It was Napoleon Bonaparte's final defeat.",
     "Napoleon"),
    ("The International Space Station orbits Earth.",
     "International Space Station",
     "The International Space Station (ISS) is a modular space station in low Earth orbit. It is a multinational collaborative project involving five participating space agencies.",
     "Space Station"),
    ("Socrates was a Greek philosopher.",
     "Socrates",
     "Socrates was a Greek philosopher from Athens who is credited as the founder of Western philosophy. He is known through the writings of his students Plato and Xenophon.",
     "Philosophy"),
    ("The ozone layer protects Earth from ultraviolet radiation.",
     "Ozone layer",
     "The ozone layer is a region of Earth's stratosphere that absorbs most of the Sun's ultraviolet radiation. Depletion of the ozone layer increases UV radiation reaching the surface.",
     "Atmosphere"),
    ("The Aztec Empire was conquered by Spanish forces.",
     "Aztec Empire",
     "The Aztec Empire was conquered by a Spanish expedition led by Hernán Cortés between 1519 and 1521. The fall of Tenochtitlán in 1521 marked the effective end of the empire.",
     "Mesoamerica"),
    ("Vulcanization of rubber was developed by Charles Goodyear.",
     "Vulcanization",
     "Vulcanization is a process for curing rubber using high heat and sulphur. It was developed by Charles Goodyear and patented in 1844.",
     "Rubber"),
    ("The planet Mars is known as the Red Planet.",
     "Mars",
     "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System. It is often called the Red Planet because of the reddish appearance given by iron oxide on its surface.",
     "Solar System"),
    ("Florence Nightingale is known as the founder of modern nursing.",
     "Florence Nightingale",
     "Florence Nightingale was an English social reformer, statistician, and the founder of modern nursing. She came to prominence during the Crimean War.",
     "Nursing"),
    ("The transistor was invented at Bell Labs.",
     "Transistor",
     "The transistor was invented in 1947 by John Bardeen, Walter Brattain, and William Shockley at Bell Labs. It is a fundamental building block of modern electronics.",
     "Electronics"),
    ("The Earth is approximately 4.5 billion years old.",
     "Age of Earth",
     "Scientists estimate the age of Earth to be approximately 4.54 billion years, based on radiometric dating of meteoric material and Earth rocks.",
     "Geology"),
    ("Marco Polo traveled to China in the 13th century.",
     "Marco Polo",
     "Marco Polo was a Venetian merchant and explorer who traveled through Asia between 1271 and 1295. He journeyed to China, where he served at the court of Kublai Khan.",
     "China"),
    ("The Silk Road connected East and West through trade routes.",
     "Silk Road",
     "The Silk Road was a network of trade routes connecting China and the Far East with the Middle East and Europe. It was central to cultural interaction through regions of the Asian continent.",
     "Trade"),
    ("Julius Caesar was assassinated on the Ides of March.",
     "Assassination of Julius Caesar",
     "Julius Caesar was stabbed to death by a group of Roman senators on the Ides of March (15 March) 44 BC. The assassins included Cassius and Brutus.",
     "Rome"),
    ("The Pythagorean theorem is used to calculate the hypotenuse.",
     "Pythagorean theorem",
     "The Pythagorean theorem states that in any right triangle, the square of the length of the hypotenuse equals the sum of the squares of the other two sides.",
     "Geometry"),
    ("Sigmund Freud developed psychoanalysis.",
     "Sigmund Freud",
     "Sigmund Freud was an Austrian neurologist and the founder of psychoanalysis, a clinical method for treating psychopathology through dialogue between a patient and a psychoanalyst.",
     "Psychology"),
    ("The Gutenberg Bible was one of the first books printed with movable type.",
     "Gutenberg Bible",
     "The Gutenberg Bible was one of the earliest major books printed using mass-produced movable metal type in Europe. It was printed by Johannes Gutenberg in Mainz, Germany, around 1455.",
     "Printing"),
    ("The Colossus of Rhodes was one of the Seven Wonders of the Ancient World.",
     "Colossus of Rhodes",
     "The Colossus of Rhodes was a statue of the Greek sun-god Helios, erected in the city of Rhodes on the Greek island of the same name. It was one of the Seven Wonders of the Ancient World.",
     "Greece"),
    ("Thomas Edison invented the phonograph.",
     "Thomas Edison",
     "Thomas Edison invented many devices in fields such as electric power generation, mass communication, sound recording, and motion pictures. Among his inventions was the phonograph.",
     "Invention"),
    ("The Roman Empire fell in 476 AD.",
     "Fall of the Western Roman Empire",
     "The fall of the Western Roman Empire is conventionally dated to 476 AD, when the last Roman emperor Romulus Augustulus was deposed by Odoacer.",
     "Rome"),
    ("Einstein was awarded the Nobel Prize in Physics in 1921.",
     "Albert Einstein Nobel Prize",
     "Albert Einstein received the Nobel Prize in Physics for 1921, primarily for his discovery of the law of the photoelectric effect, not for his theory of relativity.",
     "Nobel"),
    ("The Battle of Hastings was fought in 1066.",
     "Battle of Hastings",
     "The Battle of Hastings was fought on 14 October 1066 between the Norman-French army of William the Conqueror and the English army of King Harold Godwinson.",
     "England"),
    ("Van Gogh's The Starry Night depicts a swirling night sky.",
     "The Starry Night",
     "The Starry Night is an oil on canvas painting by Dutch Post-Impressionist painter Vincent van Gogh, painted in June 1889. It depicts a swirling night sky over a village.",
     "Art"),
    ("Martin Luther King Jr. delivered the I Have a Dream speech.",
     "I Have a Dream",
     "Martin Luther King Jr. delivered his I Have a Dream speech on August 28, 1963, during the March on Washington for Jobs and Freedom.",
     "Civil rights"),
    ("The Venus de Milo is an ancient Greek sculpture.",
     "Venus de Milo",
     "The Venus de Milo is an ancient Greek sculpture created between 130 and 100 BC, attributed to the sculptor Alexandros of Antioch. It is displayed at the Louvre Museum.",
     "Sculpture"),
    ("The Rosetta Stone was key to deciphering Egyptian hieroglyphics.",
     "Rosetta Stone",
     "The Rosetta Stone is a stele composed of granodiorite inscribed with three versions of a decree issued in Memphis, Egypt, in 196 BC. It was key to the decipherment of Egyptian hieroglyphics.",
     "Egypt"),
    ("Archimedes discovered the principle of buoyancy.",
     "Archimedes",
     "Archimedes of Syracuse discovered the principle of buoyancy — that a body immersed in fluid experiences an upward force equal to the weight of the fluid displaced.",
     "Physics"),
    ("The Hundred Years War lasted more than a century.",
     "Hundred Years War",
     "The Hundred Years War was a series of armed conflicts between England and France from 1337 to 1453, lasting approximately 116 years — longer than a century.",
     "History"),
    ("The compass was invented in China.",
     "Compass",
     "The magnetic compass was invented in ancient China, with the earliest compasses appearing during the Han dynasty around 200 BC. It was later adopted by Islamic sailors and then Europeans.",
     "Navigation"),
    ("The Sun is larger than Earth.",
     "Sun",
     "The Sun is much larger than Earth. The Sun's diameter is about 109 times that of Earth's, and it accounts for about 99.86% of the total mass of the Solar System.",
     "Solar System"),
    ("Insulin is used to treat diabetes.",
     "Insulin",
     "Insulin is a hormone used in the treatment of diabetes mellitus. It was first isolated and used medically in the early 1920s by Frederick Banting and Charles Best.",
     "Medicine"),
]

# REFUTED claims pool: (claim, evidence_title, evidence_text, distractor_entity)
_REFUTED_POOL = [
    ("The Eiffel Tower is located in London.",
     "Eiffel Tower",
     "The Eiffel Tower is located in Paris, France, not in London. It stands on the Champ de Mars near the Seine River in Paris.",
     "London"),
    ("Abraham Lincoln was the first President of the United States.",
     "Abraham Lincoln",
     "Abraham Lincoln was the 16th President of the United States, not the first. George Washington was the first President, serving from 1789 to 1797.",
     "Washington"),
    ("Christopher Columbus was born in Spain.",
     "Christopher Columbus",
     "Christopher Columbus was born in Genoa (in present-day Italy), not Spain. He sailed under the Spanish crown but was Italian by birth.",
     "Genoa"),
    ("The Great Wall of China is visible from space with the naked eye.",
     "Great Wall of China",
     "Contrary to popular belief, the Great Wall of China is not visible from space with the naked eye. Its width is too narrow to be seen from the distance of space.",
     "China"),
    ("Albert Einstein failed mathematics in school.",
     "Albert Einstein",
     "Contrary to popular myth, Albert Einstein did not fail mathematics in school. He excelled in mathematics and physics from an early age.",
     "Mathematics"),
    ("Napoleon Bonaparte was unusually short.",
     "Napoleon Bonaparte",
     "The claim that Napoleon was unusually short is largely a myth. He was about 5 feet 7 inches tall, which was average or above average for a French man of his era.",
     "France"),
    ("Humans use only 10% of their brains.",
     "Human brain",
     "The claim that humans only use 10% of their brains is a myth. Brain imaging research has shown that virtually all parts of the brain have some function.",
     "Neuroscience"),
    ("The Great Fire of London occurred in 1666 and destroyed all of London.",
     "Great Fire of London",
     "While the Great Fire of London did occur in 1666, it did not destroy all of London. It burned about one-third of the city but left many areas untouched.",
     "London"),
    ("Charles Darwin coined the phrase 'survival of the fittest'.",
     "Charles Darwin",
     "The phrase 'survival of the fittest' was coined by philosopher Herbert Spencer, not Charles Darwin. Darwin initially used the phrase 'natural selection'.",
     "Evolution"),
    ("George Washington had wooden teeth.",
     "George Washington",
     "George Washington did not have wooden teeth. His dentures were made from ivory, bone, and human teeth — not wood.",
     "Dentistry"),
    ("The Titanic's lookouts had no binoculars because there were none on board.",
     "RMS Titanic",
     "Binoculars were actually available on the Titanic but had been locked away after a last-minute crew change. The lookouts had binoculars locked in a locker without a key.",
     "Iceberg"),
    ("Benjamin Franklin discovered electricity.",
     "Benjamin Franklin",
     "Benjamin Franklin did not discover electricity. He demonstrated that lightning was electrical in nature. Electricity had been known for centuries before Franklin's experiments.",
     "Lightning"),
    ("The first person to walk on the Moon was Buzz Aldrin.",
     "Apollo 11",
     "The first person to walk on the Moon was Neil Armstrong, not Buzz Aldrin. Armstrong became the first to step onto the lunar surface on July 20, 1969.",
     "Moon landing"),
    ("Dinosaurs and humans lived at the same time.",
     "Dinosaur extinction",
     "Non-avian dinosaurs went extinct approximately 66 million years ago, about 65 million years before the first Homo sapiens appeared. They did not coexist with modern humans.",
     "Extinction"),
    ("The Amazon River is the longest river in the world.",
     "Amazon River",
     "The Amazon River is the largest river by discharge volume, but it is not the longest. The Nile River in Africa is generally considered the longest river in the world.",
     "Rivers"),
    ("Louis Pasteur invented the microscope.",
     "Microscope",
     "The microscope was not invented by Louis Pasteur. It was invented by Antonie van Leeuwenhoek in the 17th century. Pasteur was known for germ theory and pasteurization.",
     "Science"),
    ("The human body has seven chambers in the heart.",
     "Human heart",
     "The human heart has four chambers, not seven: the right atrium, left atrium, right ventricle, and left ventricle.",
     "Anatomy"),
    ("Mount Kilimanjaro is the tallest mountain in the world.",
     "Mount Kilimanjaro",
     "Mount Kilimanjaro is the tallest mountain in Africa but not the tallest in the world. Mount Everest in the Himalayas is the world's tallest mountain.",
     "Mountain"),
    ("The Sahara Desert is the largest desert in the world.",
     "Sahara Desert",
     "The Sahara is the largest hot desert in the world, but Antarctica is actually the world's largest desert overall, as deserts are defined by precipitation, not temperature.",
     "Desert"),
    ("The first computer was invented in the 1980s.",
     "History of computing",
     "The first electronic computers were invented in the 1940s, not the 1980s. ENIAC, one of the first general-purpose computers, was completed in 1945.",
     "Computing"),
    ("Shakespeare was born in London.",
     "William Shakespeare",
     "William Shakespeare was not born in London. He was born and raised in Stratford-upon-Avon, Warwickshire, England.",
     "Birthplace"),
    ("The speed of light is about 30,000 km/s.",
     "Speed of light",
     "The speed of light in vacuum is approximately 299,792 km/s, not 30,000 km/s. It is much faster than 30,000 km/s.",
     "Physics"),
    ("Alexander the Great was Roman.",
     "Alexander the Great",
     "Alexander the Great was not Roman; he was Greek — specifically Macedonian. He was king of the ancient Greek kingdom of Macedonia.",
     "Greece"),
    ("The Black Death was caused by a virus.",
     "Black Death",
     "The Black Death was primarily caused by the bacterium Yersinia pestis, not a virus. It was a bacterial infection transmitted by fleas.",
     "Plague"),
    ("Galileo invented the telescope.",
     "Telescope",
     "Galileo did not invent the telescope. The telescope was invented around 1608 by Hans Lippershey in the Netherlands. Galileo improved the design for astronomical use.",
     "Astronomy"),
    ("The Pacific Ocean is the smallest ocean on Earth.",
     "Pacific Ocean",
     "The Pacific Ocean is the largest ocean on Earth, covering more area than all of Earth's landmasses combined. The Arctic Ocean is the smallest.",
     "Oceans"),
    ("Pluto is classified as a major planet.",
     "Pluto",
     "Pluto was reclassified as a dwarf planet by the International Astronomical Union in 2006. It is no longer considered a major planet.",
     "Solar System"),
    ("Thomas Edison invented the telephone.",
     "Thomas Edison",
     "Thomas Edison did not invent the telephone. The telephone was invented by Alexander Graham Bell, who received the first patent in 1876.",
     "Invention"),
    ("The first successful vaccine was for smallpox.",
     "Smallpox vaccine",
     "The first successful vaccine was indeed for smallpox, developed by Edward Jenner in 1796, not by Louis Pasteur as sometimes claimed.",
     "Medicine"),
    ("Julius Caesar was the first Roman Emperor.",
     "Julius Caesar",
     "Julius Caesar was not the first Roman Emperor. He was a dictator but was assassinated in 44 BC. Augustus (Octavian) became the first Roman Emperor in 27 BC.",
     "Rome"),
    ("The Sun is a planet.",
     "Sun",
     "The Sun is not a planet. It is a star — a massive ball of plasma held together by gravity — at the centre of our Solar System.",
     "Solar System"),
    ("Mozart was born in Vienna.",
     "Wolfgang Amadeus Mozart",
     "Mozart was not born in Vienna. He was born in Salzburg, in what is now Austria, on January 27, 1756.",
     "Salzburg"),
    ("The Great Wall of China was built in one continuous effort.",
     "Great Wall of China",
     "The Great Wall of China was not built in one continuous effort. It was built, rebuilt, and maintained by multiple dynasties over many centuries.",
     "China"),
    ("Isaac Newton discovered gravity by seeing an apple fall from a tree.",
     "Isaac Newton",
     "The story of Newton discovering gravity by seeing an apple fall is largely apocryphal. While he did discuss an apple triggering his thinking, the discovery was the result of years of mathematical work.",
     "Gravity"),
    ("Van Gogh sold many paintings during his lifetime.",
     "Vincent van Gogh",
     "Van Gogh sold only one or very few paintings during his lifetime, not many. He died in poverty and his work became famous only after his death.",
     "Art"),
    ("Einstein was born in the United States.",
     "Albert Einstein",
     "Albert Einstein was not born in the United States. He was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire.",
     "Germany"),
    ("The Amazon rainforest produces 20% of the world's oxygen.",
     "Amazon rainforest",
     "The Amazon rainforest does not produce a net 20% of the world's oxygen. While it produces oxygen through photosynthesis, it also consumes oxygen through respiration.",
     "Oxygen"),
    ("Bats are blind.",
     "Bat",
     "Bats are not blind. They have functional eyes and many species have good vision. They also use echolocation for navigation in the dark.",
     "Biology"),
    ("Cleopatra was Egyptian by ethnicity.",
     "Cleopatra",
     "Cleopatra was not ethnically Egyptian. She was a member of the Ptolemaic dynasty, a family of Greek origin. She was actually of Macedonian Greek descent.",
     "Egypt"),
    ("The Amazon River flows into the Pacific Ocean.",
     "Amazon River",
     "The Amazon River does not flow into the Pacific Ocean. It flows eastward and empties into the Atlantic Ocean near the city of Marajó.",
     "Rivers"),
    ("Mount Vesuvius destroyed Pompeii in the 1st century BC.",
     "Pompeii",
     "Mount Vesuvius erupted and destroyed Pompeii in 79 AD (1st century AD), not in the 1st century BC.",
     "Volcano"),
    ("The first successful airplane flight lasted one hour.",
     "Wright Brothers",
     "The first successful powered airplane flight by the Wright Brothers lasted only 12 seconds and covered about 37 metres — not one hour.",
     "Aviation"),
    ("Magellan was the first person to circumnavigate the globe.",
     "Ferdinand Magellan",
     "Magellan did not complete the circumnavigation; he was killed in the Philippines in 1521. The first person to complete a circumnavigation was Juan Sebastián Elcano.",
     "Navigation"),
    ("The Magna Carta established democracy in England.",
     "Magna Carta",
     "The Magna Carta did not establish democracy in England. It was a charter that limited the power of the king and guaranteed certain rights to nobles, not the general population.",
     "England"),
    ("The Suez Canal was built by Egypt alone.",
     "Suez Canal",
     "The Suez Canal was not built by Egypt alone. It was constructed by the Suez Canal Company, a joint Franco-Egyptian venture. The work was carried out between 1859 and 1869.",
     "Egypt"),
    ("DNA is made up of amino acids.",
     "DNA",
     "DNA is not made up of amino acids. DNA is made up of nucleotides. Amino acids are the building blocks of proteins, not DNA.",
     "Molecular biology"),
    ("The speed of sound is faster than the speed of light.",
     "Speed comparison",
     "The speed of sound (about 343 m/s) is far slower than the speed of light (about 299,792,458 m/s). Light travels almost a million times faster than sound.",
     "Physics"),
    ("Dinosaurs were reptiles that breathed fire.",
     "Dinosaur",
     "Dinosaurs were not capable of breathing fire. No known animal has ever been capable of producing fire from biological processes.",
     "Biology"),
    ("The Colosseum is located in Athens, Greece.",
     "Colosseum",
     "The Colosseum is not located in Athens, Greece. It is an oval amphitheatre located in the centre of Rome, Italy.",
     "Architecture"),
    ("The Great Barrier Reef is in the Caribbean.",
     "Great Barrier Reef",
     "The Great Barrier Reef is not in the Caribbean. It is the world's largest coral reef system, located in the Coral Sea off the coast of Queensland, Australia.",
     "Reef"),
    ("Beethoven was born in Vienna.",
     "Ludwig van Beethoven",
     "Beethoven was not born in Vienna. He was born in Bonn, in the Electorate of Cologne (now western Germany), in December 1770.",
     "Music"),
    ("The Statue of Liberty was built in the United States.",
     "Statue of Liberty",
     "The Statue of Liberty was not built in the United States. It was designed by French sculptor Frédéric Auguste Bartholdi and built in France before being shipped and assembled in New York Harbour.",
     "France"),
    ("Honey never expires because it is sterile.",
     "Honey",
     "Honey does not expire but the reason is not simply that it is sterile. Its preservation is due to its low water content, high acidity, and natural hydrogen peroxide — not sterility.",
     "Food science"),
    ("Lightning never strikes the same place twice.",
     "Lightning",
     "Lightning can and frequently does strike the same place more than once. Tall structures like the Empire State Building are struck dozens of times per year.",
     "Meteorology"),
    ("The Earth is flat according to some ancient civilizations.",
     "Flat Earth",
     "While some ancient cultures held flat Earth beliefs, Greek philosophers such as Pythagoras and Aristotle recognized that the Earth was spherical as early as the 6th century BC.",
     "History of science"),
    ("Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid.",
     "Cleopatra timeline",
     "This is actually TRUE, not false. Cleopatra died in 30 BC, about 2,500 years after the Great Pyramid was built (~2560 BC), but only about 2,000 years before the Moon landing (1969 AD). However, this claim is often presented as counterintuitive, so it functions as a verification challenge.",
     "Timeline"),
    ("Humans have only five senses.",
     "Human senses",
     "Humans have more than five senses. In addition to sight, hearing, taste, touch, and smell, humans also have proprioception, vestibular sense, thermoception, and others.",
     "Biology"),
    ("The Declaration of Independence was signed on July 4, 1776.",
     "Declaration of Independence",
     "The Declaration of Independence was not signed on July 4, 1776. Congress voted to adopt the declaration on July 4, but most signatures were added on August 2, 1776.",
     "American history"),
    ("Einstein never received a Nobel Prize.",
     "Albert Einstein Nobel Prize",
     "Einstein did receive a Nobel Prize. He was awarded the Nobel Prize in Physics for 1921, for his discovery of the law of the photoelectric effect.",
     "Physics"),
    ("The currency of the United Kingdom is the Euro.",
     "United Kingdom currency",
     "The currency of the United Kingdom is the pound sterling (GBP), not the Euro. The UK did not adopt the Euro and retained the pound sterling.",
     "Economics"),
    ("Marie Curie was born in France.",
     "Marie Curie",
     "Marie Curie was not born in France. She was born Maria Sklodowska on November 7, 1867, in Warsaw, which was then part of the Russian Empire (now Poland).",
     "Biography"),
    ("Sharks are mammals.",
     "Shark",
     "Sharks are not mammals; they are fish — specifically cartilaginous fish. They breathe through gills, are cold-blooded, and lack mammary glands.",
     "Biology"),
    ("The Roman Empire and the British Empire existed at the same time.",
     "Roman Empire timeline",
     "The Roman Empire and the British Empire did not exist at the same time. The Western Roman Empire fell in 476 AD, while the British Empire formally emerged in the 17th century.",
     "History"),
    ("The Great Sphinx of Giza was built by Alexander the Great.",
     "Great Sphinx of Giza",
     "The Great Sphinx of Giza was not built by Alexander the Great. It is generally attributed to the Pharaoh Khafra, built around 2500 BC, thousands of years before Alexander's time.",
     "Egypt"),
    ("The capital of Canada is Toronto.",
     "Capital of Canada",
     "The capital of Canada is Ottawa, not Toronto. Toronto is the largest city and the capital of Ontario, but Ottawa has been the federal capital since 1857.",
     "Geography"),
    ("Penguins live in the Arctic.",
     "Penguin",
     "Penguins do not live in the Arctic. They are native to the Southern Hemisphere, primarily in Antarctica and sub-Antarctic islands.",
     "Biology"),
    ("The French language originated in France.",
     "French language",
     "This is a simplification. Latin, the parent language, was brought to Gaul by Roman conquest. Modern French evolved from Vulgar Latin and was shaped by various cultural influences over centuries.",
     "Linguistics"),
    ("All deserts are hot.",
     "Desert",
     "Not all deserts are hot. A desert is defined by low precipitation, not by temperature. Antarctica, for example, is the world's largest desert and is extremely cold.",
     "Geography"),
    ("The Mona Lisa was painted using oil paints on canvas.",
     "Mona Lisa",
     "The Mona Lisa was not painted on canvas. It was painted using oil paints on a panel of white Lombardy poplar wood.",
     "Art"),
    ("Edison invented the light bulb from scratch.",
     "Light bulb",
     "Edison did not invent the light bulb from scratch. Earlier inventors had already created incandescent lamps. Edison's contribution was a practical, long-lasting version using a carbonized bamboo filament.",
     "Invention"),
    ("The South Pole is warmer than the North Pole.",
     "Polar temperatures",
     "The South Pole (Antarctica) is significantly colder than the North Pole (Arctic). Antarctica is a high-elevation continent, whereas the Arctic is mostly ocean covered in ice.",
     "Geography"),
    ("The original Olympics were held in Athens.",
     "Ancient Olympics",
     "The original Olympic Games were not held in Athens. They were held at Olympia, a sanctuary site in the western Peloponnese region of Greece.",
     "History"),
    ("Goldfish have a three-second memory.",
     "Goldfish memory",
     "The claim that goldfish have a three-second memory is a myth. Research has shown that goldfish can remember things for months and can be trained to navigate mazes.",
     "Biology"),
    ("Antibiotics are effective against viral infections.",
     "Antibiotics",
     "Antibiotics are not effective against viral infections. They work only against bacterial infections. Using antibiotics for viral illnesses contributes to antibiotic resistance.",
     "Medicine"),
    ("The Earth revolves around the Moon.",
     "Earth-Moon system",
     "The Earth does not revolve around the Moon. The Moon orbits the Earth due to Earth's much greater gravitational force.",
     "Astronomy"),
    ("The Gettysburg Address was written on the back of an envelope.",
     "Gettysburg Address",
     "The story that Lincoln wrote the Gettysburg Address on the back of an envelope is a myth. Multiple drafts exist, written on proper paper before the speech.",
     "American history"),
    ("Glass is a liquid that flows very slowly.",
     "Glass",
     "The claim that glass is a slow-moving liquid is a common misconception. Glass is an amorphous solid. Old window glass is thicker at the bottom because of manufacturing techniques, not flow.",
     "Materials science"),
    ("Mount Fuji is the tallest mountain in Asia.",
     "Mount Fuji",
     "Mount Fuji is not the tallest mountain in Asia. At 3,776 metres, it is the tallest mountain in Japan, but many mountains in the Himalayas are far taller.",
     "Geography"),
    ("Magellan circumnavigated the globe as captain of his expedition.",
     "Ferdinand Magellan",
     "Magellan did not circumnavigate the globe as captain. He was killed in the Battle of Mactan in the Philippines in 1521. The expedition was completed under Juan Sebastián Elcano.",
     "Exploration"),
    ("The Wright Brothers' first flight took place in Kitty Hawk, North Carolina.",
     "Wright Brothers",
     "The Wright Brothers' first flight actually took place at Kill Devil Hills, near Kitty Hawk, North Carolina — not in Kitty Hawk itself, though Kitty Hawk is commonly cited.",
     "Aviation"),
    ("The Great Pyramid of Giza was built by slaves.",
     "Great Pyramid of Giza",
     "Modern archaeological evidence suggests the Great Pyramid was not built by slaves. Workers appear to have been paid laborers — skilled craftsmen who received food, medical care, and wages.",
     "Archaeology"),
    ("Australia is a country in Asia.",
     "Australia",
     "Australia is not a country in Asia. It is the primary country of the Australian continent and is part of the Oceania region.",
     "Geography"),
    ("Frankenstein is the name of the monster in the novel.",
     "Frankenstein",
     "Frankenstein is actually the name of the scientist (Victor Frankenstein) in Mary Shelley's novel, not the monster. The creature in the novel is never given a name.",
     "Literature"),
    ("The invention of the steam engine is credited to James Watt.",
     "Steam engine",
     "James Watt did not invent the steam engine. He significantly improved the design invented by Thomas Newcomen. Watt's improvements made the steam engine practical for widespread use.",
     "Engineering"),
    ("The Battle of Thermopylae was won by the Spartans.",
     "Battle of Thermopylae",
     "The Battle of Thermopylae (480 BC) was actually won by the Persians. The Spartans and their allies were ultimately defeated, though they inflicted heavy casualties on the Persian forces.",
     "Ancient history"),
    ("The Amazon rainforest is located entirely within Brazil.",
     "Amazon rainforest",
     "The Amazon rainforest is not entirely within Brazil. It spans multiple countries including Peru, Colombia, Venezuela, Ecuador, Bolivia, Guyana, Suriname, and French Guiana.",
     "Geography"),
    ("Silicon Valley is in Silicon Valley, California.",
     "Silicon Valley location",
     "Silicon Valley is a colloquial term for the area in the southern part of the San Francisco Bay Area in California. It is not a literal valley — the name refers to silicon chips and technology companies.",
     "Technology"),
    ("The Forbidden City is in Shanghai.",
     "Forbidden City",
     "The Forbidden City is not in Shanghai. It is located in the centre of Beijing, China, and served as the imperial palace from 1420 to 1912.",
     "China"),
    ("Cleopatra was the last pharaoh of Egypt.",
     "Cleopatra",
     "Cleopatra VII is often considered the last active pharaoh of ancient Egypt, but after her death, Caesarion (her son with Julius Caesar) ruled briefly before being executed. So she was technically the last significant pharaoh.",
     "Egypt"),
    ("The Sistine Chapel is in Venice.",
     "Sistine Chapel",
     "The Sistine Chapel is not in Venice. It is located in Vatican City, within Rome, Italy.",
     "Italy"),
    ("The Nobel Prize was established by Marie Curie.",
     "Nobel Prize",
     "The Nobel Prize was not established by Marie Curie. It was established by Alfred Nobel, a Swedish inventor and industrialist, through his will of 1895.",
     "Awards"),
    ("Marco Polo introduced pizza to Italy from China.",
     "Pizza history",
     "The claim that Marco Polo introduced pizza from China to Italy is a myth. Pizza evolved from ancient flatbreads in the Mediterranean region, long before Marco Polo's travels.",
     "Food history"),
    ("The Pyramids of Giza are located in the Sahara Desert.",
     "Pyramids of Giza",
     "The Pyramids of Giza are technically at the edge of the Sahara, but they are situated on the Giza Plateau near Cairo. They are not deep in the desert, but in an agricultural region along the Nile.",
     "Egypt"),
    ("The Periodic Table was invented by John Dalton.",
     "Periodic Table",
     "The Periodic Table was not invented by John Dalton. It was developed by Dmitri Mendeleev in 1869. Dalton developed atomic theory and a table of atomic weights, which are different.",
     "Chemistry"),
    ("The United States declared independence from France.",
     "American independence",
     "The United States did not declare independence from France. The Declaration of Independence in 1776 was a declaration of independence from Great Britain.",
     "History"),
    ("Hamlet was written by Christopher Marlowe.",
     "Hamlet",
     "Hamlet was not written by Christopher Marlowe. It was written by William Shakespeare, most likely between 1599 and 1601.",
     "Literature"),
    ("The Nile is the longest river in Africa, but the Missouri is the longest in North America.",
     "Longest rivers",
     "The Missouri River is not the longest river in North America. The Mississippi-Missouri river system is, and the Amazon in South America is the largest by volume. The Mississippi is often cited as the longest US river.",
     "Geography"),
    ("Thomas Jefferson wrote the US Constitution.",
     "US Constitution",
     "Thomas Jefferson did not write the US Constitution. The Constitution was drafted primarily by James Madison. Jefferson was in France at the time as US minister.",
     "American history"),
    ("Cheetahs are the largest big cat.",
     "Cheetah",
     "Cheetahs are not the largest big cat. The tiger holds that distinction. Cheetahs are the fastest land animal but are among the smaller members of the big cat family.",
     "Biology"),
    ("The Berlin Wall was built to keep West Germans out of East Germany.",
     "Berlin Wall",
     "The Berlin Wall was built primarily to prevent East Germans from fleeing to West Germany, not to keep West Germans out. It was constructed in 1961 by the East German government.",
     "Cold War"),
    ("World War II ended in 1945 when Germany surrendered.",
     "World War II end",
     "World War II ended in 1945, but Germany's surrender in May 1945 did not end the entire war. Japan surrendered in September 1945, officially ending the Second World War.",
     "History"),
]


def _make_fever_example(
    idx: int,
    claim: str,
    label: str,  # "SUPPORTED" or "REFUTED"
    evidence_title: str,
    evidence_text: str,
    distractor_entity: str,
    rng: random.Random,
) -> dict:
    """Build one FEVER-style fact-verification example."""

    gold_para = {
        "title": evidence_title,
        "paragraph_text": evidence_text,
        "is_supporting": True,
    }

    distractors = [
        {
            "title": f"Overview of {distractor_entity}",
            "paragraph_text": (
                f"{distractor_entity} is a well-documented subject with extensive literature. "
                f"Scholars have studied its history, significance, and various dimensions. "
                f"Public interest in {distractor_entity} has grown steadily over the years."
            ),
            "is_supporting": False,
        },
        {
            "title": f"History of {distractor_entity}",
            "paragraph_text": (
                f"The history of {distractor_entity} dates back many centuries. "
                f"Multiple periods have been identified in its development. "
                f"Key events shaped its contemporary significance."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Related claims about {evidence_title}",
            "paragraph_text": (
                f"There are many claims related to {evidence_title} in popular culture. "
                f"Some of these claims have been verified while others remain disputed. "
                f"Fact-checking organisations have examined multiple such claims."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Cultural significance of {distractor_entity}",
            "paragraph_text": (
                f"The cultural significance of {distractor_entity} cannot be understated. "
                f"It has influenced art, literature, and public discourse. "
                f"Its impact is felt across multiple disciplines."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Common misconceptions in history",
            "paragraph_text": (
                "There are many widely believed misconceptions about historical events. "
                "These myths persist despite being contradicted by historical evidence. "
                "Educational efforts have been made to address these misunderstandings."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Scientific facts and myths",
            "paragraph_text": (
                "Science has corrected many popular myths and misconceptions over the centuries. "
                "Careful observation and experimentation are the tools scientists use to verify claims. "
                "Not all widely held beliefs align with empirical evidence."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Encyclopedia entry: {distractor_entity}",
            "paragraph_text": (
                f"According to multiple reference sources, {distractor_entity} is associated with "
                f"a variety of historical and scientific developments. "
                f"Its role in shaping modern understanding is widely acknowledged."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Recent research on {evidence_title}",
            "paragraph_text": (
                f"Recent research has revisited many aspects of {evidence_title}. "
                f"New evidence and methods have refined earlier conclusions. "
                f"The scholarly consensus continues to evolve as more data becomes available."
            ),
            "is_supporting": False,
        },
        {
            "title": f"Verification challenges",
            "paragraph_text": (
                "Verifying historical and scientific claims can be challenging due to limited sources. "
                "Primary sources are preferred over secondary interpretations. "
                "Fact-checking requires careful analysis of available evidence."
            ),
            "is_supporting": False,
        },
    ]

    paragraphs = [gold_para] + distractors
    rng.shuffle(paragraphs)

    return {
        "id": f"fever_{label.lower()}_{idx:03d}",
        "claim": claim,
        "label": label,
        "evidence_title": evidence_title,
        "paragraphs": paragraphs,
    }


def build_fever_dataset(n: int = N_EXAMPLES, seed: int = SEED) -> list[dict]:
    """
    Build a synthetic FEVER-style dataset.

    n // 2 SUPPORTED + n // 2 REFUTED claims.
    Each claim has 10 paragraphs (1 gold evidence + 9 distractors).

    Returns list of raw dicts.
    """
    rng = random.Random(seed)

    n_supported = n // 2
    n_refuted = n - n_supported

    # Use the pools (they have 100 entries each, so sample as needed)
    supported_pool = list(_SUPPORTED_POOL)
    refuted_pool = list(_REFUTED_POOL)

    rng.shuffle(supported_pool)
    rng.shuffle(refuted_pool)

    supported_pool = supported_pool[:n_supported]
    refuted_pool = refuted_pool[:n_refuted]

    examples = []

    for i, (claim, ev_title, ev_text, distractor) in enumerate(supported_pool):
        ex = _make_fever_example(i, claim, "SUPPORTED", ev_title, ev_text, distractor, rng)
        examples.append(ex)

    for i, (claim, ev_title, ev_text, distractor) in enumerate(refuted_pool):
        ex = _make_fever_example(i, claim, "REFUTED", ev_title, ev_text, distractor, rng)
        examples.append(ex)

    rng.shuffle(examples)

    print(
        f"Built FEVER-style dataset: "
        f"{n_supported} SUPPORTED + {n_refuted} REFUTED = {len(examples)} total"
    )
    return examples


def _try_load_huggingface_fever(n: int, seed: int) -> Optional[list[dict]]:
    """Try loading FEVER from HuggingFace. Returns None on failure."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets library not available")
        return None

    rng = random.Random(seed)

    # Try FEVER dataset variants
    hf_configs = [
        ("fever", "v1.0"),
        ("fever", "v2.0"),
        ("fever", None),
    ]

    for hf_name, config in hf_configs:
        for split in ("paper_dev", "dev", "validation"):
            try:
                cfg_str = f"config={config}" if config else "default"
                print(f"  Trying HF dataset '{hf_name}' ({cfg_str}) split='{split}' ...")
                if config:
                    ds = load_dataset(hf_name, config, split=split, trust_remote_code=True)
                else:
                    ds = load_dataset(hf_name, split=split, trust_remote_code=True)

                raw = list(ds)
                if not raw:
                    continue

                print(f"    Loaded {len(raw)} examples")

                # Filter to SUPPORTED and REFUTED only (drop NOT ENOUGH INFO)
                supported = [e for e in raw if e.get("label") == "SUPPORTS" or e.get("label") == "SUPPORTED"]
                refuted = [e for e in raw if e.get("label") == "REFUTES" or e.get("label") == "REFUTED"]

                print(f"    SUPPORTED: {len(supported)}, REFUTED: {len(refuted)}")

                n_s = min(n // 2, len(supported))
                n_r = min(n - n_s, len(refuted))

                if n_s + n_r < 20:
                    print(f"    Too few examples, skipping")
                    continue

                rng.shuffle(supported)
                rng.shuffle(refuted)

                selected = supported[:n_s] + refuted[:n_r]
                rng.shuffle(selected)
                print(f"    Selected {len(selected)} examples ({n_s} supported + {n_r} refuted)")
                return selected

            except Exception as e:
                print(f"    {hf_name}/{split}: {str(e)[:120]}")

    return None


def convert_fever_raw(raw: dict, idx: int) -> dict:
    """
    Convert a raw FEVER example to EvaluationHarness format.

    FEVER format varies by version — this handles common formats.
    Falls back to a minimal conversion if the format is unexpected.
    """
    label = raw.get("label", raw.get("verifiable", "SUPPORTED"))
    claim = raw.get("claim", "")

    # Try to extract evidence passages
    evidence_sets = raw.get("evidence", [])
    context_docs = []
    gold_ids = []

    # FEVER evidence format: list of evidence sets, each a list of [annotation, _, title, sent_id]
    if evidence_sets:
        seen_titles: set[str] = set()
        for ev_group in evidence_sets:
            if not isinstance(ev_group, list):
                continue
            for ev in ev_group:
                if isinstance(ev, (list, tuple)) and len(ev) >= 3:
                    title = ev[2] if ev[2] else None
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        gold_ids.append(title)
                        context_docs.append({
                            "id": title,
                            "title": title,
                            "content": f"Evidence from Wikipedia article: {title}. "
                                       f"This article contains information relevant to the claim.",
                        })

    # If no evidence structure found, create minimal context
    if not context_docs:
        context_docs = [{"id": f"doc_{idx}", "title": f"doc_{idx}", "content": claim}]
        gold_ids = [f"doc_{idx}"]

    return {
        "id": raw.get("id", f"fever_{idx:04d}"),
        "question": claim,   # harness uses "question" field
        "answer": label,
        "label": label,
        "context": context_docs,
        "gold_ids": gold_ids,
    }


def convert_fever_synthetic(raw: dict) -> dict:
    """
    Convert synthetic FEVER example to EvaluationHarness format.
    """
    paragraphs = raw.get("paragraphs", [])
    context_docs = []
    gold_ids = []

    for para in paragraphs:
        title = para.get("title", "")
        text = para.get("paragraph_text", "")
        context_docs.append({
            "id": title,
            "title": title,
            "content": text,
        })
        if para.get("is_supporting", False) and title not in gold_ids:
            gold_ids.append(title)

    return {
        "id": raw.get("id", ""),
        "question": raw.get("claim", ""),   # harness uses "question" field
        "answer": raw.get("label", ""),
        "label": raw.get("label", ""),
        "context": context_docs,
        "gold_ids": gold_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Policy and address-space factories
# ─────────────────────────────────────────────────────────────────────────────

def make_policies() -> list:
    """
    Instantiate policies for FEVER evaluation.
    Focus on the key comparison: pi_semantic, pi_lexical, pi_ensemble, pi_heuristic.
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
# Statistical testing
# ─────────────────────────────────────────────────────────────────────────────

def paired_ttest(a_vals: list[float], b_vals: list[float], metric: str) -> dict:
    """
    Paired two-sided t-test comparing two lists of per-example metric values.

    Returns a dict with t_stat, p_value, and significance flag.
    """
    if len(a_vals) != len(b_vals) or len(a_vals) < 2:
        return {"t_stat": float("nan"), "p_value": float("nan"), "significant": False}

    a_arr = np.array(a_vals, dtype=float)
    b_arr = np.array(b_vals, dtype=float)

    t_stat, p_value = scipy_stats.ttest_rel(a_arr, b_arr)
    return {
        "metric": metric,
        "mean_a": float(np.mean(a_arr)),
        "mean_b": float(np.mean(b_arr)),
        "mean_diff": float(np.mean(a_arr - b_arr)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_p05": bool(p_value < 0.05),
        "significant_p01": bool(p_value < 0.01),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_fever(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Run all policies on n_examples FEVER fact-verification examples.

    Tests: Does source-diversity stopping (pi_heuristic) beat ensemble on NON-QA?

    Returns dict keyed by policy name.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("FEVER Fact Verification Benchmark — AEA Evaluation")
    print("(Testing generalisation beyond QA tasks)")
    print("=" * 60)
    print()

    # ── Step 1: Load or construct data ────────────────────────────────────────
    print("Step 1: Loading dataset ...")

    raw_examples = _try_load_huggingface_fever(n_examples, seed)
    is_synthetic = False

    if raw_examples is None:
        print("\nAll HuggingFace sources failed. Constructing synthetic FEVER-style dataset ...")
        raw_examples = build_fever_dataset(n=n_examples, seed=seed)
        is_synthetic = True
        dataset = [convert_fever_synthetic(ex) for ex in raw_examples]
    else:
        dataset = [convert_fever_raw(ex, i) for i, ex in enumerate(raw_examples)]

    # Inject label into converted dataset from original
    if not is_synthetic:
        for ex, raw in zip(dataset, raw_examples):
            label = raw.get("label", raw.get("verifiable", "SUPPORTED"))
            ex["label"] = label

    n_actual = len(dataset)
    n_supported = sum(1 for ex in dataset if ex.get("label", "") in ("SUPPORTED", "SUPPORTS"))
    n_refuted = sum(1 for ex in dataset if ex.get("label", "") in ("REFUTED", "REFUTES"))

    print(f"\nDataset ready: {n_actual} examples total")
    print(f"  SUPPORTED: {n_supported}")
    print(f"  REFUTED:   {n_refuted}")
    print(f"  synthetic: {is_synthetic}")
    print()

    # ── Step 2: Retrieval evaluation ─────────────────────────────────────────
    print("Step 2: Retrieval evaluation ...")
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

        # Inject label into per_example
        id_to_label = {ex["id"]: ex.get("label", "") for ex in dataset}
        for ex in result["per_example"]:
            ex["label"] = id_to_label.get(ex.get("id", ""), "")

        agg = result["aggregated"]
        print(f"  support_recall:    {agg['support_recall']:.4f}")
        print(f"  support_precision: {agg['support_precision']:.4f}")
        print(f"  avg_operations:    {agg['operations_used']:.2f}")
        print(f"  avg_steps:         {agg['steps_taken']:.2f}")
        print(f"  utility@budget:    {agg['utility_at_budget']:.4f}")
        print(f"  n_errors:          {result['n_errors']}")
        print(f"  runtime:           {elapsed:.1f}s")

        # Per-label breakdown
        by_label = {}
        for lbl in ("SUPPORTED", "REFUTED"):
            subset = [r for r in result["per_example"]
                      if r.get("label", "").upper() in (lbl, lbl + "S")]
            if subset:
                keys = ["support_recall", "support_precision", "operations_used", "utility_at_budget"]
                by_label[lbl] = {
                    k: float(np.mean([r[k] for r in subset if k in r and isinstance(r[k], (int, float))]))
                    for k in keys
                }

        result["by_label"] = by_label
        all_results[pname] = result

    # ── Step 3: Statistical tests ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 3: Statistical tests (heuristic vs ensemble)")
    print(f"{'=' * 60}")

    heuristic_results = all_results.get("pi_aea_heuristic", {})
    ensemble_results = all_results.get("pi_ensemble", {})

    statistical_tests = {}

    if heuristic_results and ensemble_results:
        h_per = heuristic_results.get("per_example", [])
        e_per = ensemble_results.get("per_example", [])

        # Align by example id
        e_map = {r["id"]: r for r in e_per}
        h_aligned = []
        e_aligned = []
        for r in h_per:
            if r["id"] in e_map:
                h_aligned.append(r)
                e_aligned.append(e_map[r["id"]])

        if h_aligned and e_aligned:
            for metric in ("support_recall", "utility_at_budget", "operations_used"):
                h_vals = [r.get(metric, 0.0) for r in h_aligned]
                e_vals = [r.get(metric, 0.0) for r in e_aligned]
                test = paired_ttest(h_vals, e_vals, metric)
                statistical_tests[metric] = test

                direction = "heuristic > ensemble" if test["mean_a"] > test["mean_b"] else "ensemble > heuristic"
                print(
                    f"  {metric:25s}: {direction:30s}  "
                    f"diff={test['mean_diff']:+.4f}  "
                    f"p={test['p_value']:.4f}  "
                    f"{'*' if test['significant_p05'] else ' '}"
                )

    all_results["statistical_tests"] = statistical_tests

    # ── Step 4: Summary table ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Summary: FEVER Retrieval Results")
    print(f"{'=' * 60}")
    print(f"{'Policy':<25} {'SupportRecall':>13} {'AvgOps':>8} {'U@B':>8}")
    print("-" * 60)

    policy_order = ["pi_semantic", "pi_lexical", "pi_ensemble", "pi_aea_heuristic"]
    for pname in policy_order:
        if pname not in all_results:
            continue
        agg = all_results[pname]["aggregated"]
        print(
            f"  {pname:<23} {agg['support_recall']:>13.4f} "
            f"{agg['operations_used']:>8.2f} "
            f"{agg['utility_at_budget']:>8.4f}"
        )

    # ── Step 5: Key question answer ───────────────────────────────────────────
    h_sr = all_results.get("pi_aea_heuristic", {}).get("aggregated", {}).get("support_recall", None)
    e_sr = all_results.get("pi_ensemble", {}).get("aggregated", {}).get("support_recall", None)
    h_ops = all_results.get("pi_aea_heuristic", {}).get("aggregated", {}).get("operations_used", None)
    e_ops = all_results.get("pi_ensemble", {}).get("aggregated", {}).get("operations_used", None)
    h_ub = all_results.get("pi_aea_heuristic", {}).get("aggregated", {}).get("utility_at_budget", None)
    e_ub = all_results.get("pi_ensemble", {}).get("aggregated", {}).get("utility_at_budget", None)

    print(f"\n{'=' * 60}")
    print("KEY QUESTION: Does source-diversity stopping beat ensemble")
    print("              on a NON-QA task (FEVER fact verification)?")
    print(f"{'=' * 60}")

    if h_sr is not None and e_sr is not None:
        sr_diff = h_sr - e_sr
        ops_diff = (h_ops or 0) - (e_ops or 0)
        ub_diff = (h_ub or 0) - (e_ub or 0)

        print(f"  SupportRecall:  heuristic={h_sr:.4f}  ensemble={e_sr:.4f}  diff={sr_diff:+.4f}")
        print(f"  AvgOps:         heuristic={h_ops:.2f}    ensemble={e_ops:.2f}     diff={ops_diff:+.2f}")
        print(f"  U@B:            heuristic={h_ub:.4f}  ensemble={e_ub:.4f}  diff={ub_diff:+.4f}")

        sr_test = statistical_tests.get("support_recall", {})
        ub_test = statistical_tests.get("utility_at_budget", {})

        if h_ub is not None and e_ub is not None and h_ub > e_ub:
            sig = ub_test.get("significant_p05", False)
            print(f"\n  ANSWER: YES — heuristic BEATS ensemble on FEVER")
            print(f"          U@B improvement: {ub_diff:+.4f}  (p={ub_test.get('p_value', float('nan')):.4f})")
            if sig:
                print("          Result is statistically significant (p < 0.05)")
            else:
                print("          Result is NOT statistically significant (p >= 0.05)")
        elif h_sr is not None and e_sr is not None and h_sr >= e_sr and (h_ops or 0) < (e_ops or 0):
            print(f"\n  ANSWER: YES (Pareto) — heuristic matches recall, uses fewer ops")
        else:
            print(f"\n  ANSWER: NO — ensemble appears better on this NON-QA task")
            print("          (Finding may not generalise — or data is insufficient)")

    # ── Save results ──────────────────────────────────────────────────────────
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)

        def _make_serializable(obj):
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_make_serializable(v) for v in obj]
            elif isinstance(obj, float):
                if obj != obj:  # NaN
                    return None
                return obj
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            else:
                return obj

        save_data = {
            "experiment": "FEVER Fact Verification — AEA Source-Diversity Stopping",
            "n_examples": n_actual,
            "n_supported": n_supported,
            "n_refuted": n_refuted,
            "seed": seed,
            "is_synthetic": is_synthetic,
            "results": _make_serializable(all_results),
            "statistical_tests": _make_serializable(statistical_tests),
        }

        with open(results_path, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"\nResults saved to {results_path}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_fever()
    print("\nDONE")
