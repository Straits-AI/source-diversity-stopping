"""
Structural Navigation Benchmark for AEA Framework.

100 structure-dependent questions where the document TITLE (not content)
is the primary navigation signal — simulating filesystem browsing.

Question types
--------------
1. discovery (50 q):
   "Which department handles X?" — the answer is in a document whose TITLE
   contains the department name.  The question never names the department
   directly; only the topic/function is mentioned.

2. extraction (50 q):
   "What is Y's role in organization Z?" — requires finding the document
   titled "Organization Z" then reading the relevant section within it.

Dataset schema (harness-compatible)
------------------------------------
Each example:
  id        — str
  question  — str
  answer    — str
  context   — list of {"id": str, "title": str, "content": str}
              10 documents per question: 2 gold, 8 distractors
              Titles are informative (like filenames)
  gold_ids  — list of str (document ids of the 2 gold documents)

Seed: 42, no API calls.
"""

from __future__ import annotations

import random
import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary pools
# ─────────────────────────────────────────────────────────────────────────────

# --- DISCOVERY questions (50)
# Each tuple: (department_name, function_topic, answer_fact, dept_content, policy_content)
#   department_name  — the TITLE of the gold doc (agent must navigate to it)
#   function_topic   — what the question asks about (NOT in title)
#   answer_fact      — the answer string
#   dept_content     — content of the "who handles X" document
#   policy_content   — content of a related policy document (2nd gold)
_DISCOVERY_POOL = [
    (
        "Procurement and Vendor Management",
        "supplier contracts",
        "Procurement and Vendor Management",
        "The Procurement and Vendor Management department oversees all supplier contracts, vendor evaluations, and purchasing agreements across the organisation.",
        "All supplier contract renewals above $50,000 require sign-off from the Procurement and Vendor Management director.",
    ),
    (
        "Environmental Health and Safety",
        "workplace hazard inspections",
        "Environmental Health and Safety",
        "Environmental Health and Safety (EHS) is responsible for workplace hazard inspections, safety audits, and regulatory compliance reporting.",
        "Incident reports must be submitted to Environmental Health and Safety within 24 hours of the event.",
    ),
    (
        "Corporate Communications",
        "press releases and media inquiries",
        "Corporate Communications",
        "Corporate Communications manages all press releases, media inquiries, brand messaging, and external stakeholder engagement.",
        "Media inquiries from journalists must be routed through Corporate Communications before any response is issued.",
    ),
    (
        "Talent Acquisition",
        "new hire onboarding",
        "Talent Acquisition",
        "Talent Acquisition handles all recruitment campaigns, candidate screening, offer negotiations, and new hire onboarding programmes.",
        "The 90-day onboarding curriculum is designed and delivered by Talent Acquisition in coordination with hiring managers.",
    ),
    (
        "Data Governance and Privacy",
        "personal data access requests",
        "Data Governance and Privacy",
        "Data Governance and Privacy processes all personal data access requests (SARs), data retention schedules, and GDPR compliance activities.",
        "Any team receiving a data subject access request must escalate it to Data Governance and Privacy within two business days.",
    ),
    (
        "Facilities and Real Estate",
        "office space allocation",
        "Facilities and Real Estate",
        "Facilities and Real Estate manages office space allocation, building maintenance, lease negotiations, and physical security infrastructure.",
        "Requests for additional desk space or meeting room assignments are submitted to Facilities and Real Estate via the internal portal.",
    ),
    (
        "Customer Success",
        "enterprise client renewals",
        "Customer Success",
        "Customer Success oversees enterprise client renewals, health-score monitoring, and adoption coaching for contracted accounts.",
        "Renewal risk alerts are generated automatically and assigned to the relevant Customer Success manager.",
    ),
    (
        "Internal Audit",
        "financial control reviews",
        "Internal Audit",
        "Internal Audit conducts financial control reviews, operational audits, and compliance assessments on a risk-prioritised schedule.",
        "Business units must provide Internal Audit with unrestricted access to ledgers and transaction records during scheduled reviews.",
    ),
    (
        "Learning and Development",
        "mandatory compliance training",
        "Learning and Development",
        "Learning and Development designs and administers all mandatory compliance training, skills programmes, and leadership academies.",
        "Completion rates for mandatory compliance training are tracked and reported monthly by Learning and Development.",
    ),
    (
        "Treasury and Cash Management",
        "foreign currency hedging",
        "Treasury and Cash Management",
        "Treasury and Cash Management handles foreign currency hedging, liquidity forecasting, and intercompany funding arrangements.",
        "All FX hedging transactions above $1 million must be approved by the Treasury and Cash Management committee.",
    ),
    (
        "Regulatory Affairs",
        "product compliance certifications",
        "Regulatory Affairs",
        "Regulatory Affairs manages product compliance certifications, agency submissions, and post-market surveillance obligations.",
        "New product launches require a compliance certificate issued by Regulatory Affairs before market release.",
    ),
    (
        "Business Continuity",
        "disaster recovery planning",
        "Business Continuity",
        "Business Continuity owns disaster recovery planning, crisis response playbooks, and annual business impact analyses.",
        "Each division must validate its recovery time objectives with Business Continuity during the annual BIA cycle.",
    ),
    (
        "Investor Relations",
        "shareholder communications",
        "Investor Relations",
        "Investor Relations coordinates shareholder communications, earnings call logistics, and analyst briefing materials.",
        "Quarterly earnings scripts must be reviewed and approved by Investor Relations before the CFO call.",
    ),
    (
        "Product Lifecycle Management",
        "end-of-life product transitions",
        "Product Lifecycle Management",
        "Product Lifecycle Management governs end-of-life product transitions, version deprecation schedules, and migration support plans.",
        "Customer migration timelines for deprecated products are published by Product Lifecycle Management six months in advance.",
    ),
    (
        "Cloud Infrastructure",
        "capacity planning for production systems",
        "Cloud Infrastructure",
        "Cloud Infrastructure is responsible for capacity planning for production systems, cloud cost optimisation, and platform reliability.",
        "Monthly capacity utilisation reports are produced by Cloud Infrastructure and shared with finance for budget alignment.",
    ),
    (
        "Supply Chain Operations",
        "logistics and freight coordination",
        "Supply Chain Operations",
        "Supply Chain Operations manages logistics and freight coordination, warehouse inventory, and last-mile delivery contracts.",
        "All freight invoices above $10,000 require dual approval from Supply Chain Operations and Finance.",
    ),
    (
        "Information Security",
        "vulnerability disclosures",
        "Information Security",
        "Information Security triages and manages vulnerability disclosures, penetration test findings, and patch prioritisation.",
        "External researchers must submit vulnerability reports to Information Security via the published responsible-disclosure form.",
    ),
    (
        "Enterprise Architecture",
        "technology standardisation decisions",
        "Enterprise Architecture",
        "Enterprise Architecture sets technology standardisation decisions, reviews architecture proposals, and maintains the IT blueprint.",
        "Any new SaaS tool acquisition must be assessed by Enterprise Architecture before procurement approval.",
    ),
    (
        "Compensation and Benefits",
        "salary band reviews",
        "Compensation and Benefits",
        "Compensation and Benefits administers salary band reviews, bonus plan design, and employee benefits enrolment.",
        "Annual pay equity analyses are conducted by Compensation and Benefits and presented to the People Committee.",
    ),
    (
        "Quality Assurance",
        "product defect escalation",
        "Quality Assurance",
        "Quality Assurance owns product defect escalation processes, test plan sign-off, and release quality gates.",
        "A severity-1 defect must be escalated to Quality Assurance leadership within one hour of discovery.",
    ),
    (
        "Partner Ecosystem",
        "reseller programme management",
        "Partner Ecosystem",
        "Partner Ecosystem manages the reseller programme, channel incentives, and co-sell agreements with technology partners.",
        "New reseller applications are reviewed and approved by Partner Ecosystem on a quarterly basis.",
    ),
    (
        "Financial Planning and Analysis",
        "budget variance reporting",
        "Financial Planning and Analysis",
        "Financial Planning and Analysis produces budget variance reports, reforecast models, and scenario analyses for the CFO.",
        "Departmental heads submit actuals to Financial Planning and Analysis by the fifth business day of each month.",
    ),
    (
        "Legal and Compliance",
        "contract dispute resolution",
        "Legal and Compliance",
        "Legal and Compliance handles contract dispute resolution, litigation management, and regulatory penalty responses.",
        "Any contract disputes involving amounts over $25,000 must be referred to Legal and Compliance for review.",
    ),
    (
        "Research and Development",
        "patent filing and IP protection",
        "Research and Development",
        "Research and Development manages patent filing, intellectual property protection strategies, and innovation grants.",
        "Inventor disclosure forms must be submitted to Research and Development within 30 days of a novel concept being documented.",
    ),
    (
        "Workplace Experience",
        "employee wellbeing initiatives",
        "Workplace Experience",
        "Workplace Experience designs and runs employee wellbeing initiatives, engagement surveys, and social events.",
        "Results from the biannual engagement survey are presented to the Executive Team by Workplace Experience.",
    ),
    (
        "Marketing Operations",
        "campaign performance analytics",
        "Marketing Operations",
        "Marketing Operations tracks campaign performance analytics, manages the marketing tech stack, and reports ROI to leadership.",
        "All new marketing automation workflows require sign-off from Marketing Operations before activation.",
    ),
    (
        "Revenue Operations",
        "sales forecasting accuracy",
        "Revenue Operations",
        "Revenue Operations is accountable for sales forecasting accuracy, CRM data integrity, and pipeline reporting.",
        "The weekly revenue forecast is compiled by Revenue Operations every Monday by 9 a.m.",
    ),
    (
        "Sustainability and ESG",
        "carbon footprint reporting",
        "Sustainability and ESG",
        "Sustainability and ESG owns carbon footprint reporting, ESG disclosures, and supplier sustainability assessments.",
        "Scope 3 emissions data is collected by Sustainability and ESG from tier-one suppliers annually.",
    ),
    (
        "Mergers and Acquisitions",
        "due diligence coordination",
        "Mergers and Acquisitions",
        "Mergers and Acquisitions coordinates due diligence, integration planning, and deal-close activities for all corporate transactions.",
        "Due diligence data rooms are provisioned and managed by Mergers and Acquisitions.",
    ),
    (
        "Technical Documentation",
        "API reference maintenance",
        "Technical Documentation",
        "Technical Documentation maintains API reference guides, developer tutorials, and release notes across all product lines.",
        "All public-facing API changes must be documented by Technical Documentation before the release date.",
    ),
    (
        "Payroll Services",
        "overtime pay calculations",
        "Payroll Services",
        "Payroll Services processes overtime pay calculations, statutory deductions, and year-end tax filings for all employees.",
        "Overtime hours must be approved by line managers and submitted to Payroll Services by the 15th of each month.",
    ),
    (
        "Risk Management",
        "enterprise risk assessments",
        "Risk Management",
        "Risk Management conducts enterprise risk assessments, maintains the corporate risk register, and advises the Board.",
        "Each business unit updates its risk register entries with Risk Management on a semi-annual basis.",
    ),
    (
        "Digital Marketing",
        "search engine optimisation strategy",
        "Digital Marketing",
        "Digital Marketing owns search engine optimisation strategy, paid media campaigns, and social channel management.",
        "Organic traffic targets are set jointly by Digital Marketing and the VP of Growth each quarter.",
    ),
    (
        "Hardware Engineering",
        "printed circuit board validation",
        "Hardware Engineering",
        "Hardware Engineering is responsible for printed circuit board validation, prototype builds, and hardware reliability testing.",
        "All PCB designs must pass review by Hardware Engineering before being sent to the manufacturing partner.",
    ),
    (
        "Software Platform",
        "API gateway management",
        "Software Platform",
        "Software Platform manages the API gateway, authentication services, and shared microservice infrastructure.",
        "Any change to the API gateway routing rules requires approval from Software Platform's technical lead.",
    ),
    (
        "Customer Support",
        "escalated technical complaints",
        "Customer Support",
        "Customer Support handles escalated technical complaints, refund processing, and service-level-agreement breach resolution.",
        "Tickets unresolved after 72 hours are automatically escalated within Customer Support to tier-2 agents.",
    ),
    (
        "Strategic Alliances",
        "joint venture governance",
        "Strategic Alliances",
        "Strategic Alliances oversees joint venture governance, memoranda of understanding, and strategic partnership frameworks.",
        "All joint venture steering committee meetings are chaired by a representative from Strategic Alliances.",
    ),
    (
        "Logistics Technology",
        "warehouse management system integration",
        "Logistics Technology",
        "Logistics Technology owns warehouse management system integration, barcode scanning infrastructure, and IoT sensor deployments.",
        "WMS upgrade plans must be reviewed and approved by Logistics Technology before vendor selection.",
    ),
    (
        "Employee Relations",
        "grievance and disciplinary procedures",
        "Employee Relations",
        "Employee Relations manages grievance and disciplinary procedures, mediation services, and trade union liaison.",
        "Formal grievances must be acknowledged by Employee Relations within three working days of receipt.",
    ),
    (
        "Knowledge Management",
        "internal wiki governance",
        "Knowledge Management",
        "Knowledge Management governs the internal wiki, document version control standards, and institutional knowledge capture.",
        "Outdated articles are identified and queued for review by Knowledge Management on a monthly basis.",
    ),
    (
        "Service Desk",
        "IT incident ticket routing",
        "Service Desk",
        "Service Desk triages IT incident tickets, routes them to appropriate resolver groups, and tracks resolution SLAs.",
        "Priority-one incidents are handled by Service Desk within 15 minutes of being raised.",
    ),
    (
        "Actuarial Services",
        "insurance reserve calculations",
        "Actuarial Services",
        "Actuarial Services produces insurance reserve calculations, pricing models, and mortality assumption reviews.",
        "Reserve adequacy opinions from Actuarial Services are required before each regulatory filing.",
    ),
    (
        "Brand Management",
        "trademark registration",
        "Brand Management",
        "Brand Management handles trademark registration, brand guideline enforcement, and licensing of brand assets.",
        "Any use of the company logo in third-party materials must be approved by Brand Management.",
    ),
    (
        "Health Informatics",
        "clinical data interoperability",
        "Health Informatics",
        "Health Informatics manages clinical data interoperability standards, EHR integration, and patient data pipelines.",
        "HL7 FHIR implementation guides are published and maintained by Health Informatics.",
    ),
    (
        "Grants and Funding",
        "research grant applications",
        "Grants and Funding",
        "Grants and Funding coordinates research grant applications, funder compliance, and project reporting obligations.",
        "All external funding applications must be reviewed by Grants and Funding before submission to the funder.",
    ),
    (
        "Corporate Tax",
        "transfer pricing documentation",
        "Corporate Tax",
        "Corporate Tax prepares transfer pricing documentation, manages tax authority audits, and advises on M&A tax structuring.",
        "Intercompany loan agreements must be reviewed by Corporate Tax to ensure arm's-length pricing.",
    ),
    (
        "Fleet Management",
        "vehicle maintenance scheduling",
        "Fleet Management",
        "Fleet Management coordinates vehicle maintenance scheduling, driver compliance checks, and fleet procurement.",
        "Annual roadworthiness certificates for company vehicles are tracked and renewed by Fleet Management.",
    ),
    (
        "Events and Sponsorship",
        "trade show participation",
        "Events and Sponsorship",
        "Events and Sponsorship manages trade show participation, conference sponsorships, and internal company events.",
        "Booth design assets for trade shows are produced in coordination with Events and Sponsorship and the brand team.",
    ),
    (
        "Clinical Operations",
        "clinical trial site management",
        "Clinical Operations",
        "Clinical Operations oversees clinical trial site management, patient recruitment, and protocol deviation reporting.",
        "Site initiation visits are arranged by Clinical Operations at least four weeks before first patient enrolment.",
    ),
    (
        "Airline Partnerships",
        "code-share agreement negotiations",
        "Airline Partnerships",
        "Airline Partnerships negotiates code-share agreements, frequent-flyer programme alliances, and interline ticketing deals.",
        "New code-share routes require regulatory approval coordinated by Airline Partnerships and Legal.",
    ),
    (
        "Industrial Relations",
        "collective bargaining agreements",
        "Industrial Relations",
        "Industrial Relations manages collective bargaining agreement negotiations, union consultation forums, and strike contingency plans.",
        "Proposed changes to shift patterns must be presented to Industrial Relations 90 days before implementation.",
    ),
]

# --- EXTRACTION questions (50)
# Each tuple: (org_name, person_name, role_title, extra_fact, org_content, person_content)
#   org_name      — title of doc 1 (the organisation)
#   person_name   — the person the question asks about
#   role_title    — the answer
#   extra_fact    — additional fact in person doc
#   org_content   — paragraph about the organisation
#   person_content— paragraph about the person inside the organisation
_EXTRACTION_POOL = [
    (
        "Thornwick Foundation",
        "Mirela Ostroff",
        "Chief Programme Officer",
        "Mirela joined in 2017 after a decade in international development.",
        "The Thornwick Foundation is a charitable organisation focused on rural literacy and STEM education in underserved regions.",
        "Mirela Ostroff serves as Chief Programme Officer at the Thornwick Foundation, overseeing all grant-making and impact evaluation.",
    ),
    (
        "Crestfield Polytechnic",
        "Edmund Vasara",
        "Dean of Engineering",
        "Edmund holds a PhD in mechanical engineering from Uppsala University.",
        "Crestfield Polytechnic is a technical university offering undergraduate and postgraduate programmes in engineering, computing, and applied sciences.",
        "Edmund Vasara is the Dean of Engineering at Crestfield Polytechnic, responsible for curriculum oversight and faculty recruitment.",
    ),
    (
        "Aldervane Municipal Council",
        "Petra Holub",
        "Director of Urban Planning",
        "Petra led the award-winning city-centre regeneration plan of 2021.",
        "Aldervane Municipal Council is the local government authority responsible for planning, transport, housing, and civic infrastructure.",
        "Petra Holub holds the role of Director of Urban Planning within Aldervane Municipal Council.",
    ),
    (
        "Harrowgate Insurance Group",
        "Kofi Mensah-Asante",
        "Head of Claims Analytics",
        "Kofi introduced machine-learning triage for claims in 2020.",
        "Harrowgate Insurance Group provides property, casualty, and life insurance products to retail and commercial customers across Europe.",
        "Kofi Mensah-Asante is Head of Claims Analytics at Harrowgate Insurance Group, managing a team of 15 data scientists.",
    ),
    (
        "Velstrom Genomics",
        "Ingrid Sollberg",
        "VP of Regulatory Strategy",
        "Ingrid previously worked at the EMA as a scientific assessor.",
        "Velstrom Genomics develops gene-therapy platforms for rare inherited metabolic disorders.",
        "Ingrid Sollberg is VP of Regulatory Strategy at Velstrom Genomics, coordinating submissions to the FDA and EMA.",
    ),
    (
        "Brindlemoor Publishing",
        "Tomas Ferreira",
        "Commissioning Editor",
        "Tomas has commissioned over 200 titles in the history and social-science genres.",
        "Brindlemoor Publishing is an independent book publisher specialising in academic and general-interest non-fiction.",
        "Tomas Ferreira is a Commissioning Editor at Brindlemoor Publishing, focusing on history, politics, and social sciences.",
    ),
    (
        "Duskhollow Shipyard",
        "Annette Kovacs",
        "Chief Naval Architect",
        "Annette designed the yard's flagship container-vessel series launched in 2019.",
        "Duskhollow Shipyard constructs commercial vessels including bulk carriers, container ships, and offshore support vessels.",
        "Annette Kovacs serves as Chief Naval Architect at Duskhollow Shipyard, leading structural design and class approval.",
    ),
    (
        "Porthmere Water Authority",
        "Seun Adeyemi",
        "Head of Infrastructure Resilience",
        "Seun manages a $40 million programme to replace ageing pipe networks.",
        "Porthmere Water Authority operates drinking water treatment, distribution, and wastewater services for the Porthmere region.",
        "Seun Adeyemi is Head of Infrastructure Resilience at Porthmere Water Authority, overseeing asset replacement programmes.",
    ),
    (
        "Wychford Academy of Arts",
        "Celia Drummond",
        "Registrar",
        "Celia introduced a fully digital admissions system in 2022.",
        "Wychford Academy of Arts is a conservatoire offering professional training in music, dance, and visual arts.",
        "Celia Drummond is the Registrar of Wychford Academy of Arts, managing student records and admissions processes.",
    ),
    (
        "Greymount Logistics",
        "Yusuf Al-Rashid",
        "Chief Operations Officer",
        "Yusuf expanded the company's cold-chain capacity by 30% in 2023.",
        "Greymount Logistics provides temperature-controlled warehousing, last-mile delivery, and supply-chain consulting services.",
        "Yusuf Al-Rashid is Chief Operations Officer at Greymount Logistics, accountable for network performance and fleet management.",
    ),
    (
        "Calven Energy Solutions",
        "Nora Blackwood",
        "Head of Grid Integration",
        "Nora holds a patent on adaptive load-balancing algorithms for renewable sources.",
        "Calven Energy Solutions designs and installs grid-scale battery storage and renewable energy management systems.",
        "Nora Blackwood leads the Grid Integration team at Calven Energy Solutions, connecting battery assets to national grid infrastructure.",
    ),
    (
        "West Valmere Health Trust",
        "Domingo Sanchez-Ruiz",
        "Medical Director",
        "Domingo previously served as a consultant cardiologist for 14 years.",
        "West Valmere Health Trust operates four district hospitals and twelve community health centres across the region.",
        "Domingo Sanchez-Ruiz is Medical Director of West Valmere Health Trust, responsible for clinical governance and safety.",
    ),
    (
        "Amber City Breweries",
        "Fiona Tannock",
        "Head Brewer",
        "Fiona introduced the brewery's award-winning saison range in 2020.",
        "Amber City Breweries is a craft brewery producing ales, lagers, and seasonal specials distributed across four countries.",
        "Fiona Tannock is Head Brewer at Amber City Breweries, overseeing all fermentation processes and recipe development.",
    ),
    (
        "Saint Aldric Hospital",
        "Oscar Lindqvist",
        "Director of Surgical Services",
        "Oscar performed the hospital's first robotic-assisted cardiac surgery in 2021.",
        "Saint Aldric Hospital is a tertiary-care medical centre providing specialist surgical, oncological, and emergency services.",
        "Oscar Lindqvist is Director of Surgical Services at Saint Aldric Hospital, managing theatre scheduling and surgical governance.",
    ),
    (
        "Greymount Polytechnic",
        "Vera Halvorsen",
        "Professor of Applied Mathematics",
        "Vera has published 48 peer-reviewed papers on numerical optimisation.",
        "Greymount Polytechnic is a research-intensive university known for its engineering and science faculties.",
        "Vera Halvorsen holds the position of Professor of Applied Mathematics at Greymount Polytechnic.",
    ),
    (
        "Northbrook Asset Management",
        "Claude Fontaine",
        "Portfolio Manager",
        "Claude manages a $2.4 billion multi-asset income fund.",
        "Northbrook Asset Management is an investment firm offering equity, fixed-income, and multi-asset funds to institutional clients.",
        "Claude Fontaine is a Portfolio Manager at Northbrook Asset Management, specialising in dividend income strategies.",
    ),
    (
        "Eastmark Pharmaceuticals",
        "Lara Novotny",
        "Senior Vice President of Clinical Development",
        "Lara has overseen Phase III trials for three approved oncology drugs.",
        "Eastmark Pharmaceuticals is a biopharmaceutical company focused on oncology and rare-disease drug development.",
        "Lara Novotny is Senior Vice President of Clinical Development at Eastmark Pharmaceuticals.",
    ),
    (
        "Falcon Ridge Aerospace",
        "Bertrand Leclerc",
        "Chief Test Pilot",
        "Bertrand has logged over 8,000 flight hours across 45 aircraft types.",
        "Falcon Ridge Aerospace designs and manufactures regional turboprop aircraft and unmanned aerial vehicles.",
        "Bertrand Leclerc is Chief Test Pilot at Falcon Ridge Aerospace, conducting initial flight testing for all new platforms.",
    ),
    (
        "Ironforge Mining",
        "Siobhan Maher",
        "Group Safety Manager",
        "Siobhan reduced lost-time injury frequency by 62% over four years.",
        "Ironforge Mining operates open-cast and underground mines extracting copper, nickel, and cobalt.",
        "Siobhan Maher is Group Safety Manager at Ironforge Mining, responsible for safety culture and incident investigation.",
    ),
    (
        "Redwood Digital Bank",
        "Arjun Nair",
        "Chief Risk Officer",
        "Arjun implemented the bank's real-time transaction-fraud detection platform.",
        "Redwood Digital Bank is a challenger bank offering current accounts, lending, and investment products entirely online.",
        "Arjun Nair serves as Chief Risk Officer at Redwood Digital Bank, overseeing credit, market, and operational risk.",
    ),
    (
        "Coastal Fisheries Cooperative",
        "Brigitte Moulin",
        "Sustainability Director",
        "Brigitte negotiated the cooperative's first Marine Stewardship Council certification.",
        "Coastal Fisheries Cooperative is a member-owned fishing enterprise operating trawlers across the North Atlantic.",
        "Brigitte Moulin is Sustainability Director at Coastal Fisheries Cooperative, managing environmental compliance and certifications.",
    ),
    (
        "Solberg Institute of Technology",
        "Hamid Karimi",
        "Director of Robotics Research",
        "Hamid leads a group of 22 researchers working on soft-robotics actuators.",
        "Solberg Institute of Technology is a public research university with strengths in robotics, AI, and biomedical engineering.",
        "Hamid Karimi is Director of Robotics Research at Solberg Institute of Technology.",
    ),
    (
        "Meridian Architecture Studio",
        "Yuki Tanaka",
        "Principal Architect",
        "Yuki won the national architecture prize in 2022 for the Meridian Cultural Centre design.",
        "Meridian Architecture Studio is a design practice specialising in public buildings, cultural spaces, and sustainable urban design.",
        "Yuki Tanaka is Principal Architect at Meridian Architecture Studio, leading all major public commission projects.",
    ),
    (
        "Borealis Semiconductor",
        "Gunnar Eriksen",
        "VP of Process Engineering",
        "Gunnar's team achieved 3nm node production at the Borealis fab in 2024.",
        "Borealis Semiconductor manufactures advanced logic chips and memory modules for consumer and industrial markets.",
        "Gunnar Eriksen is VP of Process Engineering at Borealis Semiconductor, responsible for fab yield and process development.",
    ),
    (
        "Thornside Conservation Trust",
        "Amara Diallo",
        "Head of Species Recovery",
        "Amara led the successful reintroduction programme for the white-tailed eagle.",
        "Thornside Conservation Trust is a non-profit protecting biodiversity across 120,000 hectares of woodland and wetland.",
        "Amara Diallo is Head of Species Recovery at Thornside Conservation Trust.",
    ),
    (
        "Pinnacle Consulting Group",
        "Helena Vasquez",
        "Managing Partner",
        "Helena opened the group's Asia-Pacific office in Singapore in 2021.",
        "Pinnacle Consulting Group provides strategy, technology, and organisational transformation advisory services.",
        "Helena Vasquez is Managing Partner at Pinnacle Consulting Group, responsible for the firm's global client portfolio.",
    ),
    (
        "Harbour Rail Network",
        "Thomas Osei",
        "Network Planning Director",
        "Thomas authored the 30-year capacity investment plan published in 2023.",
        "Harbour Rail Network operates commuter and regional rail services across a metropolitan area of three million passengers per day.",
        "Thomas Osei is Network Planning Director at Harbour Rail Network, leading timetable development and infrastructure investment analysis.",
    ),
    (
        "Summit Renewables",
        "Anya Borisova",
        "Head of Offshore Wind",
        "Anya's team is developing a 1.2 GW offshore wind farm off the northern coast.",
        "Summit Renewables develops and operates wind, solar, and battery storage projects across Europe and North America.",
        "Anya Borisova is Head of Offshore Wind at Summit Renewables, managing project development from site assessment to financial close.",
    ),
    (
        "Lakeside Data Systems",
        "Priya Venkatesh",
        "Chief Data Officer",
        "Priya established the company's enterprise data mesh architecture in 2022.",
        "Lakeside Data Systems provides cloud data warehousing, analytics platforms, and data-integration solutions.",
        "Priya Venkatesh is Chief Data Officer at Lakeside Data Systems, setting the data strategy and governance framework.",
    ),
    (
        "Crownfield Media Group",
        "Sebastien Beaumont",
        "Editor-in-Chief",
        "Sebastien expanded the group's digital subscriber base to 4.2 million.",
        "Crownfield Media Group publishes national newspapers and operates news websites across six countries.",
        "Sebastien Beaumont is Editor-in-Chief of Crownfield Media Group, responsible for editorial standards across all titles.",
    ),
    (
        "Alpine Telecom",
        "Marta Cieslak",
        "VP of Network Infrastructure",
        "Marta oversaw deployment of 5G to 85% of the country's population.",
        "Alpine Telecom is a national telecommunications operator providing mobile, broadband, and enterprise connectivity services.",
        "Marta Cieslak is VP of Network Infrastructure at Alpine Telecom, accountable for capital investment in towers and fibre.",
    ),
    (
        "Clearwater Food Group",
        "Devon Ashworth",
        "Director of Food Safety",
        "Devon introduced the HACCP digital monitoring system across all plants in 2021.",
        "Clearwater Food Group manufactures packaged foods across eight factories and distributes to retail chains nationwide.",
        "Devon Ashworth is Director of Food Safety at Clearwater Food Group, overseeing quality management and regulatory inspections.",
    ),
    (
        "Riverstone Legal",
        "Natalie Okonkwo",
        "Head of Employment Law",
        "Natalie has represented clients in over 300 employment tribunal hearings.",
        "Riverstone Legal is a full-service law firm with specialist practices in corporate, employment, real estate, and litigation.",
        "Natalie Okonkwo leads the Employment Law practice at Riverstone Legal.",
    ),
    (
        "Cascade Biotech",
        "Felipe Amaral",
        "Director of Process Development",
        "Felipe developed the bioreactor scale-up protocol used in the firm's flagship cell-therapy product.",
        "Cascade Biotech develops cell and gene therapies for haematological cancers and immunological disorders.",
        "Felipe Amaral is Director of Process Development at Cascade Biotech, responsible for manufacturing process design and tech transfer.",
    ),
    (
        "Meridian Rail Consortium",
        "Grace Oduya",
        "Chief Commercial Officer",
        "Grace secured a 12-year operating franchise valued at £3.4 billion.",
        "Meridian Rail Consortium is a joint venture operating long-distance intercity rail services across three countries.",
        "Grace Oduya is Chief Commercial Officer of Meridian Rail Consortium, leading revenue management and franchise compliance.",
    ),
    (
        "Holmwood Institute",
        "Lars Bergstrom",
        "Research Director",
        "Lars has published over 60 papers on climate modelling and extreme weather prediction.",
        "Holmwood Institute is an independent scientific research organisation focused on environmental science and public health.",
        "Lars Bergstrom serves as Research Director at Holmwood Institute, overseeing all scientific programmes and external partnerships.",
    ),
    (
        "Irondale Manufacturing",
        "Claudia Weiß",
        "VP of Production",
        "Claudia led the rollout of lean manufacturing across all five production sites.",
        "Irondale Manufacturing produces industrial valves, actuators, and flow-control equipment for the oil-and-gas and water sectors.",
        "Claudia Weiß is VP of Production at Irondale Manufacturing, managing factory output and continuous improvement.",
    ),
    (
        "Sunrise Hospitality Group",
        "Marcus Chen",
        "Group Food and Beverage Director",
        "Marcus has developed the dining concept for 18 new hotel openings.",
        "Sunrise Hospitality Group manages a portfolio of 45 hotels ranging from budget to luxury across Asia and Europe.",
        "Marcus Chen is Group Food and Beverage Director at Sunrise Hospitality Group, setting culinary strategy across the portfolio.",
    ),
    (
        "Foxvale Insurance",
        "Ingeborg Holm",
        "Chief Underwriting Officer",
        "Ingeborg redesigned the commercial property underwriting framework, reducing loss ratios by 8 percentage points.",
        "Foxvale Insurance offers commercial and personal lines insurance with a focus on mid-market and SME customers.",
        "Ingeborg Holm is Chief Underwriting Officer at Foxvale Insurance, responsible for pricing strategy and portfolio mix.",
    ),
    (
        "Northgate Civic Theatre",
        "Damian Okafor",
        "Artistic Director",
        "Damian has directed 14 world premieres since joining the theatre in 2018.",
        "Northgate Civic Theatre is a producing theatre presenting drama, musical theatre, and new writing to 200,000 visitors annually.",
        "Damian Okafor is Artistic Director of Northgate Civic Theatre, selecting the programme and directing headline productions.",
    ),
    (
        "Pacific Basin Shipping",
        "Yuri Morozov",
        "Head of Fleet Management",
        "Yuri oversees a fleet of 42 bulk carriers with a combined deadweight of 3.8 million tonnes.",
        "Pacific Basin Shipping operates dry-bulk cargo shipping services across the Asia-Pacific and Indian Ocean trade lanes.",
        "Yuri Morozov is Head of Fleet Management at Pacific Basin Shipping, responsible for vessel operations and crewing.",
    ),
    (
        "Clearsky Satellite",
        "Laila Ibrahim",
        "VP of Ground Systems",
        "Laila manages the network of 17 ground stations supporting real-time earth-observation data delivery.",
        "Clearsky Satellite operates a constellation of earth-observation satellites providing imagery and analytics services.",
        "Laila Ibrahim is VP of Ground Systems at Clearsky Satellite, overseeing telemetry, tracking, and command infrastructure.",
    ),
    (
        "Ridgeline Capital",
        "Benjamin Strauss",
        "Head of Private Equity",
        "Benjamin has led investments totalling $1.8 billion across healthcare and technology buyouts.",
        "Ridgeline Capital is an alternative investment firm managing private equity, real estate, and infrastructure funds.",
        "Benjamin Strauss leads the Private Equity division at Ridgeline Capital.",
    ),
    (
        "Tidal Energy Authority",
        "Roisin Brennan",
        "Director of Licensing",
        "Roisin managed the permitting process for the UK's first commercial tidal stream array.",
        "Tidal Energy Authority is a public body responsible for licensing and regulating marine renewable energy projects.",
        "Roisin Brennan is Director of Licensing at Tidal Energy Authority, overseeing consent applications and environmental assessments.",
    ),
    (
        "Clearwater Climate Fund",
        "Samuel Nkrumah",
        "Investment Director",
        "Samuel has deployed $600 million into carbon reduction projects across Sub-Saharan Africa.",
        "Clearwater Climate Fund finances clean energy, sustainable agriculture, and nature-based solutions in emerging markets.",
        "Samuel Nkrumah is Investment Director at Clearwater Climate Fund, leading deal origination and portfolio oversight.",
    ),
    (
        "Metropolis Transit Authority",
        "Diana Sotomayor",
        "Chief of Staff",
        "Diana coordinates the authority's legislative agenda and board relations.",
        "Metropolis Transit Authority manages bus, metro, and light-rail services for a city of 5 million residents.",
        "Diana Sotomayor is Chief of Staff at Metropolis Transit Authority, supporting the CEO on strategy and cross-divisional initiatives.",
    ),
    (
        "Brightline Digital Agency",
        "Soren Christoffersen",
        "Head of Experience Design",
        "Soren's team won three Cannes Lions awards in 2023 for interactive digital campaigns.",
        "Brightline Digital Agency provides UX design, brand strategy, and digital product development services to enterprise clients.",
        "Soren Christoffersen is Head of Experience Design at Brightline Digital Agency, leading user research and service design.",
    ),
    (
        "Carrington Biomedical",
        "Elena Petrov",
        "VP of Translational Research",
        "Elena bridges bench research and clinical application for the firm's diagnostics pipeline.",
        "Carrington Biomedical develops in-vitro diagnostic kits, laboratory instruments, and point-of-care testing solutions.",
        "Elena Petrov is VP of Translational Research at Carrington Biomedical.",
    ),
    (
        "Moonridge Asset Services",
        "Oluwaseun Adebayo",
        "Director of Client Reporting",
        "Oluwaseun migrated the reporting platform to a cloud-native architecture, cutting report generation time by 70%.",
        "Moonridge Asset Services provides fund administration, custody, and reporting services to asset managers and pension funds.",
        "Oluwaseun Adebayo is Director of Client Reporting at Moonridge Asset Services.",
    ),
    (
        "Ashford City Council",
        "Penelope Straub",
        "Head of Economic Development",
        "Penelope attracted four major inward investment projects totalling £220 million in 2023.",
        "Ashford City Council is the principal local authority responsible for planning, economic development, and social services.",
        "Penelope Straub leads Economic Development at Ashford City Council, promoting business investment and job creation.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Distractor document pools
# ─────────────────────────────────────────────────────────────────────────────

_DISTRACTOR_TITLES = [
    "Annual Report Summary",
    "Board Meeting Minutes",
    "Employee Handbook Section 4",
    "IT Security Policy v3",
    "Health and Safety Regulations Overview",
    "Marketing Strategy Q3",
    "Product Roadmap 2024",
    "Customer Onboarding Guide",
    "Vendor Assessment Framework",
    "Financial Statements FY2023",
    "Corporate Governance Policy",
    "Data Protection Impact Assessment",
    "Network Architecture Diagram Notes",
    "Change Management Playbook",
    "Crisis Communication Protocol",
    "Training Calendar Q2",
    "Expense Reimbursement Guidelines",
    "Strategic Planning Workshop Notes",
    "Sustainability Report Draft",
    "Quarterly Business Review Template",
    "Incident Response Runbook",
    "Budget Approval Workflow",
    "Risk Register Template",
    "Compliance Checklist",
    "Stakeholder Communication Plan",
    "Service Level Agreement Template",
    "Project Charter Template",
    "Contractor Engagement Policy",
    "Performance Review Framework",
    "Knowledge Base Article 1042",
    "System Integration Architecture",
    "Procurement Request Form",
    "Legal Entity Overview",
    "Brand Assets Usage Guide",
    "Succession Planning Framework",
    "Internal Mobility Programme Details",
    "Diversity and Inclusion Strategy",
    "Innovation Lab Charter",
    "Customer Feedback Summary Q4",
    "Office Safety Procedures Manual",
    "Software Licensing Inventory",
    "Travel Policy Update",
    "Capacity Planning Assumptions",
    "Meeting Room Booking Policy",
    "New Starter Checklist",
    "Exit Interview Process",
    "Benefits Enrolment Guide",
    "Archive and Records Management Policy",
    "Code of Conduct Summary",
    "Whistleblower Policy Overview",
]

_DISTRACTOR_CONTENTS = [
    "This document summarises the key financial and operational results for the fiscal year.",
    "The board reviewed progress against strategic objectives and approved the revised budget forecast.",
    "Section 4 covers employee conduct, performance expectations, and disciplinary procedures.",
    "This policy defines acceptable use of IT systems, password requirements, and incident reporting.",
    "All employees must complete mandatory health and safety induction training within 30 days of joining.",
    "The marketing strategy outlines target segments, campaign priorities, and channel allocation for Q3.",
    "The product roadmap details feature releases, platform improvements, and deprecation timelines.",
    "This guide walks new customers through account setup, initial configuration, and support channels.",
    "The framework sets out criteria for evaluating vendors on quality, cost, and delivery performance.",
    "The financial statements include the balance sheet, income statement, and cash flow statement.",
    "Corporate governance policy defines board composition, committee structures, and executive accountability.",
    "The DPIA assesses privacy risks and mitigation measures for the new data processing activity.",
    "Network architecture notes describe the logical topology and security zone structure.",
    "The change management playbook defines roles and responsibilities for organisational change initiatives.",
    "This protocol establishes communication escalation paths during a major organisational crisis.",
    "The training calendar lists all scheduled sessions, facilitators, and registration deadlines.",
    "Employees may claim reimbursement for pre-approved business expenses within 30 days of incurring them.",
    "Workshop notes capture the key themes, decisions, and action items from the annual planning session.",
    "The sustainability report documents environmental performance against the published ESG targets.",
    "The QBR template structures the agenda for quarterly performance reviews with business unit leaders.",
    "The runbook defines step-by-step response procedures for common cybersecurity incident categories.",
    "Budget approval workflows specify authorisation thresholds and required documentation for spend requests.",
    "The risk register captures identified risks, likelihood ratings, impact scores, and mitigation actions.",
    "The compliance checklist covers regulatory and internal policy requirements for the relevant business area.",
    "The stakeholder communication plan maps messages, channels, and frequencies for each audience segment.",
    "This SLA template specifies response times, availability targets, and escalation procedures.",
    "A project charter defines scope, objectives, stakeholders, and governance for a new initiative.",
    "The contractor engagement policy covers statement-of-work requirements, IR35 compliance, and invoicing.",
    "The performance review framework describes goal-setting, mid-year check-ins, and annual calibration.",
    "This knowledge base article provides troubleshooting steps for the described technical issue.",
    "System integration architecture documentation describes API contracts and data exchange protocols.",
    "The procurement request form collects purchase justification, vendor details, and budget code.",
    "This document provides an overview of the legal entities within the corporate group structure.",
    "Brand asset usage guidelines specify approved colours, typefaces, and logo placement rules.",
    "Succession planning identifies critical roles and potential successors across the organisation.",
    "The internal mobility programme enables employees to apply for open roles and short-term project assignments.",
    "The D&I strategy sets multi-year representation targets and inclusion programme commitments.",
    "The innovation lab charter defines governance, funding allocation, and idea-to-pilot process.",
    "Customer feedback from Q4 surveys highlights key satisfaction drivers and areas for improvement.",
    "Office safety procedures cover evacuation routes, first-aid locations, and fire warden responsibilities.",
    "The software licensing inventory lists all enterprise applications, licence counts, and renewal dates.",
    "The updated travel policy reflects new booking procedures, permitted carriers, and reimbursement caps.",
    "Capacity planning assumptions underpin the infrastructure sizing model for the next three years.",
    "Meeting room bookings must be made through the facilities portal at least 24 hours in advance.",
    "The new starter checklist ensures all IT, HR, and compliance onboarding tasks are completed on time.",
    "Exit interviews are conducted by HR within the final week of employment to capture departure feedback.",
    "The benefits enrolment guide explains health, pension, and flexible benefits choices and deadlines.",
    "The archive and records policy defines retention periods, classification, and secure disposal procedures.",
    "The code of conduct summarises expected behaviours regarding integrity, conflicts of interest, and respect.",
    "The whistleblower policy provides confidential channels for reporting concerns about misconduct.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark generator
# ─────────────────────────────────────────────────────────────────────────────

class StructuralNavBenchmark:
    """
    Generates 100 structure-dependent evaluation examples.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.  Default 42.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def generate(self) -> list[dict]:
        """Generate and return all 100 examples."""
        examples: list[dict] = []
        examples.extend(self._generate_discovery(50))
        examples.extend(self._generate_extraction(50))
        return examples

    # ─────────────────────────────────────────────────────────
    # Discovery question generation (50)
    # ─────────────────────────────────────────────────────────

    def _generate_discovery(self, n: int) -> list[dict]:
        """
        Generate n discovery questions from the pool.

        Question pattern: "Which department handles [function_topic]?"
        The answer is the department name, which is also the TITLE of the
        first gold document.  The question uses only the functional description
        — never the department name.
        """
        pool = list(_DISCOVERY_POOL)
        examples: list[dict] = []
        pool_sample = pool[:n]  # Pool has exactly 50 entries

        for idx, entry in enumerate(pool_sample):
            (dept_name, function_topic, answer,
             dept_content, policy_content) = entry

            ex_id = f"disc_{idx:03d}"
            question = f"Which department handles {function_topic}?"

            # Gold documents
            gold_doc_1 = {
                "id": dept_name,
                "title": dept_name,
                "content": dept_content,
            }
            gold_doc_2_id = f"{dept_name}_Policy"
            gold_doc_2 = {
                "id": gold_doc_2_id,
                "title": f"{dept_name} Policy",
                "content": policy_content,
            }

            # 8 distractor documents
            distractors = self._sample_distractors(8, exclude_titles={dept_name, f"{dept_name} Policy"})

            # Shuffle context: gold docs at random positions
            context = [gold_doc_1, gold_doc_2] + distractors
            self._rng.shuffle(context)

            examples.append({
                "id": ex_id,
                "question": question,
                "answer": answer,
                "context": context,
                "gold_ids": [dept_name, gold_doc_2_id],
                "question_type": "discovery",
            })

        return examples

    # ─────────────────────────────────────────────────────────
    # Extraction question generation (50)
    # ─────────────────────────────────────────────────────────

    def _generate_extraction(self, n: int) -> list[dict]:
        """
        Generate n extraction questions from the pool.

        Question pattern: "What is [person_name]'s role in [org_name]?"
        Answer: role_title.
        Gold doc 1: the organisation document (title = org_name)
        Gold doc 2: the person document (title = org_name + " — " + person_name)
        """
        pool = list(_EXTRACTION_POOL)
        examples: list[dict] = []
        pool_sample = pool[:n]  # Pool has exactly 50 entries

        for idx, entry in enumerate(pool_sample):
            (org_name, person_name, role_title, extra_fact,
             org_content, person_content) = entry

            ex_id = f"extr_{idx:03d}"
            question = f"What is {person_name}'s role in {org_name}?"

            # Gold documents
            gold_doc_1_id = org_name
            gold_doc_1 = {
                "id": gold_doc_1_id,
                "title": org_name,
                "content": org_content,
            }
            gold_doc_2_id = f"{org_name} — {person_name}"
            gold_doc_2 = {
                "id": gold_doc_2_id,
                "title": f"{org_name} — {person_name}",
                "content": person_content,
            }

            # 8 distractor documents
            distractors = self._sample_distractors(
                8,
                exclude_titles={org_name, f"{org_name} — {person_name}"},
            )

            context = [gold_doc_1, gold_doc_2] + distractors
            self._rng.shuffle(context)

            examples.append({
                "id": ex_id,
                "question": question,
                "answer": role_title,
                "context": context,
                "gold_ids": [gold_doc_1_id, gold_doc_2_id],
                "question_type": "extraction",
            })

        return examples

    # ─────────────────────────────────────────────────────────
    # Shared helpers
    # ─────────────────────────────────────────────────────────

    def _sample_distractors(self, n: int, exclude_titles: set[str]) -> list[dict]:
        """Sample n distractor documents with unique titles."""
        candidates: list[tuple[str, str]] = [
            (t, c) for t, c in zip(_DISTRACTOR_TITLES, _DISTRACTOR_CONTENTS)
            if t not in exclude_titles
        ]
        self._rng.shuffle(candidates)
        chosen = candidates[:n]
        return [
            {
                "id": f"dist_{title.replace(' ', '_')}",
                "title": title,
                "content": content,
            }
            for title, content in chosen
        ]
