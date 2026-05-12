"""
Fetches Wikipedia articles about IPL and saves them to data/raw/.
Also builds a combined corpus.txt and reports token count.
Target: 500+ articles, 2M+ tokens.
"""

import sys
import io
import time
import tiktoken
from pathlib import Path
from tqdm import tqdm

import wikipediaapi

# Force UTF-8 output so Unicode article titles don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW_DIR, CORPUS_FILE, WIKIPEDIA_DELAY

# ---------------------------------------------------------------------------
# Topic list (~650 entries; unavailable titles are skipped automatically)
# ---------------------------------------------------------------------------
TOPICS = [
    # ── Core ────────────────────────────────────────────────────────────────
    "Indian Premier League",
    "Indian Premier League records and statistics",
    "IPL records and statistics",
    "History of the Indian Premier League",
    "Indian Premier League governing council",
    "Indian Premier League broadcasting rights",
    "Indian Premier League controversies",
    "Lalit Modi",
    "N. Srinivasan",

    # ── Current teams (10) ───────────────────────────────────────────────────
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Lucknow Super Giants",
    "Gujarat Titans",

    # ── Former teams (6) ─────────────────────────────────────────────────────
    "Deccan Chargers",
    "Kochi Tuskers Kerala",
    "Pune Warriors India",
    "Rising Pune Supergiant",
    "Delhi Daredevils",
    "Kings XI Punjab",

    # ── Seasons (2008–2024) ──────────────────────────────────────────────────
    "2008 Indian Premier League",
    "2009 Indian Premier League",
    "2010 Indian Premier League",
    "2011 Indian Premier League",
    "2012 Indian Premier League",
    "2013 Indian Premier League",
    "2014 Indian Premier League",
    "2015 Indian Premier League",
    "2016 Indian Premier League",
    "2017 Indian Premier League",
    "2018 Indian Premier League",
    "2019 Indian Premier League",
    "2020 Indian Premier League",
    "2021 Indian Premier League",
    "2022 Indian Premier League",
    "2023 Indian Premier League",
    "2024 Indian Premier League",

    # ── IPL Finals ──────────────────────────────────────────────────────────
    "2008 Indian Premier League Final",
    "2009 Indian Premier League Final",
    "2010 Indian Premier League Final",
    "2011 Indian Premier League Final",
    "2012 Indian Premier League Final",
    "2013 Indian Premier League Final",
    "2014 Indian Premier League Final",
    "2015 Indian Premier League Final",
    "2016 Indian Premier League Final",
    "2017 Indian Premier League Final",
    "2018 Indian Premier League Final",
    "2019 Indian Premier League Final",
    "2020 Indian Premier League Final",
    "2021 Indian Premier League Final",
    "2022 Indian Premier League Final",
    "2023 Indian Premier League Final",
    "2024 Indian Premier League Final",

    # ── Indian players ──────────────────────────────────────────────────────
    "MS Dhoni",
    "Virat Kohli",
    "Rohit Sharma",
    "Jasprit Bumrah",
    "Suresh Raina",
    "Ravindra Jadeja",
    "KL Rahul",
    "Hardik Pandya",
    "Shubman Gill",
    "Rishabh Pant",
    "Yuzvendra Chahal",
    "Ravichandran Ashwin",
    "Mohammed Shami",
    "Harbhajan Singh",
    "Yuvraj Singh",
    "Sachin Tendulkar",
    "Gautam Gambhir",
    "Virender Sehwag",
    "Zaheer Khan",
    "Dinesh Karthik",
    "Robin Uthappa",
    "Ambati Rayudu",
    "Irfan Pathan",
    "Yusuf Pathan",
    "Ajinkya Rahane",
    "Shreyas Iyer",
    "Ishan Kishan",
    "Sanju Samson",
    "Ruturaj Gaikwad",
    "Prithvi Shaw",
    "Arshdeep Singh",
    "Umran Malik",
    "Avesh Khan",
    "Deepak Chahar",
    "Bhuvneshwar Kumar",
    "T. Natarajan",
    "Axar Patel",
    "Krunal Pandya",
    "Washington Sundar",
    "Mohammed Siraj",
    "Devdutt Padikkal",
    "Tilak Varma",
    "Rinku Singh",
    "Yashasvi Jaiswal",
    "Suryakumar Yadav",
    "Wriddhiman Saha",
    "Parthiv Patel",
    "Manish Pandey",
    "Murali Vijay",
    "Cheteshwar Pujara",
    "Naman Ojha",
    "Piyush Chawla",
    "Pragyan Ojha",
    "Amit Mishra",
    "Stuart Binny",
    "Manoj Tiwary",
    "Rahul Dravid",
    "VVS Laxman",
    "Sourav Ganguly",
    "Anil Kumble",
    "Javagal Srinath",
    "Kapil Dev",
    "Sunil Gavaskar",
    "Ravi Shastri",
    "Ravi Bishnoi",
    "Varun Chakravarthy",
    "Prasidh Krishna",
    "Shivam Dube",
    "Harshal Patel",
    "Shardul Thakur",
    "T Dilshan",
    "S. Sreesanth",
    "Vinay Kumar",
    "Saurabh Tiwary",
    "Dhawal Kulkarni",

    # ── Overseas players ─────────────────────────────────────────────────────
    "Chris Gayle",
    "AB de Villiers",
    "David Warner",
    "Shane Watson",
    "Lasith Malinga",
    "Brendon McCullum",
    "Adam Gilchrist",
    "Ricky Ponting",
    "Kevin Pietersen",
    "Andrew Flintoff",
    "Jacques Kallis",
    "Kieron Pollard",
    "Dwayne Bravo",
    "Dale Steyn",
    "Mitchell Johnson",
    "Shane Warne",
    "Glenn Maxwell",
    "Marcus Stoinis",
    "Aaron Finch",
    "Pat Cummins",
    "Mitchell Starc",
    "Josh Hazlewood",
    "Steve Smith",
    "Kane Williamson",
    "Trent Boult",
    "Tim Southee",
    "Ross Taylor",
    "Martin Guptill",
    "Faf du Plessis",
    "Quinton de Kock",
    "Imran Tahir",
    "Hashim Amla",
    "Rashid Khan",
    "Mohammad Nabi",
    "Jos Buttler",
    "Ben Stokes",
    "Eoin Morgan",
    "Jason Roy",
    "Sam Curran",
    "Chris Jordan",
    "Moeen Ali",
    "Jonny Bairstow",
    "Jofra Archer",
    "Nicholas Pooran",
    "Shimron Hetmyer",
    "Andre Russell",
    "Sunil Narine",
    "Lendl Simmons",
    "Marlon Samuels",
    "JP Duminy",
    "David Miller",
    "Chris Morris",
    "Daniel Vettori",
    "Scott Styris",
    "Daryl Mitchell (cricketer)",
    "Mitchell McClenaghan",
    "Adam Milne",
    "Colin Munro",
    "Devon Conway",
    "Lockie Ferguson",
    "Michael Hussey",
    "Matthew Hayden",
    "Michael Clarke",
    "Brad Hogg",
    "James Faulkner",
    "George Bailey",
    "Nathan Coulter-Nile",
    "Brad Haddin",
    "Ryan ten Doeschate",
    "Ravi Bopara",
    "Albie Morkel",
    "Morne Morkel",
    "Wayne Parnell",
    "Lonwabo Tsotsobe",
    "Robin Peterson",
    "Ryan McLaren",
    "Thisara Perera",
    "Angelo Mathews",
    "Tillakaratne Dilshan",
    "Kumar Sangakkara",
    "Mahela Jayawardene",
    "Muttiah Muralitharan",
    "Chamara Silva",
    "Nuwan Kulasekara",
    "Sanath Jayasuriya",
    "Shahid Afridi",
    "Shoaib Akhtar",
    "Mohammad Hafeez",
    "Umar Gul",
    "Sohail Tanvir",
    "Kamran Akmal",
    "Saeed Ajmal",
    "Wahab Riaz",
    "Mustafizur Rahman",
    "Shakib Al Hasan",
    "Tamim Iqbal",
    "Luke Wright",
    "Dirk Nannes",
    "Dawid Malan",
    "Liam Plunkett",
    "David Willey",
    "Mark Wood",
    "Chris Woakes",
    "Jason Holder",
    "Carlos Brathwaite",
    "Oshane Thomas",
    "Fabian Allen",
    "Jerome Taylor",
    "Rovman Powell",
    "Shai Hope",
    "Kyle Mayers",
    "Brandon King",
    "Obed McCoy",
    "Alzarri Joseph",
    "Odean Smith",
    "Mukesh Choudhary",

    # ── Coaches and officials ────────────────────────────────────────────────
    "Stephen Fleming",
    "Gary Kirsten",
    "Tom Moody",
    "Mike Hesson",
    "John Wright (cricketer)",
    "Wasim Akram",
    "Mahela Jayawardene",
    "Simon Katich",
    "Robin Singh (cricketer)",
    "Sridharan Sriram",
    "W. V. Raman",
    "Lalchand Rajput",
    "Chandrakant Pandit",

    # ── Venues ──────────────────────────────────────────────────────────────
    "Wankhede Stadium",
    "M. A. Chidambaram Stadium",
    "Eden Gardens",
    "M. Chinnaswamy Stadium",
    "Narendra Modi Stadium",
    "Rajiv Gandhi International Cricket Stadium",
    "Arun Jaitley Stadium",
    "Punjab Cricket Association Stadium",
    "Sawai Mansingh Stadium",
    "Ekana Cricket Stadium",
    "DY Patil Stadium",
    "Maharashtra Cricket Association Stadium",
    "Brabourne Stadium",
    "Subrata Roy Sahara Stadium",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "Holkar Cricket Stadium",
    "JSCA International Cricket Stadium",
    "Himachal Pradesh Cricket Association Stadium",
    "Sheikh Zayed Cricket Stadium",
    "Dubai International Cricket Stadium",
    "Sharjah Cricket Stadium",
    "Green Park Stadium",
    "Barabati Stadium",
    "Vidarbha Cricket Association Stadium",

    # ── Awards and records ──────────────────────────────────────────────────
    "Orange Cap",
    "Purple Cap",
    "IPL Most Valuable Player",
    "Emerging Player of the Year Award (IPL)",
    "Fair Play Award (IPL)",

    # ── Governing bodies and related topics ─────────────────────────────────
    "Board of Control for Cricket in India",
    "Decision Review System",
    "Twenty20 cricket",
    "T20 International",
    "Champions League Twenty20",
    "2013 Indian Premier League spot-fixing controversy",
    "2010 Indian Premier League corruption scandal",
    "Duckworth–Lewis–Stern method",
    "Power play (cricket)",
    "Super over",
    "No-ball (cricket)",
    "Wide (cricket)",
    "Free hit",
    "Cricket field",
    "Indian cricket team",
    "ICC World Twenty20",
    "ICC T20 World Cup 2007",
    "ICC T20 World Cup 2009",
    "ICC T20 World Cup 2010",
    "ICC T20 World Cup 2012",
    "ICC T20 World Cup 2014",
    "ICC T20 World Cup 2016",
    "ICC Men's T20 World Cup 2021",
    "ICC Men's T20 World Cup 2022",
    "ICC Men's T20 World Cup 2024",
    "Cricket Australia",
    "England and Wales Cricket Board",
    "Cricket South Africa",
    "Sri Lanka Cricket",
    "Bangladesh Cricket Board",
    "West Indies Cricket Board",
    "Afghanistan Cricket Board",
    "Pakistan Cricket Board",

    # ── Tournaments and cups ─────────────────────────────────────────────────
    "Vijay Hazare Trophy",
    "Deodhar Trophy",
    "Duleep Trophy",
    "Syed Mushtaq Ali Trophy",
    "Ranji Trophy",

    # ── More players (depth) ─────────────────────────────────────────────────
    "Nitish Rana",
    "Rahul Tripathi",
    "Venkatesh Iyer",
    "Harbhajan Singh",
    "Zaheer Khan",
    "Munaf Patel",
    "Ashish Nehra",
    "RP Singh",
    "Sreesanth",
    "Rudra Pratap Singh",
    "Dhawal Kulkarni",
    "Siddarth Kaul",
    "Shardul Thakur",
    "Kuldeep Yadav",
    "Kuldeep Sen",
    "Umesh Yadav",
    "Mohit Sharma",
    "Pawan Negi",
    "Kedar Jadhav",
    "Vijay Shankar",
    "Dhruv Jurel",
    "Rajat Patidar",
    "Mayank Agarwal",
    "Karun Nair",
    "Piyush Chawla",
    "Akshay Wakhare",
    "Abhishek Sharma (cricketer)",
    "Riyan Parag",
    "Naman Dhir",
    "Jitesh Sharma",
    "Phil Salt",
    "Rachin Ravindra",
    "Will Jacks",
    "Tom Latham",
    "Travis Head",
    "David Payne",
    "Reece Topley",
    "Brydon Carse",
    "Finn Allen",
    "Michael Bracewell",
    "Tim David",
    "Romario Shepherd",
    "Akeal Hosein",
    "Gudakesh Motie",
    "Hayden Walsh Jr.",
    "Nkrumah Bonner",
    "Keemo Paul",
    "Sherfane Rutherford",
    "Dominic Drakes",
    "Matheesha Pathirana",
    "Dushmantha Chameera",
    "Wanindu Hasaranga",
    "Chamika Karunaratne",
    "Kusal Mendis",
    "Niroshan Dickwella",
    "Charith Asalanka",
    "Dasun Shanaka",
    "Bhanuka Rajapaksa",
    "Lahiru Kumara",
    "Gerald Coetzee",
    "Kagiso Rabada",
    "Anrich Nortje",
    "Lungi Ngidi",
    "Marco Jansen",
    "Rassie van der Dussen",
    "Heinrich Klaasen",
    "Aiden Markram",
    "Dewald Brevis",
    "Ryan Rickelton",
    "Tristan Stubbs",
    "Dwaine Pretorius",
    "Nandre Burger",
    "Tabraiz Shamsi",
    "Keshav Maharaj",

    # ── IPL auction and operations ────────────────────────────────────────────
    "Indian Premier League auction",
    "IPL 2022 mega auction",
    "IPL 2024 auction",
    "Retention (Indian Premier League)",

    # ── Historical cricket topics ─────────────────────────────────────────────
    "Cricket World Cup",
    "1983 Cricket World Cup",
    "2007 Cricket World Cup",
    "2011 Cricket World Cup",
    "2015 Cricket World Cup",
    "2019 Cricket World Cup",
    "2023 Cricket World Cup",
    "Test cricket",
    "One Day International",
    "History of cricket",

    # ── Additional IPL season / match detail pages ───────────────────────────
    "2008 Indian Premier League season",
    "2009 Indian Premier League season",
    "2010 Indian Premier League season",
    "2011 Indian Premier League season",
    "2012 Indian Premier League season",
    "2013 Indian Premier League season",
    "2014 Indian Premier League season",
    "2015 Indian Premier League season",
    "2016 Indian Premier League season",
    "2017 Indian Premier League season",
    "2018 Indian Premier League season",
    "2019 Indian Premier League season",
    "2020 Indian Premier League season",
    "2021 Indian Premier League season",
    "2022 Indian Premier League season",
    "2023 Indian Premier League season",
    "2024 Indian Premier League season",
    "Indian Premier League playoffs",
    "Indian Premier League points table",

    # ── Cricket legends (rich biography pages) ───────────────────────────────
    "Brian Lara",
    "Viv Richards",
    "Clive Lloyd",
    "Ian Botham",
    "Imran Khan (cricketer)",
    "Waqar Younis",
    "Wasim Akram",
    "Curtly Ambrose",
    "Glenn McGrath",
    "Brett Lee",
    "Andrew Symonds",
    "Darren Sammy",
    "Shikhar Dhawan",
    "Allan Donald",
    "Shaun Pollock",
    "Jonty Rhodes",
    "Mark Boucher",
    "Graeme Smith",
    "AB de Villiers cricket",
    "Mike Hussey cricket",
    "Michael Bevan",
    "Ricky Ponting cricket",
    "Matthew Hayden cricket",
    "Adam Gilchrist cricket",
    "Justin Langer",
    "Damien Martyn",
    "Andrew McDonald (cricketer)",
    "Shane Bond",
    "Chris Cairns",
    "Jacob Oram",
    "Shoaib Malik",
    "Misbah-ul-Haq",
    "Younis Khan",
    "Mohammad Yousuf",
    "Saeed Anwar",
    "Inzamam-ul-Haq",
    "Javed Miandad",
    "Zaheer Abbas",

    # ── Team owners and management ───────────────────────────────────────────
    "Mukesh Ambani",
    "Nita Ambani",
    "Shah Rukh Khan",
    "Juhi Chawla",
    "Jay Mehta",
    "Preity Zinta",
    "Ness Wadia",
    "Shilpa Shetty",
    "Raj Kundra",
    "Manoj Badale",
    "Ranjit Barthakur",
    "Sanjiv Goenka",
    "Sun Group",
    "India Cements",
    "Reliance Industries",

    # ── IPL auction articles ─────────────────────────────────────────────────
    "2022 Indian Premier League auction",
    "2023 Indian Premier League auction",
    "2024 Indian Premier League auction",
    "2018 Indian Premier League auction",
    "2019 Indian Premier League auction",
    "2020 Indian Premier League auction",
    "IPL Player Retention",
    "Indian Premier League auction",
    "Right to Match card",

    # ── Global T20 leagues (context for comparisons) ─────────────────────────
    "Big Bash League",
    "Caribbean Premier League",
    "Pakistan Super League",
    "SA20",
    "Hundred (cricket competition)",
    "Lanka Premier League",
    "Bangladesh Premier League",
    "Global T20 Canada",
    "Major League Cricket",
    "T20 Blast",

    # ── Major bilateral / multi-team series ──────────────────────────────────
    "The Ashes",
    "Border–Gavaskar Trophy",
    "ICC Champions Trophy",
    "Nidahas Trophy",
    "Asia Cup",
    "ICC World Test Championship",
    "Triangular series (cricket)",
    "ICC Knockout Trophy",

    # ── More venues ──────────────────────────────────────────────────────────
    "Feroz Shah Kotla",
    "Buffalo Park",
    "Newlands Cricket Ground",
    "Wanderers Stadium",
    "Kingsmead Cricket Ground",
    "SuperSport Park",
    "St George's Park Cricket Ground",
    "Centurion Park",
    "Lord's Cricket Ground",
    "The Oval",
    "Old Trafford cricket ground",
    "Headingley Stadium",
    "Melbourne Cricket Ground",
    "Sydney Cricket Ground",
    "Adelaide Oval",
    "WACA Ground",
    "Gabba",
    "Bellerive Oval",

    # ── Token-dense Wikipedia list articles ──────────────────────────────────
    "List of Indian Premier League records",
    "List of Indian Premier League centuries",
    "List of Indian Premier League winners",
    "List of cricketers by Test debut for India",
    "List of international cricket centuries by Virat Kohli",
    "List of international cricket centuries by Rohit Sharma",
    "List of international cricket centuries by MS Dhoni",
    "List of cricket grounds in India",
    "List of cricketers by country represented",
    "List of international cricket stadiums",
    "List of highest individual scores in Twenty20 cricket",
    "List of Twenty20 International records",
    "List of most wickets in Twenty20 International cricket",
    "List of IPL Orange Cap winners",
    "List of IPL Purple Cap winners",
    "List of fastest centuries in IPL",
    "List of most sixes in IPL",
    "List of T20 cricket records",
    "List of Indian cricketers",
    "List of Pakistani cricketers",
    "List of Australian cricketers",
    "List of West Indian cricketers",

    # ── Cricket rules, formats, and administration ────────────────────────────
    "Laws of Cricket",
    "Cricket scoring and statistics",
    "Batting (cricket)",
    "Bowling (cricket)",
    "Fielding (cricket)",
    "Wicket-keeper",
    "Cricket statistics",
    "Run rate (cricket)",
    "Net run rate",
    "Economy rate (cricket)",
    "Strike rate (cricket)",
    "Cricket ball tampering",
    "Spirit of Cricket",
    "International Cricket Council",
    "Cricket West Indies",
    "New Zealand Cricket",
    "Zimbabwe Cricket",
    "Netherlands cricket team",
    "Umpire (cricket)",
    "Third umpire",
    "Hot Spot (cricket)",
    "Hawk-Eye",
    "Snickometer",

    # ── Iconic IPL moments and controversies ─────────────────────────────────
    "IPL spot-fixing scandal",
    "2015 Indian Premier League corruption",
    "Chennai Super Kings controversies",
    "IPL 2013 spot-fixing controversy",
    "Lalit Modi Twitter controversy",

    # ── Additional Indian domestic cricket ───────────────────────────────────
    "Irani Cup",
    "Board President's XI",
    "India A cricket team",
    "India Under-19 cricket team",
    "National Cricket Academy",
    "Indian Premier League Development",
    "BCCI Anti-Corruption Unit",

    # ── Long cricket history and overview articles ────────────────────────────
    "Cricket in India",
    "History of cricket in India",
    "Cricket",
    "History of cricket",
    "History of cricket to 1725",
    "Cricket in England",
    "Cricket in Australia",
    "Cricket in South Africa",
    "Cricket in the West Indies",
    "Cricket in Pakistan",
    "Cricket in Sri Lanka",
    "Cricket in New Zealand",
    "Cricket in Bangladesh",
    "Cricket in Zimbabwe",
    "Cricket in Afghanistan",
    "Women's cricket",
    "Women's Indian Premier League",

    # ── More complete player biographies (long articles) ─────────────────────
    "Sachin Tendulkar",
    "Sachin Tendulkar international cricket career",
    "Sachin Tendulkar biography",
    "List of international centuries by Sachin Tendulkar",
    "MS Dhoni career statistics",
    "Virat Kohli career statistics",
    "Rohit Sharma career statistics",
    "David Warner cricket",
    "Steve Waugh",
    "Mark Taylor (cricketer)",
    "Richie Benaud",
    "Don Bradman",
    "Garfield Sobers",
    "Wasim Akram career",
    "Muttiah Muralitharan",
    "Shane Warne cricket",
    "Shane Warne bowling career",
    "Jacques Kallis cricket",
    "Brian Lara cricket",
    "Ricky Ponting cricket career",
    "Allan Border",
    "Greg Chappell",
    "Ian Chappell",
    "Barry Richards",
    "Graeme Pollock",
    "Mike Procter",
    "Sunil Gavaskar career",
    "Kapil Dev cricket career",
    "Bishan Singh Bedi",
    "Erapalli Prasanna",
    "B. S. Chandrasekhar",
    "Srinivas Venkataraghavan",

    # ── Specifically long IPL / T20 tournament articles ───────────────────────
    "Indian Premier League financial aspects",
    "Indian Premier League team owners",
    "Indian Premier League viewership",
    "Indian Premier League statistics",
    "IPL 2008 playoffs",
    "2008 Indian Premier League group stage",
    "IPL 2023 season",
    "IPL 2022 season",
    "Mumbai Indians season 2023",
    "Chennai Super Kings 2023 season",
    "Royal Challengers Bengaluru season 2023",
    "Kolkata Knight Riders 2024 IPL season",

    # ── Wikipedia "Indian Premier League" sub-articles ────────────────────────
    "Indian Premier League 2008",
    "Indian Premier League 2009",
    "Indian Premier League 2010",
    "Indian Premier League 2011",
    "Indian Premier League 2012",
    "Indian Premier League 2013",
    "Indian Premier League 2014",
    "Indian Premier League 2015",
    "Indian Premier League 2016",
    "Indian Premier League 2017",
    "Indian Premier League 2018",
    "Indian Premier League 2019",
    "Indian Premier League 2020",
    "Indian Premier League 2021",
    "Indian Premier League 2022",
    "Indian Premier League 2023",
    "Indian Premier League 2024",

    # ── Records and statistics (alternate titles) ─────────────────────────────
    "Indian Premier League all-time records",
    "List of most runs in IPL",
    "List of most wickets in IPL",
    "List of highest team totals in IPL",
    "List of lowest team totals in IPL",
    "Twenty20 cricket records",
    "List of T20 records",
    "ICC rankings",
    "Wisden Cricketers of the Year",
    "Wisden Leading Cricketer in the World",
    "ICC Cricketer of the Year",
    "ICC Men's ODI Cricketer of the Year",
    "ICC Men's Test Cricketer of the Year",
    "Arjuna Award",
    "Rajiv Gandhi Khel Ratna",
    "Padma Bhushan",

    # ── Cricket formats and techniques ───────────────────────────────────────
    "T20 cricket batting techniques",
    "Yorker",
    "Bouncer (cricket)",
    "Googly",
    "Leg spin",
    "Off spin",
    "Swing bowling",
    "Reverse swing",
    "Doosra",
    "Carrom ball",
    "Slow left-arm orthodox",
    "Wrist spin",
    "Fast bowling",
    "Batting average",
    "Bowling average",

    # ── More overseas players with long pages ─────────────────────────────────
    "Kumar Sangakkara career",
    "Mahela Jayawardene career",
    "Sanath Jayasuriya career",
    "Arjuna Ranatunga",
    "Chaminda Vaas",
    "Muttiah Muralitharan bowling career",
    "Lasith Malinga cricket career",
    "Chris Gayle cricket career",
    "Shivnarine Chanderpaul",
    "Carl Hooper",
    "Brian Lara career statistics",
    "Courtney Walsh",
    "Joel Garner",
    "Malcolm Marshall",
    "Michael Holding",
    "Andy Roberts (cricketer)",
    "Joel Garner",
    "Gordon Greenidge",
    "Desmond Haynes",
    "Alvin Kallicharran",
    "Rohan Kanhai",
    "Lance Gibbs",
    "Wes Hall",
    "Sir Frank Worrell",
    "Sir Everton Weekes",
    "Sir Clyde Walcott",

    # ── IPL-adjacent competitions ─────────────────────────────────────────────
    "Pepsi Indian Premier League",
    "DLF Indian Premier League",
    "VIVO Indian Premier League",
    "Tata Indian Premier League",
    "Dream11 Indian Premier League",
    "Champions League Twenty20 2009",
    "Champions League Twenty20 2010",
    "Champions League Twenty20 2011",
    "Champions League Twenty20 2012",
    "Champions League Twenty20 2013",
    "Champions League Twenty20 2014",

    # ── Classic England legends (long Wikipedia bios) ─────────────────────────
    "Geoffrey Boycott",
    "David Gower",
    "Mike Brearley",
    "Tony Greig",
    "Derek Underwood",
    "Bob Willis",
    "Fred Trueman",
    "Jim Laker",
    "Peter May (cricketer)",
    "Denis Compton",
    "Len Hutton",
    "Alec Bedser",
    "Cyril Washbrook",
    "Tom Graveney",
    "Colin Cowdrey",
    "Ted Dexter",
    "Ray Illingworth",
    "Brian Close",
    "John Edrich",
    "Dennis Amiss",
    "Graham Gooch",
    "David Gower cricket career",
    "Alastair Cook",
    "Kevin Pietersen cricket career",

    # ── Australian legends (long bios) ───────────────────────────────────────
    "Dennis Lillee",
    "Jeff Thomson",
    "Kim Hughes",
    "Rod Marsh",
    "Doug Walters",
    "Bill Lawry",
    "Bob Simpson (cricketer)",
    "Neil Harvey",
    "Keith Miller",
    "Ray Lindwall",
    "Bill Johnston (cricketer)",
    "Alan Davidson (cricketer)",
    "Richie Benaud career",
    "Lindsay Hassett",
    "Sid Barnes",
    "Arthur Morris",
    "Bill Brown (cricketer)",
    "Stan McCabe",
    "Clarrie Grimmett",
    "Bill O'Reilly (cricketer)",

    # ── South African legends ─────────────────────────────────────────────────
    "Aubrey Faulkner",
    "Herbie Taylor",
    "Dudley Nourse",
    "Bruce Mitchell (cricketer)",
    "Eric Rowan",
    "Roy McLean (cricketer)",
    "Hugh Tayfield",
    "Neil Adcock",
    "Peter Heine",
    "Ali Bacher",
    "Barry Richards cricket",
    "Lee Irvine",
    "Vincent van der Bijl",

    # ── Indian cricket history ────────────────────────────────────────────────
    "History of cricket in India",
    "Cricket in India before independence",
    "Indian cricket team in the 1990s",
    "Indian cricket team in the 2000s",
    "Indian cricket team in the 2010s",
    "Indian cricket team records",
    "India national cricket team",
    "India national cricket team captains",
    "Indian cricket team in World Cups",
    "BCCI history",

    # ── Detailed tournament articles ──────────────────────────────────────────
    "2007 ICC World Twenty20",
    "2009 ICC World Twenty20",
    "2010 ICC World Twenty20",
    "2012 ICC World Twenty20",
    "2014 ICC World Twenty20",
    "2016 ICC World Twenty20",
    "2021 ICC Men's T20 World Cup",
    "2022 ICC Men's T20 World Cup",
    "2024 ICC Men's T20 World Cup",
    "1983 Cricket World Cup",
    "1987 Cricket World Cup",
    "1992 Cricket World Cup",
    "1996 Cricket World Cup",
    "1999 Cricket World Cup",
    "2003 Cricket World Cup",
    "2007 Cricket World Cup",
    "2011 Cricket World Cup",
    "2015 Cricket World Cup",
    "2019 Cricket World Cup",
    "2023 Cricket World Cup",
    "ICC Champions Trophy 2002",
    "ICC Champions Trophy 2004",
    "ICC Champions Trophy 2006",
    "ICC Champions Trophy 2009",
    "ICC Champions Trophy 2013",
    "ICC Champions Trophy 2017",

    # ── Cricket records pages (alternate spellings) ───────────────────────────
    "List of One Day International cricket records",
    "List of Test cricket records",
    "List of Twenty20 International cricket records",
    "List of highest individual scores in Test cricket",
    "List of most runs in ODI cricket",
    "List of most wickets in Test cricket",
    "List of most wickets in ODI cricket",
    "List of cricket records",
]

# De-duplicate while preserving order
seen = set()
TOPICS_DEDUPED = []
for t in TOPICS:
    if t.lower() not in seen:
        seen.add(t.lower())
        TOPICS_DEDUPED.append(t)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_filename(title: str) -> str:
    """Turn a Wikipedia title into a safe filename."""
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip()[:100]


def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def fetch_article(wiki: wikipediaapi.Wikipedia, title: str) -> tuple[str, int]:
    """Return (text, token_count) or ("", 0) if not found."""
    page = wiki.page(title)
    if not page.exists():
        return "", 0
    text = page.text.strip()
    if len(text) < 200:  # skip stubs
        return "", 0
    return text, count_tokens(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)

    wiki = wikipediaapi.Wikipedia(
        user_agent="GraphRAG-Hackathon/1.0 (research@graphrag-hackathon.com)",
        language="en",
    )

    saved = 0
    skipped = 0
    total_tokens = 0
    corpus_parts: list[str] = []

    print(f"Fetching up to {len(TOPICS_DEDUPED)} Wikipedia articles...\n")

    for title in tqdm(TOPICS_DEDUPED, unit="article"):
        out_path = DATA_RAW_DIR / f"{sanitize_filename(title)}.txt"

        # Use cached file if it already exists
        if out_path.exists():
            cached = out_path.read_text(encoding="utf-8")
            tokens = count_tokens(cached)
            corpus_parts.append(f"\n\n{'='*60}\nARTICLE: {title}\n{'='*60}\n{cached}")
            total_tokens += tokens
            saved += 1
            continue

        text, tokens = fetch_article(wiki, title)
        if not text:
            skipped += 1
            time.sleep(WIKIPEDIA_DELAY)
            continue

        out_path.write_text(text, encoding="utf-8")
        corpus_parts.append(f"\n\n{'='*60}\nARTICLE: {title}\n{'='*60}\n{text}")
        total_tokens += tokens
        saved += 1

        tqdm.write(f"  [OK] {title:<55} {tokens:>8,} tokens")
        time.sleep(WIKIPEDIA_DELAY)

    # Write corpus
    print("\nWriting corpus.txt ...")
    CORPUS_FILE.write_text("\n".join(corpus_parts), encoding="utf-8")

    print()
    print("=" * 56)
    print("  Dataset collection complete")
    print(f"  Articles saved  : {saved}")
    print(f"  Articles skipped: {skipped}  (not found / stubs)")
    print(f"  Total tokens    : {total_tokens:,}")
    print(f"  Corpus file     : {CORPUS_FILE}")
    print("=" * 56)
    print()

    if total_tokens < 2_000_000:
        print(f"[WARN] Token count {total_tokens:,} is below 2M target.")
        print("       Re-run after adding more topics to TOPICS list.")
    else:
        print(f"[OK] Target reached: {total_tokens:,} tokens (>= 2M)")


if __name__ == "__main__":
    main()
