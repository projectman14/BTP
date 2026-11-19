"""Synthetic long-term conversational memory benchmark (LoCoMo / LongMemEval style).

Each simulated user talks to an assistant over many sessions. Personal facts are
dropped into the conversation once, surrounded by small talk and by *distractor*
facts about other people that reuse the same vocabulary. Some facts are later
updated (temporal reasoning), some require linking two memories through a named
entity (multi-hop), and some questions are unanswerable (abstention test).

Usage: python benchmarks/make_dataset.py --users 20 --sessions 12 --pairs 12 --out data/longmem_synth.json
"""
import argparse
import json
import random
from datetime import date, timedelta

CITIES = ["Jaipur", "Pune", "Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad",
          "Lucknow", "Indore", "Bhopal", "Chandigarh", "Kochi", "Surat", "Nagpur", "Dehradun", "Udaipur",
          "Jodhpur", "Mysuru", "Goa", "Shimla"]
NAMES = ["Priya", "Rohan", "Anita", "Kabir", "Meera", "Arjun", "Sneha", "Vikram", "Neha", "Aditya", "Ishita",
         "Karan", "Pooja", "Rahul", "Tanvi", "Dev", "Riya", "Sameer", "Ananya", "Nikhil", "Simran", "Yash",
         "Divya", "Manav", "Kavya", "Harsh", "Nandini", "Parth", "Shreya", "Varun"]
PETS = ["Bruno", "Milo", "Coco", "Simba", "Leo", "Oreo", "Max", "Bella", "Luna", "Toffee", "Pepper", "Ginger",
        "Shadow", "Rocky", "Zoe", "Biscuit", "Nala", "Buddy", "Momo", "Snowy"]
JOBS = ["data analyst", "software engineer", "civil engineer", "graphic designer", "school teacher", "pharmacist",
        "architect", "chartered accountant", "product manager", "UX designer", "mechanical engineer",
        "content writer", "nurse", "lawyer", "dentist", "journalist"]
COMPANIES = ["Infosys", "Zomato", "Wipro", "Razorpay", "Swiggy", "Flipkart", "Freshworks", "Tata Motors", "Paytm",
             "Ola", "Nykaa", "PhonePe", "Zoho", "Meesho", "CRED", "Groww"]
FOODS = ["biryani", "dal makhani", "masala dosa", "chole bhature", "pav bhaji", "butter chicken", "rajma chawal",
         "paneer tikka", "pani puri", "momos", "thepla", "litti chokha", "vada pav", "idli sambar"]
CARS = ["Maruti Swift", "Hyundai Creta", "Tata Nexon", "Honda City", "Kia Seltos", "Mahindra Thar",
        "Toyota Innova", "Mahindra XUV700", "Tata Punch", "Renault Kwid", "Skoda Slavia", "MG Hector"]
SPORTS = ["badminton", "cricket", "football", "table tennis", "basketball", "squash", "volleyball", "tennis",
          "chess", "kabaddi"]
ALLERGIES = ["peanuts", "shellfish", "dust", "pollen", "gluten", "lactose", "soy", "eggs", "wheat", "sesame"]
BIRTHDAYS = ["14 March", "2 August", "23 November", "9 January", "30 June", "17 September", "5 May",
             "21 December", "12 February", "28 October", "3 April", "19 July"]
LANGUAGES = ["Spanish", "Japanese", "German", "French", "Korean", "Italian", "Mandarin", "Portuguese"]
DOCTORS = ["Mehta", "Kapoor", "Iyer", "Bose", "Nair", "Reddy", "Saxena", "Chawla", "Desai", "Pillai"]
LAPTOPS = ["ThinkPad X1 Carbon", "MacBook Air M3", "Dell XPS 13", "HP Pavilion 15", "Asus ZenBook 14",
           "Acer Swift Go", "Lenovo Yoga Slim 7", "Samsung Galaxy Book4"]
BOOKS = ["Sapiens", "Atomic Habits", "The Alchemist", "Dune", "Ikigai", "The Psychology of Money", "Deep Work",
         "Project Hail Mary", "The Midnight Library", "Educated", "Shoe Dog", "The Martian"]
GYMS = ["Cult Fit", "Gold's Gym", "Anytime Fitness", "Snap Fitness", "Talwalkars", "Fitness First"]
PHONES = ["Pixel 8", "iPhone 15", "OnePlus 12", "Samsung Galaxy S24", "Nothing Phone 2", "Xiaomi 14", "Vivo X100"]
TEAMS = ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru", "Rajasthan Royals",
         "Kolkata Knight Riders", "Delhi Capitals", "Manchester United", "Liverpool", "Real Madrid", "Barcelona",
         "Arsenal", "Gujarat Titans"]
DATES = ["14 March", "30 June", "5 September", "21 November", "18 December", "9 February", "27 April", "15 July",
         "3 October", "11 January", "25 August", "8 May"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
INSTRUMENTS = ["guitar", "tabla", "piano", "violin", "flute", "ukulele"]
COLLEGES = ["IIT Delhi", "LNMIIT Jaipur", "BITS Pilani", "NIT Trichy", "IIIT Hyderabad", "VIT Vellore",
            "Manipal", "DTU"]

# --------------------------------------------------------------------------------------
# attribute schema: statement paraphrases, question paraphrases, value pool
# --------------------------------------------------------------------------------------
ATTRS = {
    "city": dict(values=CITIES,
                 stmts=["I live in {v}.", "I'm based in {v} these days.", "Home for me is {v}, by the way."],
                 qs=["Where does the user live?", "Which city is the user based in?", "What city does the user call home?"],
                 update=["I moved to {v} last month, so I'm living in {v} now.",
                         "Quick update: I have relocated to {v}.", "I shifted to {v} recently for work."],
                 q_current=["Where does the user currently live?", "Which city does the user live in now?"],
                 q_before=["Where did the user live before moving?", "Which city was the user based in earlier?"]),
    "pet": dict(values=PETS,
                stmts=["I have a dog named {v}.", "My dog {v} chewed my shoes again.", "My puppy is called {v}."],
                qs=["What is the name of the user's dog?", "What is the user's dog called?", "How is the user's dog named?"]),
    "job": dict(values=JOBS,
                stmts=["I work as a {v}.", "By profession I am a {v}.", "My day job is being a {v}."],
                qs=["What is the user's profession?", "What does the user do for a living?", "What is the user's job?"]),
    "company": dict(values=COMPANIES,
                    stmts=["I joined {v} last year.", "I work at {v}.", "These days I'm employed at {v}."],
                    qs=["Which company does the user work for?", "Where is the user employed?", "What is the user's employer?"],
                    update=["I switched jobs and now I work at {v}.", "I've just joined {v}, leaving my old company.",
                            "Career update: I'm at {v} now."],
                    q_current=["Which company does the user currently work for?", "Where does the user work now?"],
                    q_before=["Which company did the user work for before switching?", "What was the user's previous employer?"]),
    "food": dict(values=FOODS,
                 stmts=["My favourite food is {v}.", "I could eat {v} every single day, it's my favourite.",
                        "Nothing beats {v} for me, it's my number one dish."],
                 qs=["What is the user's favourite food?", "Which dish does the user like the most?", "What food does the user love most?"]),
    "car": dict(values=CARS,
                stmts=["I drive a {v}.", "My car is a {v}.", "I bought a {v} two years ago and still drive it."],
                qs=["What car does the user drive?", "Which vehicle does the user own?", "What is the user's car?"]),
    "sport": dict(values=SPORTS,
                  stmts=["I play {v} every weekend.", "Weekends are for {v} for me.", "My weekend sport is {v}."],
                  qs=["Which sport does the user play on weekends?", "What sport does the user play?", "What game does the user play every weekend?"]),
    "allergy": dict(values=ALLERGIES,
                    stmts=["I am allergic to {v}, so I avoid it.", "Please remember I have a {v} allergy.",
                           "{v} gives me an allergic reaction."],
                    qs=["What is the user allergic to?", "Which allergy does the user have?", "What should the user avoid because of an allergy?"]),
    "birthday": dict(values=BIRTHDAYS,
                     stmts=["My birthday is on {v}.", "I was born on {v}.", "I celebrate my birthday every {v}."],
                     qs=["When is the user's birthday?", "On which date was the user born?", "What is the user's date of birth?"]),
    "language": dict(values=LANGUAGES,
                     stmts=["I'm learning {v} on Duolingo.", "I have started studying {v}.", "I practise {v} for twenty minutes daily."],
                     qs=["Which language is the user learning?", "What language is the user studying?", "Which new language is the user picking up?"]),
    "doctor": dict(values=DOCTORS,
                   stmts=["My doctor, Dr. {v}, told me to cut down on sugar.", "I have an appointment with Dr. {v} next week.",
                          "Dr. {v} is my family physician."],
                   qs=["What is the name of the user's doctor?", "Who is the user's physician?", "Which doctor does the user consult?"]),
    "laptop": dict(values=LAPTOPS,
                   stmts=["I just bought a {v} laptop.", "My new laptop is a {v}.", "I work on a {v}."],
                   qs=["Which laptop did the user buy?", "What laptop does the user use?", "What is the model of the user's laptop?"]),
    "book": dict(values=BOOKS,
                 stmts=["I'm currently reading {v}.", "The book on my nightstand right now is {v}.", "I started {v} this week."],
                 qs=["Which book is the user currently reading?", "What is the user reading these days?", "Which book did the user start recently?"]),
    "gym": dict(values=GYMS,
                stmts=["I go to {v} gym near my place.", "I train at {v}.", "My gym membership is with {v}."],
                qs=["Which gym does the user go to?", "Where does the user work out?", "What is the name of the user's gym?"]),
    "phone": dict(values=PHONES,
                  stmts=["I use a {v} phone.", "My phone is a {v}.", "I'm on a {v} right now."],
                  qs=["Which phone does the user use?", "What is the user's phone model?", "What smartphone does the user have?"],
                  update=["I upgraded my phone to a {v}.", "Got a new phone, it's a {v}.", "I replaced my old phone with a {v}."],
                  q_current=["Which phone does the user use now?", "What is the user's current phone?"],
                  q_before=["Which phone did the user use before upgrading?", "What was the user's old phone?"]),
    "team": dict(values=TEAMS,
                 stmts=["I support {v}.", "I'm a die-hard {v} fan.", "{v} is my team, always."],
                 qs=["Which team does the user support?", "What is the user's favourite team?", "Which team is the user a fan of?"]),
    "instrument": dict(values=INSTRUMENTS,
                       stmts=["I play the {v} in my free time.", "I've been learning the {v} for a year.", "My instrument is the {v}."],
                       qs=["Which musical instrument does the user play?", "What instrument is the user learning?", "What does the user play musically?"]),
    "college": dict(values=COLLEGES,
                    stmts=["I studied at {v}.", "I graduated from {v}.", "{v} is my alma mater."],
                    qs=["Where did the user study?", "Which college did the user graduate from?", "What is the user's alma mater?"]),
}

# multi-hop chains: fact1 introduces a related person; fact2 states something about that person
CHAINS = [
    dict(name="sister_cat", values=PETS,
         s1=["My sister {n} visited me last weekend.", "My sister {n} is coming over for dinner.", "{n}, my sister, called me today."],
         s2=["{n} adopted a cat and named it {v}.", "{n} finally got a cat, it's called {v}.", "The cat {n} adopted is named {v}."],
         q=["What is the name of the cat adopted by the user's sister?", "What did the user's sister name her cat?"]),
    dict(name="friend_city", values=CITIES,
         s1=["My college friend {n} and I talk every week.", "{n} is my closest friend from college.", "I met my college friend {n} yesterday."],
         s2=["{n} just moved to {v} for a new job.", "{n} has shifted to {v}.", "{n} is now living in {v}."],
         q=["Which city did the user's college friend move to?", "Where does the user's college friend live now?"]),
    dict(name="manager_deadline", values=DATES,
         s1=["My manager {n} is quite strict about timelines.", "{n} is my manager at work.", "I report to {n}, my manager."],
         s2=["{n} set the project deadline for {v}.", "{n} wants the project delivered by {v}.", "The deadline {n} gave us is {v}."],
         q=["What deadline did the user's manager set for the project?", "By when does the user's manager want the project delivered?"]),
    dict(name="cousin_wedding", values=CITIES,
         s1=["My cousin {n} got engaged last month.", "{n} is my cousin and we grew up together.", "My cousin {n} is getting married."],
         s2=["{n}'s wedding will be in {v}.", "The wedding of {n} is happening in {v}.", "{n} is getting married in {v}."],
         q=["In which city is the user's cousin's wedding?", "Where will the wedding of the user's cousin take place?"]),
    dict(name="brother_company", values=COMPANIES,
         s1=["My brother {n} is two years younger than me.", "{n} is my younger brother.", "My brother {n} lives with me."],
         s2=["{n} got placed at {v}.", "{n} started working at {v}.", "{n} joined {v} as a fresher."],
         q=["Which company does the user's brother work at?", "Where did the user's brother get placed?"]),
]

DISTRACTORS = [
    ("city", "My colleague {n} lives in {v}."), ("city", "{n} from my office is based in {v}."),
    ("job", "{n} from my team works as a {v}."), ("pet", "My neighbour's dog is called {v}."),
    ("car", "My uncle {n} drives a {v}."), ("allergy", "{n} is allergic to {v}, so we skipped it."),
    ("food", "My roommate's favourite food is {v}."), ("laptop", "{n} just bought a {v} laptop."),
    ("team", "My friend {n} supports {v}."), ("language", "{n} is learning {v} for a trip."),
    ("birthday", "My aunt's birthday is on {v}."), ("company", "{n} works at {v}."),
    ("gym", "{n} trains at {v} every morning."), ("phone", "{n} just switched to a {v}."),
    ("book", "{n} recommended {v} to me."), ("sport", "{n} plays {v} at the club."),
]

FILLER_USER = [
    "Can you explain how recursion works in simple terms?", "Suggest a good movie for tonight.",
    "What is a good way to learn guitar chords quickly?", "Write a two-line poem about rain.",
    "How do I make cold brew coffee at home?", "What's the difference between TCP and UDP?",
    "Give me a simple three-day workout plan.", "Any tips for sleeping better?",
    "How long should I boil eggs for a soft yolk?", "Explain what an API is.",
    "What are some good productivity habits?", "Can you summarise the plot of Inception?",
    "How does compound interest work?", "What's a healthy breakfast idea?",
    "Translate 'good morning' into Hindi.", "Why is the sky blue?",
    "How do I center a div in CSS?", "What is the capital of Australia?",
    "Give me three ideas for a weekend trip.", "How can I improve my public speaking?",
    "What's the best way to store fresh coriander?", "Explain the difference between a list and a tuple in Python.",
    "How many litres of water should I drink a day?", "Recommend a podcast about history.",
    "What does GDP stand for?", "How do I remove a coffee stain from a shirt?",
    "Give me a quick recipe for pancakes.", "What is machine learning in one sentence?",
    "How do I calm down before an exam?", "Suggest a name for my new plant.",
    "What time zone is Tokyo in?", "How do I write a polite follow-up email?",
    "Explain what a black hole is.", "What's a good stretching routine for back pain?",
    "How do I convert Celsius to Fahrenheit?", "Which is better for a beginner, Python or Java?",
    "Tell me a fun fact about octopuses.", "How can I save more money each month?",
    "What's the plural of cactus?", "How do I take better photos with my phone?",
    "Can you help me plan a study timetable?", "What is the Pythagorean theorem?",
    "Give me a motivational quote.", "How do plants make food?",
    "What is the difference between weather and climate?", "How do I make a good cup of chai?",
    "Suggest a board game for four people.", "What are the symptoms of dehydration?",
    "How do I speed up my laptop?", "Explain blockchain like I'm five.",
]
FILLER_ASSISTANT = [
    "Sure! Here is a simple explanation: it breaks the problem into smaller pieces of the same kind.",
    "Great question. A good starting point is to practise a little every day and keep it fun.",
    "Here are a few options you could try, depending on your mood.",
    "That's a nice idea. Let me give you a short step-by-step approach.",
    "Of course. The key points are consistency, patience and keeping things simple.",
    "Happy to help. Here's a quick summary to get you started.",
    "Good one! The short answer is that it depends on your goal, but here is a general guideline.",
    "Absolutely. I would suggest keeping it light and building up gradually.",
    "Here's a concise explanation along with a small example.",
    "Nice. Try this and let me know how it goes.",
]
ACKS = ["Got it, I'll remember that.", "Thanks for sharing, noted!", "Nice, good to know.",
        "That's great! I'll keep that in mind.", "Noted. Anything else you'd like to tell me?",
        "Sounds good!", "Interesting, thanks for letting me know.", "Okay, noted."]


def pick(rng, pool, exclude):
    opts = [x for x in pool if x not in exclude]
    return rng.choice(opts)


def build_user(uid: int, rng: random.Random, n_sessions: int, pairs_per_session: int,
               n_single: int = 10, n_updates: int = 2, n_chains: int = 2, n_distractors: int = 8,
               n_unanswerable: int = 3):
    used_values: set[str] = set()
    used_names: set[str] = set()
    facts = []          # dicts: key, text, (answer), kind

    attr_keys = list(ATTRS.keys())
    rng.shuffle(attr_keys)
    chosen = attr_keys[:n_single]
    unchosen = attr_keys[n_single:]
    updatable = [a for a in chosen if "update" in ATTRS[a]]
    rng.shuffle(updatable)
    updated = updatable[:n_updates]

    questions = []
    # single-hop and temporal facts
    for a in chosen:
        spec = ATTRS[a]
        v = pick(rng, spec["values"], used_values)
        used_values.add(v)
        text = rng.choice(spec["stmts"]).format(v=v)
        text = text[0].upper() + text[1:]
        f1 = dict(key=f"{a}", text=text, kind="single", attr=a, answer=v)
        facts.append(f1)
        if a in updated:
            v2 = pick(rng, spec["values"], used_values)
            used_values.add(v2)
            f2 = dict(key=f"{a}_update", text=rng.choice(spec["update"]).format(v=v2), kind="update",
                      attr=a, answer=v2, after=f1["key"])
            facts.append(f2)
            questions.append(dict(type="temporal_current", attr=a, question=rng.choice(spec["q_current"]),
                                  answer=v2, gold=[f2["key"]], trap=v, answerable=True))
            questions.append(dict(type="temporal_before", attr=a, question=rng.choice(spec["q_before"]),
                                  answer=v, gold=[f1["key"]], trap=v2, answerable=True))
        else:
            questions.append(dict(type="single", attr=a, question=rng.choice(spec["qs"]), answer=v,
                                  gold=[f1["key"]], answerable=True))

    # multi-hop chains
    chains = rng.sample(CHAINS, n_chains)
    for c in chains:
        n = pick(rng, NAMES, used_names)
        used_names.add(n)
        v = pick(rng, c["values"], used_values)
        used_values.add(v)
        f1 = dict(key=f"{c['name']}_1", text=rng.choice(c["s1"]).format(n=n), kind="chain1", attr=c["name"], answer=n)
        f2 = dict(key=f"{c['name']}_2", text=rng.choice(c["s2"]).format(n=n, v=v), kind="chain2", attr=c["name"],
                  answer=v, after=f1["key"])
        facts += [f1, f2]
        questions.append(dict(type="multihop", attr=c["name"], question=rng.choice(c["q"]), answer=v,
                              gold=[f1["key"], f2["key"]], answerable=True))

    # distractors about other people (same vocabulary, different values)
    for i, (a, tmpl) in enumerate(rng.sample(DISTRACTORS, n_distractors)):
        n = pick(rng, NAMES, used_names)
        used_names.add(n)
        v = pick(rng, ATTRS[a]["values"], used_values)
        used_values.add(v)
        facts.append(dict(key=f"distractor_{i}", text=tmpl.format(n=n, v=v), kind="distractor", attr=a, answer=v))

    # unanswerable questions about attributes never mentioned
    for a in unchosen[:n_unanswerable]:
        questions.append(dict(type="unanswerable", attr=a, question=rng.choice(ATTRS[a]["qs"]),
                              answer="I don't know", gold=[], answerable=False))

    # ---- lay facts out over sessions ------------------------------------------------
    n_slots = n_sessions * pairs_per_session
    assert len(facts) < n_slots * 0.6, "too many facts for the number of slots"
    free = set(range(n_slots))
    placement: dict[str, int] = {}
    # predecessors of an 'after' chain are confined to the first 2/3 of the sessions so that a
    # strictly later session always exists for the follow-up fact
    has_follow = {f["after"] for f in facts if "after" in f}
    early_limit = max(1, (2 * n_sessions) // 3) * pairs_per_session
    for f in sorted(facts, key=lambda f: 0 if "after" not in f else 1):
        if "after" in f:
            prev_sess = placement[f["after"]] // pairs_per_session
            cands = [s for s in free if s // pairs_per_session > prev_sess]
        elif f["key"] in has_follow:
            cands = [s for s in free if s < early_limit]
        else:
            cands = list(free)
        s = rng.choice(sorted(cands))
        free.remove(s)
        placement[f["key"]] = s
    slot_to_fact = {s: k for k, s in placement.items()}
    fact_by_key = {f["key"]: f for f in facts}

    start = date(2026, 1, 5) + timedelta(days=7 * uid)
    sessions = []
    turn_ids: dict[str, str] = {}
    for s in range(n_sessions):
        turns = []
        for p in range(pairs_per_session):
            slot = s * pairs_per_session + p
            tid = f"s{s + 1}_t{len(turns)}"
            if slot in slot_to_fact:
                key = slot_to_fact[slot]
                turns.append(dict(id=tid, role="user", text=fact_by_key[key]["text"], fact=key))
                turn_ids[key] = tid
                turns.append(dict(id=f"s{s + 1}_t{len(turns)}", role="assistant", text=rng.choice(ACKS), fact=None))
            else:
                turns.append(dict(id=tid, role="user", text=rng.choice(FILLER_USER), fact=None))
                turns.append(dict(id=f"s{s + 1}_t{len(turns)}", role="assistant", text=rng.choice(FILLER_ASSISTANT), fact=None))
        sessions.append(dict(session=s + 1, date=(start + timedelta(days=7 * s)).isoformat(), turns=turns))

    for i, q in enumerate(questions):
        q["qid"] = f"u{uid}_q{i}"
        q["gold_turns"] = [turn_ids[k] for k in q["gold"]]
    return dict(user_id=f"u{uid}", sessions=sessions, questions=questions, facts=facts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=12, help="user/assistant pairs per session")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/longmem_synth.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    users = [build_user(u, rng, args.sessions, args.pairs) for u in range(args.users)]
    n_turns = sum(len(s["turns"]) for u in users for s in u["sessions"])
    n_q = sum(len(u["questions"]) for u in users)
    meta = dict(users=args.users, sessions=args.sessions, pairs=args.pairs, seed=args.seed,
                turns=n_turns, questions=n_q)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, users=users), f, indent=1, ensure_ascii=False)
    from collections import Counter
    print("wrote", args.out, meta)
    print("question types:", Counter(q["type"] for u in users for q in u["questions"]))


if __name__ == "__main__":
    main()
