# from transformers import BertTokenizerFast
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# text = "I really need a birthday cake. I want the birthday cake for tomorrow."
# out = tokenizer(text, truncation=True, padding="max_length", max_length=16)

# print(out)
# print("Tokens:", tokenizer.tokenize(text))
# print("IDs:   ", out["input_ids"])

# from src.multiheadBERT import MultiHeadBertForSequenceClassification
# from transformers import BertConfig

# config = BertConfig.from_pretrained("bert-base-uncased")
# model = MultiHeadBertForSequenceClassification(config)

# print(model)
# import json

# # === 1) Paths – CHANGE THESE IF NEEDED ===
# BASE_FILE = "data/seed_slots.jsonl"          # your current dataset
# OUT_FILE  = "data/seed_slots_with_chatbot.jsonl"  # new combined file

# # === 2) Load existing data ===
# base_examples = []
# with open(BASE_FILE, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#         base_examples.append(json.loads(line))

# print(f"Loaded {len(base_examples)} base examples from {BASE_FILE}")

# # === 3) Chatbot-style extra examples (Text + Occasion only) ===

# extra_raw = [
#     # gender reveal (16)
#     {"Text": "Hey, can you make a small cake for our baby’s gender reveal this Sunday?", "Occasion": "gender reveal"},
#     {"Text": "We’re doing a gender reveal with close friends, can I get a simple vanilla cake that’s blue inside if it’s a boy?", "Occasion": "gender reveal"},
#     {"Text": "I need a cake that says ‘He or She?’ for our gender reveal party next weekend.", "Occasion": "gender reveal"},
#     {"Text": "Can you do a gender reveal cake where the inside color shows if it’s a boy or girl?", "Occasion": "gender reveal"},
#     {"Text": "Looking for a small gender reveal cake we can cut open on a video call with family.", "Occasion": "gender reveal"},
#     {"Text": "We want a cake that says ‘Twinkle Twinkle Little Star, How We Wonder What You Are’ for our gender reveal.", "Occasion": "gender reveal"},
#     {"Text": "Do you make half-pink, half-blue cakes for gender reveal parties?", "Occasion": "gender reveal"},
#     {"Text": "I need a cake that hides pink or blue filling for a surprise gender reveal.", "Occasion": "gender reveal"},
#     {"Text": "Please make a neutral-looking gender reveal cake with the surprise color only inside.", "Occasion": "gender reveal"},
#     {"Text": "Can I order a donut-style cake with colored cream inside for a gender reveal?", "Occasion": "gender reveal"},
#     {"Text": "We’re doing a sports-themed gender reveal, can the cake say ‘Rookie Coming Soon’ and show the color inside?", "Occasion": "gender reveal"},
#     {"Text": "I’d love a small elegant cake that reveals pink inside when we cut it, for our gender reveal dinner.", "Occasion": "gender reveal"},
#     {"Text": "Can you do a gender reveal cake that looks white outside but explodes pink or blue candies when cut?", "Occasion": "gender reveal"},
#     {"Text": "Need a last-minute gender reveal cake that says ‘Boy or Girl?’ with blue or pink inside.", "Occasion": "gender reveal"},
#     {"Text": "We’re hosting a backyard gender reveal, can you make a rustic-style cake with surprise filling?", "Occasion": "gender reveal"},
#     {"Text": "I want a gender reveal cake that looks like a gift box and reveals the baby’s gender when sliced.", "Occasion": "gender reveal"},

#     # bridal shower (16)
#     {"Text": "I’m throwing a bridal shower for my best friend, can you make a classy floral cake?", "Occasion": "bridal shower"},
#     {"Text": "Need a simple white cake with gold accents for a small bridal shower brunch.", "Occasion": "bridal shower"},
#     {"Text": "Can you make a cake that says ‘From Miss to Mrs.’ for a bridal shower?", "Occasion": "bridal shower"},
#     {"Text": "Looking for a delicate pink and white cake with roses for my sister’s bridal shower.", "Occasion": "bridal shower"},
#     {"Text": "We’re hosting a tea-party themed bridal shower, can the cake be soft pastel colors with her name?", "Occasion": "bridal shower"},
#     {"Text": "Do you make heart-shaped cakes for bridal showers with custom messages?", "Occasion": "bridal shower"},
#     {"Text": "I need a small elegant cake with lace piping details for a bridal shower of 15 people.", "Occasion": "bridal shower"},
#     {"Text": "Can we get a bridal shower cake that matches a greenery and eucalyptus theme?", "Occasion": "bridal shower"},
#     {"Text": "Please make a cake that says ‘Bride to Be’ with champagne glasses for a bridal shower.", "Occasion": "bridal shower"},
#     {"Text": "I’d like a rustic naked cake with fresh flowers for an outdoor bridal shower.", "Occasion": "bridal shower"},
#     {"Text": "We’re celebrating a bridal shower at a restaurant, can you make a small but fancy cake we can bring?", "Occasion": "bridal shower"},
#     {"Text": "Do you do engagement ring–themed cakes for bridal showers?", "Occasion": "bridal shower"},
#     {"Text": "Can you make a bridal shower cake inspired by the bride’s wedding colors: navy and blush?", "Occasion": "bridal shower"},
#     {"Text": "Need a bridal shower cake that says ‘She said yes!’ in cursive on top.", "Occasion": "bridal shower"},
#     {"Text": "I want a simple two-layer cake with pearls and flowers for a classy bridal shower.", "Occasion": "bridal shower"},
#     {"Text": "Can you design a lingerie-themed cake for a fun bridal shower (still tasteful)?", "Occasion": "bridal shower"},

#     # baby shower (6)
#     {"Text": "We’re having a baby shower next month, can you make a cute teddy bear cake?", "Occasion": "baby shower"},
#     {"Text": "I need a pastel-colored baby shower cake that says ‘Welcome Little One’.", "Occasion": "baby shower"},
#     {"Text": "Can you make a baby shower cake with small booties and blocks on top?", "Occasion": "baby shower"},
#     {"Text": "We want a neutral baby shower cake since we don’t know the gender yet.", "Occasion": "baby shower"},
#     {"Text": "I’d like a baby shower cake with clouds and stars for our ‘twinkle twinkle’ theme.", "Occasion": "baby shower"},
#     {"Text": "Do you make diaper cake–style designs in actual cake form for baby showers?", "Occasion": "baby shower"},

#     # birthday (6)
#     {"Text": "Can I order a small chocolate birthday cake that says ‘Happy Birthday Mom’?", "Occasion": "birthday"},
#     {"Text": "I need a funfetti birthday cake for my friend’s 21st with simple writing.", "Occasion": "birthday"},
#     {"Text": "Do you make number-shaped cakes for kids’ birthday parties?", "Occasion": "birthday"},
#     {"Text": "I’m looking for a minimalist vanilla birthday cake with just ‘Happy Birthday Alex’.", "Occasion": "birthday"},
#     {"Text": "Can you make a soccer-themed birthday cake for a 10-year-old boy?", "Occasion": "birthday"},
#     {"Text": "We want a small elegant birthday cake for a coworker, nothing too decorated.", "Occasion": "birthday"},

#     # wedding (6)
#     {"Text": "Can you make a romantic wedding cake with white buttercream and fresh flowers?", "Occasion": "wedding"},
#     {"Text": "We need a simple two-tier wedding cake for a small ceremony next month.", "Occasion": "wedding"},
#     {"Text": "Do you do semi-naked wedding cakes with berries on top?", "Occasion": "wedding"},
#     {"Text": "Looking for a classic white wedding cake with our initials on the side.", "Occasion": "wedding"},
#     {"Text": "Can you make a modern geometric wedding cake with gold accents?", "Occasion": "wedding"},
#     {"Text": "We’re having a tiny courthouse wedding, can we get a single-tier wedding cake?", "Occasion": "wedding"},
# ]

# print(f"Prepared {len(extra_raw)} extra chatbot-style examples")

# # === 4) Fill in default values for other slots ===

# DEFAULTS = {
#     "Size": "unspecified",
#     "Due_Date": "unspecified",
#     "Flavor": ["unspecified"],
#     "Filling": ["unspecified"],
#     "Icing": ["unspecified"],
# }

# def add_defaults(ex):
#     ex = ex.copy()
#     for k, v in DEFAULTS.items():
#         ex.setdefault(k, v)
#     return ex

# extra_examples = [add_defaults(ex) for ex in extra_raw]

# # === 5) Combine and write out new JSONL ===

# all_examples = base_examples + extra_examples
# print(f"Total examples after augmentation: {len(all_examples)}")

# with open(OUT_FILE, "w", encoding="utf-8") as f:
#     for ex in all_examples:
#         f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# print(f"Wrote combined dataset to {OUT_FILE}")

from collections import Counter
from datasets import load_dataset

ds = load_dataset("json", data_files="data/seed_slots.jsonl", split="train")
counts = Counter(ex["Occasion"] for ex in ds)
print(counts)
