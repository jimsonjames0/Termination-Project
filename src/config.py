OCCASION_LABELS = [
  "birthday",
  "graduation",
  "anniversary",
  "wedding",
  "baby shower",
  "bridal shower",
  "gender reveal",
  "religious",
  "holiday",
  # …etc
]
# (birthday -> 0)
occasion_label2id = {label: i for i, label in enumerate(OCCASION_LABELS)}

# print(occasion_label2id)
#(0 -> birthday)
occasion_id2label = {i: label for label, i in occasion_label2id.items()}
# print(occasion_id2label)

SIZE_LABELS = [
  "7-inch","10-inch", "14-inch", "16-inch", "half-sheet","full-sheet", "quarter-sheet", "two-tier", "three-tier", "custom", "unspecified"
]
size_label2id = {label: i for i, label in enumerate(SIZE_LABELS)}
size_id2label = {i: label for label, i in size_label2id.items()}

DATE_LABELS   = ["explicit", "too_soon", "unspecified"]
date_label2id = {label: i for i, label in enumerate(DATE_LABELS)}
date_id2label = {i: label for label, i in date_label2id.items()}

# print(date_label2id)
# print(date_id2label)

COMPLEXITY_LABELS = ["simple","custom"]
complexity_label2id = {label: i for i, label in enumerate(COMPLEXITY_LABELS)}
complexity_id2label = {i: label for label, i in complexity_label2id.items()}

FLAVOR_LABELS = [
  "vanilla", "chocolate", "marble", "confetti", "red velvet", 
  "strawberry", "lemon", "almond", "carrot", "pumpkin", "unspecified"
]
flavor_label2id = {label: i for i, label in enumerate(FLAVOR_LABELS)}
flavor_id2label = {i: label for label, i in flavor_label2id.items()}


FILLING_LABELS = ["vanilla custard", "chocolate custard", "raspberry", "vanilla buttercream", "chocolate buttercream", "cookies & cream buttercream", "raspberry buttercream", "strawberry buttercream", 
                  "caramel buttercream", "almond buttercream", "chocolate mousse", "white chocolate mousse", "peanut butter mousse", "coconut mousse", "raspberry mousse", 
                   "cannoli", "cream cheese icing", "chocolate ganache", "white chocolate ganache", "white chocolate icing", "oreo filling", "unspecified", "no filling" ]
filling_label2id = {label: i for i, label in enumerate(FILLING_LABELS)}
filling_id2label = {i: label for label, i in filling_label2id.items()}

ICING_LABELS = ["naked", "buttercream", "vanilla buttercream", "chocolate buttercream", "cookies & cream buttercream", "raspberry buttercream", "strawberry buttercream", "caramel buttercream", "almond buttercream", 
                "cream cheese", "white chocolate ganache", "chocolate ganache", "white chocolate icing", "fondant", "unspecified"]

icing_label2id = {label: i for i, label in enumerate(ICING_LABELS)}
icing_id2label = {i : label for label, i in icing_label2id.items()}

# print(filling_label2id["vanilla custard"])
# print(filling_id2label[0])

