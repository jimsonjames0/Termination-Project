from collections import Counter
from datasets import load_dataset

ds = load_dataset("json", data_files="data/seed_slots.jsonl", split="train")
counts = Counter(ex["Occasion"] for ex in ds)
print(counts)


import itertools
def count_multi_labels(dataset_path: str):
    # Load the entire dataset
    ds = load_dataset("json", data_files=dataset_path, split="train")

    # flatten the list of lists into a single list of all labels
    all_flavors = list(itertools.chain.from_iterable(ex["Flavor"] for ex in ds))
    all_fillings = list(itertools.chain.from_iterable(ex["Filling"] for ex in ds))
    all_icings = list(itertools.chain.from_iterable(ex["Icing"] for ex in ds))

    # frequency of each unique label
    flavor_counts = Counter(all_flavors)
    filling_counts = Counter(all_fillings)
    icing_counts = Counter(all_icings)

    total_examples = len(ds)
    print(f"Total examples in dataset: {total_examples}\n")

    print("--- Flavor Counts ---")
    for label, count in flavor_counts.most_common():
        prevalence = (count / total_examples) * 100
        print(f"  {label}: {count} occurrences ({prevalence:.2f}%)")

    print("\n--- Filling Counts ---")
    for label, count in filling_counts.most_common():
        prevalence = (count / total_examples) * 100
        print(f"  {label}: {count} occurrences ({prevalence:.2f}%)")

    print("\n--- Icing Counts ---")
    for label, count in icing_counts.most_common():
        prevalence = (count / total_examples) * 100
        print(f"  {label}: {count} occurrences ({prevalence:.2f}%)")

if __name__ == "__main__":
    count_multi_labels("data/seed_slots.jsonl")