from transformers import BertTokenizerFast
import src.config as cfg

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def multi_label(label_list, label2id, num_labels):
    x = [0] * num_labels
    for lab in label_list:
        i = label2id[lab]
        x[i] = 1
    return x

def tokenize_and_label(example):
    out = tokenizer(
        example["Text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    out["labels_occasion"] = cfg.occasion_label2id[example["Occasion"]]
    out["labels_size"] = cfg.size_label2id[example["Size"]]
    out["labels_due_date"] = cfg.date_label2id[example["Due_Date"]]

    out["labels_flavor"] = multi_label(example["Flavor"], cfg.flavor_label2id, len(cfg.FLAVOR_LABELS),)

    out["labels_filling"] = multi_label(example["Filling"], cfg.filling_label2id, len(cfg.FILLING_LABELS),)
    out["labels_icing"] = multi_label(example["Icing"], cfg.icing_label2id, len(cfg.ICING_LABELS),)
    
    return out
