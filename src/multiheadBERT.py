import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
import src.config as cfg

class MultiHeadBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        #classifier per head
        self.heads = nn.ModuleDict({
            "occasion": nn.Linear(config.hidden_size, len(cfg.OCCASION_LABELS)),
            "size": nn.Linear(config.hidden_size, len(cfg.SIZE_LABELS)),
            "due_date": nn.Linear(config.hidden_size, len(cfg.DATE_LABELS)),
            "flavor": nn.Linear(config.hidden_size, len(cfg.FLAVOR_LABELS)),
            "filling": nn.Linear(config.hidden_size, len(cfg.FILLING_LABELS)),
            "icing": nn.Linear(config.hidden_size, len(cfg.ICING_LABELS)),
        })

        self.loss_ce = nn.CrossEntropyLoss()
        # #for filling, flavor, icing slots
        # flavor_weights = torch.tensor([7.0] * len(cfg.FLAVOR_LABELS))
        # filling_weights = torch.tensor([15.0] * len(cfg.FILLING_LABELS))
        # icing_weights = torch.tensor([12.0] * len(cfg.ICING_LABELS))

        # self.loss_bce_flavor = nn.BCEWithLogitsLoss(pos_weight=flavor_weights)
        # self.loss_bce_filling = nn.BCEWithLogitsLoss(pos_weight=filling_weights)
        # self.loss_bce_icing = nn.BCEWithLogitsLoss(pos_weight=icing_weights)

        self.loss_bce = nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        #ground truth labels for batch
        labels_occasion=None,
        labels_size=None,
        labels_due_date=None,
        labels_flavor=None,
        labels_filling=None,
        labels_icing=None,
        
    ):
        #Run through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.pooler_output  # [batch, hidden_size], also the CLS vector, one for each sequence
        pooled_output = self.dropout(pooled_output)

        #Each head predicts logits
        #[batch, len_slot]
        logits_occasion   = self.heads["occasion"](pooled_output)
        logits_size       = self.heads["size"](pooled_output)
        logits_due_date = self.heads["due_date"](pooled_output)
        logits_flavor = self.heads["flavor"](pooled_output)
        logits_filling = self.heads["filling"](pooled_output)
        logits_icing = self.heads["icing"](pooled_output)



        loss = None
        #Compute loss if labels are provided
        loss_o = None
        loss_s = None
        loss_d = None
        loss_fl = None
        loss_fi = None
        loss_i = None

        if labels_occasion is not None:
            loss_o = self.loss_ce(logits_occasion, labels_occasion)
        if labels_size is not None:
            loss_s = self.loss_ce(logits_size, labels_size)
        if labels_due_date is not None:
            loss_d = self.loss_ce(logits_due_date, labels_due_date)

        
        # #Multi-Label Loss
        bce_weight = 1.0
        if labels_flavor is not None:
            loss_fl = (bce_weight * self.loss_bce(logits_flavor, labels_flavor.float()))
            # loss_fl = self.loss_bce_flavor(logits_flavor, labels_flavor.float())
        if labels_filling is not None:
            loss_fi = (bce_weight * self.loss_bce(logits_filling, labels_filling.float()))
            # loss_fi = self.loss_bce_filling(logits_filling, labels_filling.float())

        if labels_icing is not None:
            loss_i = (bce_weight * self.loss_bce(logits_icing, labels_icing.float()))
        #   loss_i = self.loss_bce_icing(logits_icing, labels_icing.float())

        losses = []

        # for l in [loss_o, loss_s, loss_d]:
        #     if l is not None:
        #         losses.append(l)
        for l in [loss_o, loss_s, loss_d, loss_fl, loss_fi, loss_i]:
            if l is not None:
                losses.append(l)
        if losses:
            loss = sum(losses)
        else:
            loss = None

        # if loss_o is not None:
        #     loss = loss_o


        return (loss, logits_occasion, logits_size, logits_due_date, logits_flavor, logits_filling, logits_icing)
            # "logits_flavor": logits_flavor,
        #     "logits_filling": logits_filling,
        #     "logits_icing": logits_icing,
        
