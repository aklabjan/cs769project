import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
import torch

class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        #self.lstm = nn.LSTM(768, config.hidden_size, batch_first=True)
        self.dense = nn.Linear(768, config.hidden_size)
        
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.MultiLabelSoftMarginLoss(weight = torch.FloatTensor([1.2126, 8.5310, 9.318, 0.3885, 0.475, 2.073, 1.260]))
        #self.loss_fct = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )     
        pooled_output = outputs[1]
   
        #dense_out = self.dense(outputs[1])
      
        #dense_output = self.dropout(dense_out)
        #logits = self.classifier(dense_output)
    
        

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
