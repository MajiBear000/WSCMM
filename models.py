# -*- conding: utf-8 -*-
import torch
import torch.nn as nn

class ClassificationForBasicMean_Linear(nn.Module):
    """base model for Linear Classification"""

    def __init__(self, args, config, num_labels=2):
        """Initialize the model"""
        super(ClassificationForBasicMean_Linear, self).__init__()
        self.num_labels = num_labels
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.classifier = nn.Linear(768*2, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, basic_emb, test_emb, labels=None):
        """
        Inputs:
            'basic_emb': A torch.LongTensor of shape [batch, hidden_size(768)],
                embedding of basic means from UVA training set, generate by pre-trained models.
            'test_emb': A torch.LongTensor of shape [batch, hidden_size(768)],
                embedding of contextual target from UVA testing set, generate by pre-trained models.
        """
        contrast_input = torch.cat([basic_emb, test_emb],dim=1) #(N,D)
        contrast_input = self.dropout(contrast_input)
        logits = self.classifier(contrast_input)
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits






        
