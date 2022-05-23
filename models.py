# -*- conding: utf-8 -*-
import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self):
        """Initialize the model"""
        super(DNN, self).__init__()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self):
        return 0

class ClassificationForBasicMean_Linear(DNN):
    """base model for Linear Classification"""

    def __init__(self, args, config, num_labels=2):
        """Initialize the model"""
        super(ClassificationForBasicMean_Linear, self).__init__()
        self.num_labels = num_labels
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)

        self.hidden_1 = nn.Linear(768*2, 768)
        self.classifier = nn.Linear(768, num_labels)

        self._init_weights(self.hidden_1)
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
        #contrast_input = basic_emb - test_emb
        contrast_input = self.dropout(contrast_input)
        h_emb = self.hidden_1(contrast_input)
        h_emb = self.dropout(h_emb)
        h_emb = self.relu(h_emb)
        logits = self.classifier(h_emb)
        logits = self.relu(logits)
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits






        
