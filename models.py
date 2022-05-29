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
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self):
        return 0

class ClassificationForBasicMean_Linear(DNN):
    """
        base model for Linear Classification
            args.dropout: ratio for dropout layer
    """

    def __init__(self, roberta, num_labels=2, drop_ratio=0.2):
        """Initialize the model"""
        super(ClassificationForBasicMean_Linear, self).__init__()
        self.num_labels = num_labels
        self.config = roberta.config
        self.dropout = nn.Dropout(args.drop_ratio)

        self.hidden_1 = nn.Linear(self.config.hidden_size*2, self.config.hidden_size) if args.con_emb else nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        self._init_weights(self.hidden_1)
        self._init_weights(self.classifier)

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

class ClassificationForBasicMean_RoBERTa(DNN):
    """
        base model for RoBERTa Classification
            args.drop_ratio: ratio for dropout layer
    """

    def __init__(self, roberta, num_labels=2, drop_ratio=0.2):
        super(ClassificationForBasicMean_RoBERTa, self).__init__()
        self.num_labels = num_labels
        self.roberta = roberta
        self.config = roberta.config
        self.dropout = nn.Dropout(drop_ratio)
        self.classifier = nn.Linear(self.config.hidden_size*2, num_labels)

        self._init_weights(self.classifier)

    def forward(
        self,
        basic_ids,
        basic_attention,
        basic_mask,
        con_ids,
        con_attention,
        con_mask,
        labels=None,
        ):
        basic_out = self.roberta(basic_ids, attention_mask=basic_attention)
        con_out = self.roberta(con_ids, attention_mask=con_attention)
        
        basic_out = self.dropout(basic_out[0])
        con_out = self.dropout(con_out[0])
        
        basic_target = basic_out * basic_mask.unsqueeze(2)
        con_target = con_out * con_mask.unsqueeze(2)

        basic_target = basic_target.mean(1)
        con_target = con_target.mean(1)

        contrast_input = torch.cat([basic_target, con_target],dim=1)
        logits = self.classifier(contrast_input)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits
        
class Classification_MelBERT(DNN):
    """
        base model for RoBERTa MelBERT
            args.drop_ratio: ratio for dropout layer
    """

    def __init__(self, args, Model, num_labels=2):
        """Initialize the model"""
        super(Classification_MelBERT, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = self.encoder.config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args

        self.SPV_linear = nn.Linear(self.config.hidden_size * 2, args.classifier_hidden)
        self.MIP_linear = nn.Linear(self.config.hidden_size * 2, args.classifier_hidden)
        self.classifier = nn.Linear(args.classifier_hidden * 2, num_labels)
        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._init_weights(self.classifier)

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask,
        attention_mask_2,
        token_type_ids,
        
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the first input token indices in the vocabulary
            `input_ids_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the second input token indicies
            `target_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the first input. 1 for target word and 0 otherwise.
            `target_mask_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the second input. 1 for target word and 0 otherwise.
            `attention_mask_2`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the second input.
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the first input.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """

        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        pooled_output = outputs[1]  # [batch, hidden]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)

        target_output = target_output.mean(1)  # [batch, hidden]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        target_output_2 = target_output_2.mean(1)

        # Get hidden vectors each from SPV and MIP linear layers
        SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([target_output_2, target_output], dim=1))

        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits
    

class ClassificationForBasicMean_MelBERT(DNN):
    """
        base model for RoBERTa MelBERT
            args.drop_ratio: ratio for dropout layer
    """

    def __init__(self, args, Model, basic_encoder, num_labels=2):
        """Initialize the model"""
        super(ClassificationForBasicMean_MelBERT, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = self.encoder.config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args

        self.SPV_linear = nn.Linear(self.config.hidden_size * 2, args.classifier_hidden)
        self.MIP_linear = nn.Linear(self.config.hidden_size * 2, args.classifier_hidden)
        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)

        self.basic_encoder = basic_encoder
        self.classifier = nn.Linear(args.classifier_hidden * 2 + self.config.hidden_size, num_labels)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._init_weights(self.classifier)

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask,
        attention_mask_2,
        token_type_ids,

        basic_ids,
        basic_mask,
        basic_attention,
        basic_token_type_ids,
        
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the first input token indices in the vocabulary
            `input_ids_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the second input token indicies
            `target_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the first input. 1 for target word and 0 otherwise.
            `target_mask_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the second input. 1 for target word and 0 otherwise.
            `attention_mask_2`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the second input.
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the first input.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """

        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        pooled_output = outputs[1]  # [batch, hidden]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)

        target_output = target_output.mean(1)  # [batch, hidden]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        target_output_2 = target_output_2.mean(1)

        # Get hidden vectors each from SPV and MIP linear layers
        SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([target_output_2, target_output], dim=1))

        # Get basic mean #
        basic_out = self.basic_encoder(basic_ids, attention_mask=basic_attention, token_type_ids=basic_token_type_ids)
        con_out = self.basic_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        basic_out = self.dropout(basic_out[0])
        con_out = self.dropout(con_out[0])
        
        basic_target = basic_out * basic_mask.unsqueeze(2)
        con_target = con_out * target_mask.unsqueeze(2)

        basic_target = basic_target.mean(1)
        con_target = con_target.mean(1)
        basic_MIP_hidden = basic_target - con_target

        # Classificating #
        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden, basic_MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits































        



        
