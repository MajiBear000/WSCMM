# -*- conding: utf-8 -*-
import os
import logging

import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import sys
sys.path.append("..")

from sw.main_config import parse_args
from sw.utils import get_model, get_tokenizer, set_seed, test_metrix_log
from sw.trainer import Trainer
from processor import prepare_data
from data_loader import load_data
from models import ClassificationForBasicMean_Linear, ClassificationForBasicMean_RoBERTa, ClassificationForBasicMean_MelBERT

logger = logging.getLogger(__name__)

def get_trainer(args, encoder, tokenizer, raw_data):
    processor = prepare_data(args, raw_data)
    if args.model_name=='linear':
        data = processor.get_embs(encoder, tokenizer)
        model = ClassificationForBasicMean_Linear(encoder, drop_ratio=args.drop_ratio)
    elif args.model_name=='roberta':
        data = processor.get_ids(tokenizer)
        model = ClassificationForBasicMean_RoBERTa(encoder, drop_ratio=args.drop_ratio)
    elif args.model_name=='melbert':
        data = processor.melbert_ids(tokenizer)
        model = ClassificationForBasicMean_MelBERT(args, encoder)
    if args.n_gpu > 1 and not args.no_cuda:
        model = nn.DataParallel(model)
    trainer = Trainer(args, data, model)
    return data, trainer, model

def get_encoder(args):
    if args.model_name in ['melbert']:
        bert = AutoModel.from_pretrained(args.bert_model)
        config = bert.config
        config.type_vocab_size = 4
        if "albert" in args.bert_model:
            bert.embeddings.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.embedding_size
            )
        else:
            bert.embeddings.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )
        bert._init_weights(bert.embeddings.token_type_embeddings)

        encoder = bert
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    else:
        encoder = get_model(args.model_path)
        tokenizer = get_tokenizer(args.model_path)
    return encoder, tokenizer
    
def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args = parse_args().args
    logger.info(vars(args))
    set_seed(args)

    encoder, tokenizer = get_encoder(args)
    logger.info('***** Encoder Load Success *****')
    
    raw_data = load_data(args)
    logger.info('***** Data Load Success *****')
    
    data, trainer, _ = get_trainer(args, encoder, tokenizer, raw_data)
    
    trainer.train()

    result = trainer.evaluate(data['test'])
    test_metrix_log(os.path.join('saves', args.stamp+'.txt'), result)
    
if __name__ == '__main__':
    main()
