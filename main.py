# -*- conding: utf-8 -*-
import os
import logging
from main_config import parse_args
from data_loader import load_data
from utils import (
        get_model,
        get_tokenizer,
        set_seed,
        test_metrix_log,
        )
from basic_extract import prepare_embedding
from trainer import Trainer
from models import ClassificationForBasicMean_Linear

logger = logging.getLogger(__name__)

def get_embs(args, roberta, tokenizer, data):
    train_emb = prepare_embedding(args, roberta, tokenizer, data, 'train')
    if args.unk_emb:
        test_emb = prepare_embedding(args, roberta, tokenizer, data, 'test')
        val_emb = prepare_embedding(args, roberta, tokenizer, data, 'val')
    else:
        test_emb = prepare_embedding(args, roberta, tokenizer, data, 'test_kn')
        val_emb = prepare_embedding(args, roberta, tokenizer, data, 'val_kn')
    return {'train':train_emb, 'test':test_emb, 'val':val_emb}

def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args = parse_args().args
    logger.info(vars(args))
    set_seed(args)

    data = load_data(args)

    roberta = get_model(args.model_path)
    tokenizer = get_tokenizer(args.model_path)
    model = ClassificationForBasicMean_Linear(args, roberta.config)

    embs = get_embs(args, roberta, tokenizer, data)
    
    trainer = Trainer(args, embs['train'], embs['val'], model)
    trainer.train()

    result = trainer.evaluate(embs['test'])
    test_metrix_log(os.path.join('saves', args.stamp+'.txt'), result)
    
if __name__ == '__main__':
    main()
