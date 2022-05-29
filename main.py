# -*- conding: utf-8 -*-
import os
import logging

import sys
sys.path.append("..")
import test
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
    model.to(args.device)
    trainer = Trainer(args, data, model)
    return data, trainer, model
    
def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args = parse_args().args
    logger.info(vars(args))
    set_seed(args)

    encoder = get_model(args.model_path)
    tokenizer = get_tokenizer(args.model_path)
    logger.info('***** Model Load Success *****')
    
    raw_data = load_data(args)
    logger.info('***** Data Load Success *****')
    
    data, trainer, _ = get_trainer(args, encoder, tokenizer, raw_data)
    
    trainer.train()

    result = trainer.evaluate(data['test'])
    test_metrix_log(os.path.join('saves', args.stamp+'.txt'), result)
    
if __name__ == '__main__':
    main()
