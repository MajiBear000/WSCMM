# -*- conding: utf-8 -*-
import os
import logging
from sw.main_config import parse_args
from data_loader import load_data
from sw.utils import (
        get_model,
        get_tokenizer,
        set_seed,
        test_metrix_log,
        )
from prepare_data import get_ids, get_embs
from sw.trainer import Trainer
from models import ClassificationForBasicMean_Linear, ClassificationForBasicMean_RoBERTa

logger = logging.getLogger(__name__)

def get_trainer(args, roberta, tokenizer, raw_data):
    if args.model_name=='linear':
        data = get_embs(args, roberta, tokenizer, raw_data)
        model = ClassificationForBasicMean_Linear(args, roberta)
        trainer = Trainer(args, data['train'], data['val'], model)
    elif args.model_name=='roberta':
        data = get_ids(args, tokenizer, raw_data)
        model = ClassificationForBasicMean_RoBERTa(args, roberta)
        trainer = Trainer(args, data['train'], data['val'], model)
    return data, trainer, model
    
def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args = parse_args().args
    logger.info(vars(args))
    set_seed(args)

    roberta = get_model(args.model_path)
    tokenizer = get_tokenizer(args.model_path)

    raw_data = load_data(args)
    
    data, trainer, _ = get_trainer(args, roberta, tokenizer, raw_data)
    
    trainer.train()

    result = trainer.evaluate(data['test'])
    test_metrix_log(os.path.join('saves', args.stamp+'.txt'), result)
    
if __name__ == '__main__':
    main()
