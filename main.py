# -*- conding: utf-8 -*-
import logging
from main_config import parse_args
from data_loader import load_data
from utils import (
        get_model,
        get_tokenizer,
        set_seed,
        )
from basic_extract import prepare_embedding
from trainer import train
from models import ClassificationForBasicMean_Linear

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args = parse_args().args
    logger.info(vars(args))
    set_seed(args)

    data = load_data(args)

    roberta = get_model(args.model_path)
    tokenizer = get_tokenizer(args.model_path)
    model = ClassificationForBasicMean_Linear(args, roberta.config)

    test_emb = prepare_embedding(args, roberta, tokenizer, data)

    train(args, test_emb, model)
    
if __name__ == '__main__':
    main()
