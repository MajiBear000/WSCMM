# -*- conding: utf-8 -*-
import logging
from main_config import parse_args
from data_loader import load_data
from utils import (
        get_model,
        get_tokenizer,
        )

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args = parse_args().args
    logger.info(vars(args))

    data = load_data(args)

    model = get_model(args.model_path)
    tokenizer = get_tokenizer(args.model_path)

if __name__ == '__main__':
    main()
