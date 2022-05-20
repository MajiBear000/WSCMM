# -*- conding: utf-8 -*-
import logging
from main_config import parse_args
from data_loader import load_data

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args = parse_args().args
    logger.info(vars(args))

    data = load_data(args)

if __name__ == '__main__':
    main()
