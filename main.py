# -*- conding: utf-8 -*-
import logging
from main_config import parse_args

def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args = parse_args.args
    logger.info(vars(args))

if __init__ == '__main__':
    main()
