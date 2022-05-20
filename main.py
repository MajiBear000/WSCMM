# -*- conding: utf-8 -*-
import logging
from main_config import parse_args

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    parser = parse_args()
    args = parser.args
    logger.info(vars(args))

if __name__ == '__main__':
    main()
