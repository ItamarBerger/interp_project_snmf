import logging
import logging.config
import yaml
from constants import LOG_CONFIG_FILE

def setup_logging():
    with open(LOG_CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

