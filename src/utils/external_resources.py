import json
import logging

import gridfs
from pymongo import MongoClient
from sacred.observers import MongoObserver
from visdom import Visdom

logger = logging.getLogger(__name__)

VISDOM_CONF_PATH = 'resources/visdom.json'


def get_visdom(env, server=None, port=None, visdom_path=VISDOM_CONF_PATH):
    with open(visdom_path) as file:
        visdom_conf = json.load(file)

    server = visdom_conf['url'] if server is None else server
    port = visdom_conf['port'] if port is None else port
    logger.info('Init Visdom client to {}:{}'.format(server, port))
    return Visdom(server=server, port=port, env=env)


def get_visdom_conf(visdom_path=VISDOM_CONF_PATH):
    with open(visdom_path) as file:
        visdom_conf = json.load(file)
    return visdom_conf
