from .basic_funcs import name2func
import logging

log = logging.getLogger(__name__)


def cfg2loss(cfg):
    def loss(x, y):
        s = 0
        for mini_cfg in cfg:
            if not mini_cfg["name"] in name2func:
                msg = f"Loss {mini_cfg['name']} is no supported"
                log.critical(msg)
                raise Exception(msg)
            s += mini_cfg["weight"] * name2func[mini_cfg["name"]](x, y).mean()
        return s

    return loss
