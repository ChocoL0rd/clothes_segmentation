from .basic_funcs import name2func
import logging

log = logging.getLogger(__name__)


def cfg2metric_dict(cfg):
    """Return dict {metric_name: metric, ...}"""
    metric_dict = {}
    for metric_name in cfg:
        if metric_name not in name2func.keys():
            msg = f"Metric {metric_name} is not supported"
            log.critical(msg)
            raise Exception(msg)

        metric_dict[metric_name] = name2func[metric_name]
    return metric_dict

