import logging
import psutil
from psutil._common import bytes2human


def log_virtual_memory_stats():
    mem_usage = psutil.virtual_memory()
    total = bytes2human(mem_usage.total)
    available = bytes2human(mem_usage.available)
    percent_used = mem_usage.percent
    used = bytes2human(mem_usage.used)
    logging.info(f'Memory usage: total={total}, available={available}, percent_used={percent_used}%, used={}
