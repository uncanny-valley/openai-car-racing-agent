import os
import psutil


def log_ram_usage(proc: psutil.Process):
    logging.info(f'RAM usage: {proc.memory_info()[0]/2. ** 30} GB')

def log_memory_percentage(proc: psutil.Process):
    logging.info(f'System-wide memory usage percentage: {proc.memory_percent()}')

def log_cpu_percentage(proc: psutil.Process):
    logging.info(f'System-wide CPU usage percentage: {proc.cpu_percent()}')

def log_memory_usage_summary():
    p = psutil.Process(os.getpid())
    log_ram_usage(p)
    log_memory_percentage(p)
    log_cpu_percentage(p)