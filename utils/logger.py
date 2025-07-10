# logger_utils.py

import logging, json, os, datetime

def setup_experiment_logging(output_dir: str, args: dict,
                             log_name_prefix: str = "train") -> logging.Logger:
    """
    Initialize experiment logging. Sets up logging to a file for a specific experiment.
    Console logging should be configured once at the application entry point (e.g., in the main script).

    Parameters
    ----------
    output_dir : str
        The output_dir passed in the training script, consistent with the trainer's visualization folder.
    args : dict
        All hyperparameters / CLI parsing results that need to be logged, e.g., vars(args).
    log_name_prefix : str, optional
        Prefix for the generated filename, default is "train".

    Returns
    -------
    logging.Logger
        The configured logger (root logger), with a file handler added.
        Note: This function now *only* adds a file handler. Console handling should be set up elsewhere.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    # 为确保即使去掉时间戳目录，日志文件名也能区分每次运行，这里保留日志文件的时间戳
    log_path = os.path.join(log_dir, f"{log_name_prefix}_{ts}.log")

    # 获取根记录器
    logger = logging.getLogger()
    # 记录器的级别应该在主脚本中设置一次，这里不再设置

    # 定义格式器
    formatter = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")

    # 添加文件处理器，用于保存当前实验的日志到文件
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 避免重复添加相同路径的文件处理器 (尽管时间戳通常保证了唯一性)
    # 如果同一记录器已经有这个文件处理器，就跳过
    if not any(isinstance(h, logging.FileHandler) and os.path.abspath(h.baseFilename) == os.path.abspath(log_path) for h in logger.handlers):
         logger.addHandler(file_handler)

    # Log arguments once
    logger.info("ARGS = %s", json.dumps(args, ensure_ascii=False, indent=2))

    return logger