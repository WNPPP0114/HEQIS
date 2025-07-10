# logger_utils.py
import logging, json, os, datetime

def setup_experiment_logging(output_dir: str, args: dict,
                             log_name_prefix: str = "train") -> logging.Logger:
    """
    Initialize experiment logging.

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
        The configured logger, use directly with logger.info(...).
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name_prefix}_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()          # Keep console output
        ]
    )

    logger = logging.getLogger()          # root logger
    logger.info("ARGS = %s",
                json.dumps(args, ensure_ascii=False, indent=2))
    return logger