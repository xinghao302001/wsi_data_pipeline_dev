import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
import json


def setup_project_path():
    """
    Ensure the project root is in the Python path.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def log_configurations(configs: dict):
    """
    Log and print all configurations to the console in a structured format.
    """
    try:
        logging.info("Loaded configurations:")
        formatted_config = json.dumps(configs, indent=4)
        for line in formatted_config.splitlines():
            logging.info(line)
    except Exception as e:
        logging.error(f"Failed to log configurations: {e}")


def wsi_data_pipe(configs: dict):
    """
    Execute the WSI data pipeline tasks.
    """
    try:
        logging.info("Starting: WSI patching and embedding generation...")
        generate_wsi_patches_and_embedding(configs)
        logging.info("Completed: Patching and embedding generation")

        logging.info("Starting: Clinical report text generation...")
        generate_wsi_text_w_patch_and_prompt_embed(configs)
        logging.info("Completed: Clinical report text generation.")

        logging.info("Starting: Text aggregation...")
        aggregate_all_generated_wsi_texts(configs)
        logging.info("Completed: Text aggregation.")
    except Exception as e:
        logging.error(f"Error occurred during WSI data pipeline: {e}")


if __name__ == "__main__":
    setup_project_path()

    try:
        from src.logger.logger import LoggingSetup
        from src.data.generate_wsi_patches_and_embed import (
            generate_wsi_patches_and_embedding,
        )
        from src.data.generate_wsi_texts import (
            generate_wsi_text_w_patch_and_prompt_embed,
        )
        from src.data.aggregate_all_wsi_texts import aggregate_all_generated_wsi_texts

        # Setup logging
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger_config = LoggingSetup(
            log_file=f"logs/wsi_data_pipe_{current_time}.log",
            log_level="INFO",
            max_file_size=2 * 1024 * 1024,  # 2 MB
            backup_count=3,
        )
        logger_config.setup()

        # Parse arguments
        args_parser = argparse.ArgumentParser()
        args_parser.add_argument("--config", dest="config", required=True)
        args = args_parser.parse_args()

        # Load configuration
        try:
            with open(args.config, "r") as file:
                config_dict = yaml.safe_load(file)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logging.error(f"Failed to parse YAML configuration file: {e}")
            sys.exit(1)

        # Log configurations
        log_configurations(config_dict)

        # Run the data pipeline
        wsi_data_pipe(config_dict)
    except Exception as main_exception:
        logging.critical(f"Critical error in the main block: {main_exception}")
        sys.exit(1)
