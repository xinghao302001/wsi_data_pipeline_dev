import os
from pathlib import Path
import yaml
import argparse
import csv


def aggregate_all_generated_wsi_texts(configs: dict):
    try:
        aggregation_configs = configs.get("aggregation")
        if not aggregation_configs:
            raise KeyError(
                "The configuration file is missing the 'aggregation' section."
            )

        save_path = aggregation_configs.get("save_path", "")
        texts_path = aggregation_configs.get("texts_path", "")
        if not save_path or not texts_path:
            raise ValueError(
                "Both 'save_path' and 'texts_path' must be specified in the configuration."
            )

        csv_file_path = os.path.join(save_path, "result.csv")
        final_res = []

        # # Ensure save directory exists
        # os.makedirs(save_path, exist_ok=True)

        for report_file in Path(texts_path).glob("*.txt"):
            try:
                with open(report_file, "r", encoding="utf-8") as file:
                    final_res.append([report_file.stem, file.read()])
            except Exception as e:
                print(f"Error reading file {report_file}: {e}")

        if not final_res:
            print(
                "No text files were processed. Ensure the directory contains '.txt' files."
            )

        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["WSI Name", "Generated Text"])
            csv_writer.writerows(final_res)

        print(f"Final result.csv stored in: {csv_file_path}")

    except Exception as e:
        print(f"An error occurred during aggregation: {e}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(
        description="Aggregate WSI texts into a CSV file."
    )
    args_parser.add_argument(
        "--config",
        dest="config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = args_parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        aggregate_all_generated_wsi_texts(config_dict)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
