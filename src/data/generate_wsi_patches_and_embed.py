import yaml
import argparse
from histogpt.helpers.patching import main as patching_main, PatchingConfigs
import os


def generate_wsi_patches_and_embedding(configs: dict):
    try:
        pt_configs = configs.get("patching")
        os.makedirs(pt_configs["save_path"], exist_ok=True)
        if not pt_configs:
            raise KeyError("The configuration file is missing the 'patching' section.")

        patch_configs = PatchingConfigs(**pt_configs)
        patching_main(patch_configs)
        print("WSI patches and embeddings generation completed successfully.")
    except KeyError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An error occurred during patching and embedding generation: {e}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(
        description="Generate WSI patches and embeddings."
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
        generate_wsi_patches_and_embedding(config_dict)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
