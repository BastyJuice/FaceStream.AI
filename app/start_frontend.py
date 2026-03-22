import logging
import os
import json

from server.config.frontend import ConfigFrontend
from config.manager import (
    ConfigManager,
    initialize_app_structure,
    data_folder,
    known_faces_folder,
    config_file,
)

# logging.basicConfig(level=logging.DEBUG)


def ensure_app_setup():
    """Stellt sicher, dass Ordner und Config existieren."""
    try:
        default_config = initialize_app_structure()

        # data Ordner
        os.makedirs(data_folder, exist_ok=True)

        # knownfaces Ordner
        os.makedirs(known_faces_folder, exist_ok=True)

        # config.json
        if not os.path.isfile(config_file):
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            logging.info("Created default config.json")

    except Exception as e:
        logging.exception(f"Failed to initialize app structure: {e}")
        raise


def main():
    try:
        ensure_app_setup()

        config_manager = ConfigManager(config_file)
        config_manager.load_config()

        config_server = ConfigFrontend(config_manager)
        config_server.run()

    except KeyboardInterrupt:
        logging.info("Config server stopped by user.")
    except Exception as e:
        logging.exception(f"Fatal error in config frontend: {e}")
        raise


if __name__ == '__main__':
    main()