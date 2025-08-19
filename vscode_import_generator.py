import json
import os
import sys

PYRIGHT_CONFIG_FILE = "pyrightconfig.json"

def main():
    python_path = os.pathsep.join(sys.path).split(":")
    python_path.sort()

    with open(PYRIGHT_CONFIG_FILE, "r") as f:
        pyright_settings = json.loads(f.read())

    pyright_settings['extraPaths'] = python_path

    with open(PYRIGHT_CONFIG_FILE, "w") as f:
        f.write(json.dumps(pyright_settings, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()