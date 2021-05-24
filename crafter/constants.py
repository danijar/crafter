import pathlib

import ruamel.yaml as yaml

from . import engine


root = pathlib.Path(__file__).parent
data = engine.AttrDict(yaml.safe_load((root / 'data.yaml').read_text()))
