import pathlib

import ruamel.yaml as yaml

root = pathlib.Path(__file__).parent
yaml = yaml.YAML(typ='safe', pure=True)
for key, value in yaml.load((root / 'data.yaml').read_text()).items():
  globals()[key] = value
