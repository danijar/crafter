import pathlib

import ruamel.yaml

root = pathlib.Path(__file__).parent
yaml = ruamel.yaml.YAML(typ='safe', pure=True)
for key, value in yaml.load((root / 'data.yaml').read_text()).items():
  globals()[key] = value
