from typing import Dict, Union

from attrdict import AttrDict

config: Dict[str, Union[int, str]] = {"framework": "pt", "NUM_SENT": 15}
config = AttrDict(config)