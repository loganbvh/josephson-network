import json
from datetime import datetime

import numpy as np
import pint

DTFORMAT = "%y%m%d_%H%M%S"


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pint.Quantity):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.squeeze().tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)
