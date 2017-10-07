# console utility for pydocsim

import re

from pydocsim import *

# DEBUG AUTO LOAD
reader = Reader('../data/michigan.json')
r1 = reader.get_next_metadata_record()
r2 = reader.get_next_metadata_record()
r3 = reader.get_next_metadata_record()
r4 = reader.get_next_metadata_record()