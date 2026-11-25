"""Skip the monitoring tests when the monitoring dependencies aren't installed.

They live in monitoring/requirements.txt, so the team CI can run the rest of the
suite before it installs them.
"""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("evidently") else ["test_*.py"]
