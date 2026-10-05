"""Generate PolicyGuard checklists from a tau2-bench domain policy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from policyguard.generation import main


if __name__ == "__main__":
    main()
