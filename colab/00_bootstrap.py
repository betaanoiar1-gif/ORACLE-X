# ORACLE-X Google Colab bootstrap
# Run this cell first in a fresh Colab runtime.

!git clone https://github.com/betaanoiar1-gif/ORACLE-X.git /content/ORACLE-X
%cd /content/ORACLE-X
!pip install -q -e .

from pathlib import Path
import sys
sys.path.insert(0, str(Path('/content/ORACLE-X/src')))

import oracle_x
print('ORACLE-X', oracle_x.__version__, 'ready')
