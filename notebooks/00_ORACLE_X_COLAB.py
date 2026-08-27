# ORACLE-X Colab bootstrap
# Run this notebook/script cell-by-cell only after cloning the repository.

!git clone -q https://github.com/betaanoiar1-gif/ORACLE-X.git /content/ORACLE-X
%cd /content/ORACLE-X
!pip -q install -r requirements.txt

import sys
sys.path.insert(0, "/content/ORACLE-X/src")
print("ORACLE-X environment ready")
