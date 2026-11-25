import os
import sys

# Permite `import src.<modulo>` al correr pytest desde la raíz del repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
