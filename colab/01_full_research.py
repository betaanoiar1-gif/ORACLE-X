# ORACLE-X: single Colab execution path
# 1) Clone + install
!rm -rf /content/ORACLE-X
!git clone -q https://github.com/betaanoiar1-gif/ORACLE-X.git /content/ORACLE-X
%cd /content/ORACLE-X
!pip -q install -r requirements.txt

# 2) Set your uploaded CSV path here.
CSV_PATH = "/content/data.csv"

# 3) Run the complete research pipeline.
import sys
sys.path.insert(0, "/content/ORACLE-X/src")
from oracle_x.run_research import run

market, features, power, result, output = run(
    CSV_PATH,
    population_size=64,
    generations=10,
    horizon=12,
    capital=100.0,
    seed=42,
)

print("\n=== ORACLE-X RESULT ===")
print("Market:", market.shape)
print("Features:", features.shape)
print("Best robot:", result.evolution.best.dna.to_dict())
print("Best fitness:", result.evolution.best.metrics["fitness"])
print("Accepted robots:", sum(r.accepted for r in result.validated))
print("Artifacts:", output)
