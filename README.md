# ATLAS Paper Track release

This archive contains the audited offline study, reproducible CPU tables and figures, the Atlas helper library, and two bounded notebooks. `paper/notebooks/reproduce.ipynb` was executed locally with a clean Python environment. `paper/notebooks/model_preflight.ipynb` is a private Kaggle GPU preflight and is intentionally bounded to E4B first-token compatibility.

The archive contains derived evidence and source code only. It excludes original repository snapshots, raw JUnit trees, model weights, competition wheelhouses, credentials, and build metadata. Obtain competition inputs through Kaggle and observe their licenses. `artifact-manifest.json` records SHA256 and byte size for every project artifact, includes the wrapper documents, and explains why its own self-hash is omitted.

The paper's central numbers are descriptive diagnostics: 129 task records, 774 nonbaseline states, 80 failed perturbations, 642 known paired target/selected states, 611 agreements, a 39.55% paired command-time reduction, and 124 deterministic localization tasks. Unknown required-test coverage means no official resolved score is asserted. The Kaggle E4B preflight completed installation but failed first-token generation on one T4 because Triton attention requested 98,304 bytes of shared memory against a 65,536-byte hardware limit; the agent was not run.

Run the local checks from the project root:

```powershell
python -m pytest paper/tests atlas/tests -q -p no:cacheprovider
python paper/scripts/final_audit.py --help
```

The repository snapshot directory under `results/p1_workspace` is historical evidence and is deliberately excluded from this archive.
