"""
MLflow Model Registry — promotion script.

Compares all registered versions of the defect-predictor model,
selects the best one by AUC-ROC, and assigns the 'champion' alias.
Any previously aliased champion gets archived.

MLflow v3 uses aliases instead of stages (Staging/Production are deprecated).
Aliases used in this project:
  champion   — model currently serving production traffic
  challenger — model under evaluation (set externally by agent in Phase 3)

Usage:
    # Show registry state only
    uv run python -m training.pipelines.promote --dry-run

    # Promote best model by AUC-ROC
    uv run python -m training.pipelines.promote

    # Promote a specific version
    uv run python -m training.pipelines.promote --version 6
"""
import os
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import yaml
from pathlib import Path

load_dotenv()

CONFIG_PATH = Path("configs/model_config.yaml")

REGISTERED_MODEL = "defect-predictor"
CHAMPION_ALIAS = "champion"
METRIC_KEY = "auc_roc"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_all_versions_with_metrics(client: MlflowClient) -> list[dict]:
    """Fetch all model versions and join with their run metrics."""
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    results = []
    for v in versions:
        try:
            run = client.get_run(v.run_id)
            auc = run.data.metrics.get(METRIC_KEY)
            results.append({
                "version": int(v.version),
                "run_id": v.run_id,
                "run_name": run.info.run_name,
                "auc_roc": auc,
                "aliases": list(v.aliases),
                "tags": dict(v.tags),
            })
        except Exception:
            # Run may have been deleted
            results.append({
                "version": int(v.version),
                "run_id": v.run_id,
                "run_name": "unknown",
                "auc_roc": None,
                "aliases": list(v.aliases),
                "tags": {},
            })
    return sorted(results, key=lambda x: x["version"], reverse=True)


def print_registry_table(versions: list[dict]):
    print(f"\n{'Ver':>4}  {'Run Name':<22}  {'AUC-ROC':>8}  {'Aliases'}")
    print("-" * 58)
    for v in versions:
        auc = f"{v['auc_roc']:.4f}" if v["auc_roc"] else "  N/A  "
        aliases = ", ".join(v["aliases"]) if v["aliases"] else "-"
        print(f"  {v['version']:>2}  {v['run_name']:<22}  {auc:>8}  {aliases}")
    print()


def promote(version: int | None = None, dry_run: bool = False):
    config = load_config()
    min_auc = config["thresholds"]["min_auc_roc"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # ------------------------------------------------------------------ #
    # 1. Fetch all versions
    # ------------------------------------------------------------------ #
    versions = get_all_versions_with_metrics(client)

    print(f"\n=== MLflow Model Registry : {REGISTERED_MODEL} ===")
    print_registry_table(versions)

    # ------------------------------------------------------------------ #
    # 2. Identify current champion (if any)
    # ------------------------------------------------------------------ #
    current_champion = None
    for v in versions:
        if CHAMPION_ALIAS in v["aliases"]:
            current_champion = v
            break

    if current_champion:
        print(f"Current champion : v{current_champion['version']} "
              f"({current_champion['run_name']}) — AUC-ROC {current_champion['auc_roc']:.4f}")
    else:
        print("Current champion : None (no version aliased as 'champion' yet)")

    # ------------------------------------------------------------------ #
    # 3. Select version to promote
    # ------------------------------------------------------------------ #
    if version:
        # Explicit version requested
        candidate = next((v for v in versions if v["version"] == version), None)
        if not candidate:
            print(f"\nERROR: Version {version} not found in registry.")
            return
    else:
        # Auto-select: best AUC-ROC across all versions
        scored = [v for v in versions if v["auc_roc"] is not None]
        candidate = max(scored, key=lambda x: x["auc_roc"])

    print(f"\nCandidate for champion : v{candidate['version']} "
          f"({candidate['run_name']}) — AUC-ROC {candidate['auc_roc']:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Threshold check
    # ------------------------------------------------------------------ #
    if candidate["auc_roc"] < min_auc:
        print(f"\nREJECTED: AUC-ROC {candidate['auc_roc']:.4f} is below "
              f"min threshold {min_auc}. No promotion.")
        return

    if current_champion and candidate["version"] == current_champion["version"]:
        print("\nNo change: candidate is already the current champion.")
        return

    # ------------------------------------------------------------------ #
    # 5. Promote
    # ------------------------------------------------------------------ #
    if dry_run:
        print(f"\n[DRY RUN] Would promote v{candidate['version']} as '{CHAMPION_ALIAS}'.")
        if current_champion:
            print(f"[DRY RUN] Would remove '{CHAMPION_ALIAS}' alias from v{current_champion['version']}.")
        return

    # Remove champion alias from previous champion
    if current_champion:
        client.delete_registered_model_alias(
            name=REGISTERED_MODEL,
            alias=CHAMPION_ALIAS,
        )
        # Tag the old champion as archived
        client.set_model_version_tag(
            name=REGISTERED_MODEL,
            version=str(current_champion["version"]),
            key="status",
            value="archived",
        )
        print(f"\nArchived previous champion v{current_champion['version']}.")

    # Set champion alias on new version
    client.set_registered_model_alias(
        name=REGISTERED_MODEL,
        alias=CHAMPION_ALIAS,
        version=str(candidate["version"]),
    )

    # Tag and describe the new champion
    client.set_model_version_tag(
        name=REGISTERED_MODEL,
        version=str(candidate["version"]),
        key="status",
        value="champion",
    )
    client.update_model_version(
        name=REGISTERED_MODEL,
        version=str(candidate["version"]),
        description=(
            f"Champion model. AUC-ROC: {candidate['auc_roc']:.4f}. "
            f"Run: {candidate['run_name']} ({candidate['run_id'][:8]})."
        ),
    )

    print(f"\nPROMOTED: v{candidate['version']} ({candidate['run_name']}) "
          f"is now '{CHAMPION_ALIAS}'.")
    print(f"\nTo serve this model:")
    print(f"  uv run python -m serving.server")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=None,
                        help="Specific model version to promote (default: best by AUC-ROC)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview promotion without making changes")
    args = parser.parse_args()
    promote(version=args.version, dry_run=args.dry_run)
