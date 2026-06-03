# train.py — run this once to build data and train the model

import sys
from pathlib import Path

# add the src folder to Python's path so it can find property_intel
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from property_intel.logging_config import setup_logging
from property_intel.pipeline import Pipeline
from property_intel.features import build_features
from property_intel.modeling import train_and_evaluate, explain_model


def main() -> None:
    # config.yaml lives in the project root, so we change to it first
    import os
    os.chdir(Path(__file__).resolve().parent.parent.parent)

    logger = setup_logging()
    logger.info("=== Property Intelligence Engine starting ===")

    pipeline = Pipeline()
    df = pipeline.run()

    df = build_features(df)

    results = train_and_evaluate(df)

    print("\n=== Model Results ===")
    print(f"RMSE : R{results['rmse']:,.0f}")
    print(f"MAE  : R{results['mae']:,.0f}")
    print(f"R2   : {results['r2']:.3f}")
    print("\n" + explain_model(results["importance"]))

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()