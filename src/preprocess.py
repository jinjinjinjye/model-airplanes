# Load raw CSV → extract narratives → normalize obvious whitespace → randomly split by report → save

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Perform minimal cleaning without altering linguistic content."""
    return " ".join(str(text).split())


def preprocess(config_path="config.yaml"):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]

    raw_path = Path(config["data"]["raw_path"])
    output_dir = Path(config["data"]["processed_dir"])

    id_column = config["data"]["raw_id_column"]
    text_column = config["data"]["raw_text_column"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Only load what Part 1 actually needs.
    df = pd.read_csv(
        raw_path,
        usecols=[id_column, text_column],
    )

    df = df.rename(
        columns={
            id_column: "acn",
            text_column: "text",
        }
    )

    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].map(clean_text)

    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # 80% train, 20% temporary.
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )

    # Split temporary data equally -> 10% validation, 10% test.
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Total:      {len(df):,}")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test:       {len(test_df):,}")


if __name__ == "__main__":
    preprocess()