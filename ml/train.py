"""Train and export the selected spam classifier from local files or S3."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from preprocess import (
    BASE_DIR,
    EMAIL_FILENAME,
    RAW_DATA_DIR,
    SMS_FILENAME,
    clean_data,
    load_data,
    split_data,
    validate_splits,
)

DEFAULT_MODEL_PATH = BASE_DIR / "artifacts" / "spam_classifier_linear_svm.joblib"
FEATURES = ["text", "source"]


def train_pipeline(raw_data_dir):
    messages = clean_data(load_data(raw_data_dir))
    train_df, test_df = split_data(messages)
    validate_splits(train_df, test_df)

    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2)), "text"),
        ("source", OneHotEncoder(handle_unknown="ignore"), ["source"]),
    ])
    pipeline = make_pipeline(
        preprocessor, LinearSVC(C=300, max_iter=20000, random_state=123)
    )
    pipeline.fit(train_df[FEATURES], train_df["label"])
    predictions = pipeline.predict(test_df[FEATURES])
    print(classification_report(test_df["label"], predictions, zero_division=0))
    return pipeline


def s3_location(value):
    scheme, _, location = value.partition("://")
    bucket, _, key = location.partition("/")
    if scheme != "s3" or not bucket:
        raise argparse.ArgumentTypeError("expected s3://bucket/path")
    return bucket, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    inputs.add_argument("--s3-input", type=s3_location, metavar="S3_URI")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    outputs.add_argument("--s3-output", type=s3_location, metavar="S3_URI")
    args = parser.parse_args()
    if args.s3_output and (not args.s3_output[1] or args.s3_output[1].endswith("/")):
        parser.error("--s3-output must include an object key")

    s3 = None
    if args.s3_input or args.s3_output:
        import boto3

        s3 = boto3.client("s3")

    # Keep temporary output on the same filesystem as a local model destination
    # so a failed training run cannot leave a partial model in its place.
    output_dir = None
    if not args.s3_output:
        output_dir = args.model_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=output_dir) as temp:
        work_dir = Path(temp)
        raw_data_dir = args.raw_data_dir
        if args.s3_input:
            bucket, prefix = args.s3_input
            prefix = prefix.rstrip("/")
            raw_data_dir = work_dir / "raw"
            raw_data_dir.mkdir()
            for filename in (EMAIL_FILENAME, SMS_FILENAME):
                key = f"{prefix}/{filename}" if prefix else filename
                s3.download_file(bucket, key, str(raw_data_dir / filename))

        pipeline = train_pipeline(raw_data_dir)
        artifact = work_dir / DEFAULT_MODEL_PATH.name
        joblib.dump(pipeline, artifact)
        if args.s3_output:
            bucket, key = args.s3_output
            s3.upload_file(str(artifact), bucket, key)
            print(f"Saved model to s3://{bucket}/{key}")
        else:
            artifact.replace(args.model_path)
            print(f"Saved model to {args.model_path}")


if __name__ == "__main__":
    main()
