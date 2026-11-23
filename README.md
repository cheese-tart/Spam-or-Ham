# Spam or Ham?

Spam or Ham? classifies the text of an email or SMS message as `spam` or `ham` (legitimate). The React interface lets you paste a message, select its source, and check it with one click. A FastAPI service makes the prediction using a trained scikit-learn pipeline.

## How it works

1. The browser sends the message text and its source (`email` or `sms`) to `POST /predict`.
2. FastAPI validates the request and passes both fields to the model. The API loads the saved pipeline once at startup.
3. The pipeline extracts word unigrams and bigrams with TF-IDF, one-hot encodes the message source, and uses a linear support vector machine to predict a label.
4. The interface displays **Likely spam** or **Likely ham**. Predictions can be wrong, especially for messages that differ from the training data.

## Project layout

| Path | Purpose |
| --- | --- |
| [frontend/](frontend/) | React web interface, built and served locally with Vite. |
| [backend/app/](backend/app/) | FastAPI endpoint, request validation, and model loading. |
| [backend/benchmarks/benchmark_api.py](backend/benchmarks/benchmark_api.py) | Measures prediction latency, throughput, and errors against a running API. |
| [ml/preprocess.py](ml/preprocess.py) | Cleans the email and SMS datasets and produces training and test CSVs. |
| [ml/train.py](ml/train.py) | Trains and exports the selected model from local CSVs or S3. |
| [ml/notebooks/exploration.ipynb](ml/notebooks/exploration.ipynb) | Explores class balance, message lengths, vocabulary, and differences between email and SMS. |
| [ml/notebooks/model_training.ipynb](ml/notebooks/model_training.ipynb) | Compares models, evaluates the selected pipeline, and exports the model. |
| `ml/data/raw/`, `ml/data/processed/`, `ml/artifacts/` | Local input datasets, processed splits, and the generated model file. |

## Run locally

Use Python 3.13, Node.js 22.13+ on the 22.x line or Node.js 24+, and npm. The examples use PowerShell and start in the repository root.

### 1. Put the model in place

The API expects the trained pipeline at `ml/artifacts/spam_classifier_linear_svm.joblib`. If you already have this file, place it there or set `MODEL_PATH` in the backend terminal to its absolute path before starting the API.

```powershell
$env:MODEL_PATH = 'C:\path\to\spam_classifier_linear_svm.joblib'
```

The datasets and model file are excluded from Git. For a fresh checkout without the model, follow [Train the model](#train-the-model) first.

### 2. Start the API

In the first terminal, run the following commands and leave the process running.

```powershell
py -3.13 -m venv backend/.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
Set-Location backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The API listens at [http://127.0.0.1:8000](http://127.0.0.1:8000). FastAPI's interactive documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3. Start the web interface

In a second terminal, run the following commands from the repository root.

```powershell
Set-Location frontend
npm ci
npm run dev
```

Open [http://localhost:5173](http://localhost:5173), or use the URL printed by Vite. Select **SMS** or **Email**, paste your message, and choose **Check for spam**. The Vite dev server proxies `/predict` to `http://127.0.0.1:8000`.

On macOS or Linux, use `python3.13` to create the virtual environments, replace `Scripts\python.exe` with `bin/python` in the commands, and use `cd` to change directories.

## Prediction API

| Field | Type | Accepted values |
| --- | --- | --- |
| `text` | String | Message text with at least one non-whitespace character. |
| `source` | String | `email` or `sms`. |

```powershell
$body = @{
    text = 'Hi, are we still meeting tomorrow?'
    source = 'email'
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
    -Uri 'http://127.0.0.1:8000/predict' `
    -ContentType 'application/json' `
    -Body $body
```

A successful response returns HTTP 200 and a JSON object with one field, `label`. The model returns either `ham` or `spam`. An example response is shown below.

```json
{
  "label": "ham"
}
```

Missing fields, unsupported source values, and blank messages produce an HTTP 422 validation error.

`GET /health` returns HTTP 200 with `{"status":"ok"}` once the API has loaded the model and is accepting requests.

## Train the model

### 1. Prepare the input data

Place two labeled CSVs in `ml/data/raw/`. Both datasets use `ham` and `spam` as label values.

| File | Text column | Label column | Encoding |
| --- | --- | --- | --- |
| `spam_ham_email.csv` | `text` | `label` | UTF-8 |
| `spam_ham_sms.csv` | `v2` | `v1` | Latin-1 |

### 2. Preprocess and split the data

From the repository root, create an ML environment and run the preprocessing script. The ML requirements pin the same model dependencies as the backend.

```powershell
py -3.13 -m venv ml/.venv
.\ml\.venv\Scripts\python.exe -m pip install -r ml/requirements.txt jupyterlab
.\ml\.venv\Scripts\python.exe ml/preprocess.py
```

The script standardizes the inputs to `text`, `label`, and `source`. It removes missing and blank messages, discards text/source pairs with conflicting labels, and deduplicates within each source. The split holds out 20% of unique message-text groups, keeping identical text in the same partition even when it occurs in both sources. It writes `train.csv` and `test.csv` to `ml/data/processed/`.

### 3. Run the training notebook

Launch JupyterLab from `ml/notebooks/` so the notebook's relative paths resolve correctly.

```powershell
Set-Location ml/notebooks
..\.venv\Scripts\python.exe -m jupyterlab
```

Open `model_training.ipynb`, select the Python kernel from the ML virtual environment, and run all cells. The notebook compares a most-frequent-class baseline, logistic regression, Complement Naive Bayes, and a linear SVM. It uses five-fold stratified group cross-validation with message text as the grouping key, and tunes models using spam F1. The final cell saves the fitted pipeline to `ml/artifacts/spam_classifier_linear_svm.joblib`. Restart the API after replacing the artifact so it loads the new model.

### Recorded evaluation

The saved [training notebook](ml/notebooks/model_training.ipynb) reports the following scores for the selected linear SVM (`C=300`) on a held-out set of 2,033 messages. Scores may vary with different datasets or library versions.

| Metric | Held-out result |
| --- | ---: |
| Accuracy | 98.38% |
| Spam precision | 97.20% |
| Spam recall | 94.55% |
| Spam F1 | 95.86% |

### Train from the command line

With the dependencies and raw CSVs in place, run this command from the repository root.

```powershell
.\ml\.venv\Scripts\python.exe ml/train.py
```

The script cleans and splits the raw CSVs, fits the linear SVM with the notebook's selected settings (`C=300`), prints held-out metrics, and saves the model to `ml/artifacts/spam_classifier_linear_svm.joblib`. Use `--raw-data-dir` or `--model-path` to change the local paths.

## Build and benchmark

From `frontend/`, run `npm run lint` to check the source and `npm run build` to generate the static frontend in `frontend/dist/`. When serving that build, route `/predict` on the frontend's origin to the FastAPI backend. The Vite development proxy is configured in [frontend/vite.config.js](frontend/vite.config.js).

With the API running, benchmark it from the repository root.

```powershell
.\backend\.venv\Scripts\python.exe backend/benchmarks/benchmark_api.py --requests 200 --warmup 10 --concurrency 4
```

The script alternates between example email and SMS requests and reports median (p50) and p95 latency, successful requests per second, and the error rate. Set `--url` to target a different prediction endpoint or `--timeout` to change the HTTP timeout.

## Run the API with Docker

With Docker running and the model already available, build the image and mount the artifact directory from the repository root.

```powershell
docker build -t spam-classifier-api ./backend
$artifacts = (Resolve-Path './ml/artifacts').Path
docker run --rm -p 8000:8000 --mount "type=bind,source=$artifacts,target=/models,readonly" spam-classifier-api
```

The container serves the API on port 8000 and loads `/models/spam_classifier_linear_svm.joblib`. Start the frontend separately using the local instructions above. The [ML Dockerfile](ml/Dockerfile) provides a separate JupyterLab environment for the notebooks.

## Deploy the API to AWS

From the repository root, build the backend's `deploy` target. This copies the saved model from `ml/artifacts/` into the image.

```powershell
docker buildx build --load --platform linux/amd64 --target deploy --build-context model=./ml/artifacts -t spam-classifier-api:deploy ./backend
docker run --rm -p 8000:8000 spam-classifier-api:deploy
```

In another terminal, check `http://127.0.0.1:8000/health` and try the prediction request above. Push the image to Amazon ECR and select it when you create an ECS Express Mode service. Set the container port to `8000` and the health check path to `/health`. When you have a new model, rebuild and publish the backend image with a new tag and update the service to use it. [Push an image to Amazon ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push.html) · [Create an ECS Express Mode service](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-first-run.html)

### Run ML training as an AWS job

Build the separate training image and push it to ECR.

```powershell
docker buildx build --load --platform linux/amd64 --target training -t spam-classifier-training ./ml
```

Put both raw CSVs under the same private S3 prefix. Create an ECS Fargate task definition using the training image, then set its container command to the following JSON array, replacing the example S3 locations with your own.

```json
[
  "--s3-input", "s3://my-training-bucket/raw/",
  "--s3-output", "s3://my-training-bucket/models/run-001/spam_classifier_linear_svm.joblib"
]
```

Give the task role permission to read the two input objects and write the output object. Run the task when you want a new model. It exits after uploading the artifact. To promote the model, download it to `ml/artifacts/`, then build and deploy a new backend image using the commands above. [Run an ECS standalone task](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/standalone-task-create.html) · [ECS task role permissions](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-iam-roles.html)

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| The API fails to start with `Model file not found`. | Check the default artifact path or `MODEL_PATH`. For Docker, check the `/models` mount or build the `deploy` target with the artifact in its build context. |
| The API fails to load the model. | Check that training and serving use the pinned versions of scikit-learn, NumPy, SciPy, pandas, and Joblib from `backend/requirements.txt`. |
| The browser shows a prediction error. | Verify the API is listening on port 8000 and that `POST /predict` works in FastAPI's interactive docs. |
| Preprocessing cannot read an input. | Check both CSV filenames, column names, and encodings against the table above. |
| The training notebook cannot find `train.csv` or `test.csv`. | Run preprocessing and launch JupyterLab from `ml/notebooks/`. |
