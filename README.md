# Crop Disease Detection Backend

This project exposes your trained Keras model through a small Flask API.

## Files
- `plant_disease_detector_model.keras`: trained model
- `labels.json`: class labels in the same order as training
- `app.py`: Flask backend with `/health` and `/predict`
- `test_inference.py`: one-image CLI test script
- `requirements.txt`: Python dependencies

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run the backend

```powershell
python app.py
```

The server runs on `http://127.0.0.1:5000`.

## Health check

Open:

```text
http://127.0.0.1:5000/health
```

## Predict with an image

Use multipart form-data with the `file` field:

```powershell
curl -X POST "http://127.0.0.1:5000/predict?top_k=3" -F "file=@abr.jpeg"
```

## Run local inference test

```powershell
python test_inference.py --image abr.jpeg --top-k 3
```

## Expected behavior
- Input image is resized to `224 x 224`
- Model output has `34` classes
- Labels come from `labels.json`
- Response includes `top_prediction` and `top_k`
