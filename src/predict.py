import json
import joblib
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
INPUT_PATH = PROJECT_ROOT / "input.json"
OUTPUT_PATH = PROJECT_ROOT / "output.json"

# classification threshold
THRESHOLD = 0.3


def predict(input_path, model_path,scaler_path, output_path):
    
    """
    Load input data and trained model, then perform prediction.
    """

    # Load model
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    

    # Load JSON input
    with open(input_path, "r") as file:
        input_data = json.load(file)

    # Convert JSON to DataFrame
    X = pd.DataFrame([input_data])
    
    X_scaled = scaler.transform(X)

    # Predict probability
    probability = model.predict_proba(X_scaled)[0, 1]

    # Classification using threshold
    class_id = int(probability >= THRESHOLD)

    # Convert class ID to label
    prediction = "Fraud" if class_id == 1 else "Normal"

    # Create output
    result = {
        "prediction": prediction,
        "class_id": class_id,
        "probability": round(float(probability), 4),
        "threshold": THRESHOLD,
        "status": "success"
    }

    # Save output JSON
    with open(output_path, "w") as file:
        json.dump(result, file, indent=4)

    return result


if __name__ == "__main__":
    result = predict(
        INPUT_PATH,
        MODEL_PATH,
        SCALER_PATH,
        OUTPUT_PATH
    )

    print(json.dumps(result, indent=4))