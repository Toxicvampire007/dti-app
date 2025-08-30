import io
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st

# Optional RDKit featurization (works if your model expects fingerprints)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKit_OK = True
except Exception:
    RDKit_OK = False

st.set_page_config(page_title="DTI Predictor", page_icon="🧪", layout="wide")

# ---------- CACHED LOADERS ----------
@st.cache_resource
def load_config(path="config.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler(path="best_label_scaler.pkl"):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

@st.cache_resource
def load_model(config, path="model.pt"):
    device = torch.device("cpu")
    # 1) Try TorchScript first (works without class defs)
    try:
        m = torch.jit.load(path, map_location=device)
        m.eval()
        return m, "torchscript"
    except Exception:
        pass
    # 2) Fallback: state_dict + architecture you provide
    try:
        from model_def import build_model
    except Exception as e:
        raise RuntimeError(
            "model.pt is not TorchScript and no build_model found. "
            "Add your model architecture to model_def.py."
        ) from e
    # Either load a plain state_dict or a wrapper dict
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model = build_model(config)
    model.load_state_dict(sd)
    model.eval()
    return model, "state_dict"

# ---------- FEATURE ENGINEERING ----------
def smiles_to_morgan(smiles: str, n_bits=2048, radius=2):
    if not RDKit_OK:
        raise RuntimeError("RDKit not available; cannot featurize SMILES. "
                           "Either add rdkit-pypi to requirements or supply precomputed features.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def make_feature_vector(row, config):
    """
    Decide how to build X based on config.
    Priority:
      1) If config says 'featurizer' == 'morgan', compute from SMILES.
      2) Else, expect precomputed numeric feature columns listed in config['feature_names'].
    """
    feat_type = str(config.get("featurizer", "morgan")).lower()
    if feat_type == "morgan":
        n_bits = int(config.get("n_bits", 2048))
        radius = int(config.get("radius", 2))
        smi = row["SMILES"]
        return smiles_to_morgan(smi, n_bits=n_bits, radius=radius)
    else:
        names = config.get("feature_names")
        if not names:
            raise RuntimeError("No feature_names in config and featurizer not 'morgan'.")
        return row[names].to_numpy(dtype=np.float32)

def is_regression(config):
    return str(config.get("task", "regression")).lower() in {"regression", "reg"}

def apply_postprocess(y_pred, config, scaler):
    y = y_pred
    # Inverse scaling for regression if labels were scaled during training
    if is_regression(config) and bool(config.get("label_scaled", True)) and scaler is not None:
        y = scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()
    # For classification, optional sigmoid & thresholding
    if not is_regression(config):
        if bool(config.get("apply_sigmoid", True)):
            y = 1 / (1 + np.exp(-np.asarray(y)))
    return y

# ---------- INFERENCE ----------
def predict_single(model, config, scaler, smiles=None, row=None):
    if smiles is not None:
        row = pd.Series({"SMILES": smiles})
    x = make_feature_vector(row, config)
    x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)  # [1, F]
    with torch.no_grad():
        y = model(x).cpu().numpy().ravel()
    y = apply_postprocess(y, config, scaler)
    if is_regression(config):
        return float(y[0])
    else:
        thr = float(config.get("threshold", 0.5))
        return float(y[0]), int(y[0] >= thr)

def predict_batch(model, config, scaler, df):
    outputs = []
    for _, row in df.iterrows():
        try:
            y = predict_single(model, config, scaler, row=row)
            outputs.append(y)
        except Exception as e:
            outputs.append(np.nan if is_regression(config) else (np.nan, np.nan))
    if is_regression(config):
        df_out = df.copy()
        df_out["prediction"] = outputs
    else:
        probs, labels = zip(*outputs)
        df_out = df.copy()
        df_out["probability"] = probs
        df_out["label"] = labels
    return df_out

# ---------- UI ----------
def main():
    st.title("🧪 DTI Predictor")
    st.caption("Streamlit app for drug–target interaction prediction")

    config = load_config()
    scaler = None
    try:
        scaler = load_scaler()
    except Exception:
        pass
    model, mtype = load_model(config)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Single Prediction")
        smi = st.text_input("SMILES", placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O")
        if st.button("Predict", use_container_width=True) and smi:
            try:
                if is_regression(config):
                    y = predict_single(model, config, scaler, smiles=smi)
                    st.success(f"Predicted activity: {y:.4f}")
                else:
                    prob, label = predict_single(model, config, scaler, smiles=smi)
                    st.success(f"Probability: {prob:.4f}  →  Label: {label}")
            except Exception as e:
                st.error(f"Error: {e}")

    with right:
        st.subheader("Batch Prediction (.csv)")
        st.write("CSV with a `SMILES` column or with all feature columns in `config['feature_names']`.")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
                st.dataframe(df.head())
                if st.button("Run batch", use_container_width=True):
                    out = predict_batch(model, config, scaler, df)
                    st.success(f"Done. {len(out)} rows predicted.")
                    st.dataframe(out.head())
                    buf = io.BytesIO()
                    out.to_csv(buf, index=False)
                    st.download_button("Download predictions", buf.getvalue(),
                                       file_name="predictions.csv", mime="text/csv",
                                       use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    with st.expander("Model Info"):
        st.json({
            "inference_model": mtype,
            "task": str(config.get("task", "regression")),
            "featurizer": str(config.get("featurizer", "morgan")),
            "notes": "If model.pt is not TorchScript, ship model_def.py with build_model(config)."
        })

if __name__ == "__main__":
    main()
