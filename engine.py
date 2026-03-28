"""
VECTOR VXP2 — Inference Engine
Loads a trained Random Forest model and predicts Remaining Useful Life (RUL)
with physics-based safety validation for turbofan engine sensor data.
"""

import os
import joblib
import numpy as np


class VectorInference:
    """Core inference engine for turbofan engine health prediction."""

    # ---------- Physics constants for compressor P30/T30 check ----------
    # Expected ratio range derived from isentropic compressor relationships.
    # Values outside this band indicate sensor fault or physical implausibility.
    P30_T30_RATIO_LOW = 0.020   # lower bound (psi / °R)
    P30_T30_RATIO_HIGH = 0.065  # upper bound (psi / °R)

    # RUL threshold below which the engine is considered critical
    RUL_CRITICAL_THRESHOLD = 30

    def __init__(self, model_path: str = None):
        """
        Load the Random Forest model from disk.

        Parameters
        ----------
        model_path : str, optional
            Path to the serialised model file.
            Defaults to ``models/rf_v1.pkl`` relative to the project root.
        """
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "rf_v1.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Please place the trained Random Forest model in the /models directory."
            )

        self.model = joblib.load(model_path)
        self.model_path = model_path

    # ------------------------------------------------------------------ #
    #  Physics-based validation                                          #
    # ------------------------------------------------------------------ #
    def physics_guardrail(self, sensor_data: dict) -> tuple:
        """
        Validate that the P30 / T30 ratio falls within the expected
        compressor pressure-temperature relationship.

        Parameters
        ----------
        sensor_data : dict
            Must contain keys ``'P30'`` (total pressure at HPC outlet, psi)
            and ``'T30'`` (total temperature at HPC outlet, °R).

        Returns
        -------
        tuple[bool, str]
            ``(is_valid, message)`` where *is_valid* is True when the ratio
            is physically plausible.
        """
        sensor_data = {str(k).lower(): v for k, v in sensor_data.items()}
        p30 = sensor_data.get("p30")
        t30 = sensor_data.get("t30")

        if p30 is None or t30 is None:
            return False, "Missing P30 or T30 sensor reading."

        if t30 == 0:
            return False, "T30 is zero — invalid temperature reading."

        ratio = p30 / t30
        within_bounds = self.P30_T30_RATIO_LOW <= ratio <= self.P30_T30_RATIO_HIGH

        if within_bounds:
            message = (
                f"P30/T30 ratio ({ratio:.4f}) is within expected bounds "
                f"[{self.P30_T30_RATIO_LOW}, {self.P30_T30_RATIO_HIGH}]. "
                "Compressor readings are physically consistent."
            )
        else:
            message = (
                f"⚠ P30/T30 ratio ({ratio:.4f}) is OUTSIDE expected bounds "
                f"[{self.P30_T30_RATIO_LOW}, {self.P30_T30_RATIO_HIGH}]. "
                "Possible sensor fault or compressor anomaly detected."
            )

        return within_bounds, message

    # ------------------------------------------------------------------ #
    #  Health prediction                                                  #
    # ------------------------------------------------------------------ #
    def predict_rul(self, data_row: dict) -> float:
        """
        Predict Remaining Useful Life for a single data row.

        Parameters
        ----------
        data_row : dict
            A dictionary of feature values expected by the Random Forest
            model.

        Returns
        -------
        float
            Predicted RUL in cycles.
        """
        data_row = {str(k).lower(): v for k, v in data_row.items()}
        
        feature_values = []
        for i in range(1, 22):
            sensor_key = f"s{i}"
            val = data_row.get(sensor_key, 1500.0)
            feature_values.append(val)
            
        feature_array = np.array(feature_values, dtype=np.float64).reshape(1, -1)
        rul_prediction = float(self.model.predict(feature_array)[0])
        return round(rul_prediction, 2)
