import os
import sys

# Vercel's filesystem is read-only except /tmp — redirect the CNN model
# download target there (main.py honors CNN_MODEL_PATH if set).
os.environ.setdefault("CNN_MODEL_PATH", "/tmp/alzheimer_cnn_model.pkl")

# Defer CNN loading to the first /predict-mri call instead of failing the
# whole app at cold start if CNN_MODEL_URL isn't set/working yet — lets
# /predict, /patient-record, and /health work regardless of that endpoint's
# status. See main.py's _LAZY_CNN_LOAD for details.
os.environ.setdefault("LAZY_CNN_LOAD", "true")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from alzimer_backend.main import app  # noqa: E402
