"""Route G: TTA inference for ECG beat classification."""
from .tta import (
    tta_predict_sliding, tta_predict_batch_sliding,
    tta_augmented_views, tta_predict_augmented,
    tta_predict_batch_augmented,
    multi_beat_confirm, tta_evaluate, tta_streaming_buffer,
    tta_sliding_window,
)
