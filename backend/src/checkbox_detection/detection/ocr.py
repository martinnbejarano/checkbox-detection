"""The only place that builds a RapidOCR instance, shared by stage 0 (`orient.py`) and stage 3
(`label.py`)."""

import numpy as np
from rapidocr import RapidOCR


def load_ocr(threads: int) -> RapidOCR:
    ocr = RapidOCR(
        params={
            "Global.use_det": False,
            "Global.use_cls": False,
            "Global.log_level": "warning",
            "Rec.rec_batch_num": 20,  # orient.py reads its 2 x 10 probes in one batch
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )

    ocr(np.full((32, 320), 255, np.uint8), use_det=False, use_cls=False, use_rec=True)
    return ocr
