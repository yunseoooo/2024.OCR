# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from .DPtext_detr import TransformerPureDetector
from .DPtext_detr_aiaw import TransformerPureDetector
# from .DPtext_detr_aiaw_no_low_high import TransformerPureDetector
# from .DPtext_detr_aiaw_detection import TransformerPureDetector
# from .DPtext_detr_1stage_2stage import TransformerPureDetector
# from .DPtext_detr_inverse import TransformerPureDetector
# from .DPtext_detr_pro import TransformerPureDetector
# from .DPtext_detr_rec_det import TransformerPureDetector
# from .DPtext_detr_rec_det_each import TransformerPureDetector
# from .DPtext_detr_cross_attention import TransformerPureDetector
# from .DPtext_detr_cross_attention_2 import TransformerPureDetector
# from .DPtext_detr_add_Bridge import TransformerPureDetector
# from .DPtext_detr_add_Bridge_2 import TransformerPureDetector
# from .DPtext_detr_no_Bridge import TransformerPureDetector
from .TESTR import TransformerDetector

_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
