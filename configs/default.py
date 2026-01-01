@dataclass
class Config:
    # Model settings
    vit_model: str = "tiny"      # "tiny" or "small"
    decoder_size: str = "small"  # "small", "medium", "large"
    
    # Training settings
    seed: int = 42
    classifier_epochs: int = 10
    decoder_epochs: int = 20
    segmentation_epochs: int = 10
    
    # Weakly-supervised settings
    gt_ratio: float = 0.0        # 0.0 = fully weakly-supervised
    cam_threshold: float = 0.25  # Mask binarization threshold