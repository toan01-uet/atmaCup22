# infer_run.py
import os
from pathlib import Path
from src.inference import run_inference
from src.utils.logger import get_logger

if __name__ == "__main__":
    # Convert to absolute paths to avoid path issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup logging
    outputs_dir = Path(script_dir) / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    logger = get_logger("inference", file_dir=outputs_dir)
    
    logger.info("="*80)
    logger.info("Starting Inference Pipeline")
    logger.info("="*80)
    
    train_meta_path = os.path.join(script_dir, "inputs", "atmaCup22_2nd_meta", "train_meta.csv")
    train_img_root = os.path.join(script_dir, "inputs", "images")
    test_meta_path = os.path.join(script_dir, "inputs", "atmaCup22_2nd_meta", "test_meta.csv")
    # Test data uses crops directory (pre-cropped images)
    test_img_root = os.path.join(script_dir, "inputs")
    submission_path = os.path.join(script_dir, "outputs", "submission.csv")
    ckpt_path = "path/to/best.ckpt"
    
    logger.info(f"Checkpoint path: {ckpt_path}")
    logger.info(f"Train metadata: {train_meta_path}")
    logger.info(f"Test metadata: {test_meta_path}")
    logger.info(f"Submission output: {submission_path}")
    logger.info(f"BBox mode: drop")
    logger.info(f"Unknown ratio: 0.2")
    logger.info(f"Device: cuda")
    logger.info(f"Image size: 224")
    
    run_inference(
        ckpt_path=ckpt_path,
        train_meta_path=train_meta_path,
        train_img_root=train_img_root,
        test_meta_path=test_meta_path,
        test_img_root=test_img_root,
        bbox_mode="drop",
        unknown_ratio=0.2,
        device="cuda",
        image_size=224,
        submission_path=submission_path,
        logger=logger,
    )
    
    logger.info("="*80)
    logger.info("Inference Complete!")
    logger.info("="*80)
