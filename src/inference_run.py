# infer_run.py
from inference import run_inference

if __name__ == "__main__":
    run_inference(
        ckpt_path="path/to/best.ckpt",
        train_meta_path="train_meta.csv",
        train_img_root="train_images",
        test_meta_path="test_meta.csv",
        test_img_root="test_images",
        bbox_mode="drop",
        unknown_ratio=0.2,
        device="cuda",
        image_size=224,
        submission_path="submission.csv",
    )
