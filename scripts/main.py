"""
Entry point: orchestrates full pipeline

loss_threshold = 0.05  best for small medium use_finecam_only=True loss_threshold=0.01 10 epochs training_supervised_ws
loss_threshold = 0.35  best for tiny medium use_finecam_only=False 4 epochs training_supervised_ws
loss_threshold = 0.25 best for tiny large use_finecam_only=False 10 epochs training_supervised_ws
loss_threshold = 0.01  best for small medium use_finecam_only=True loss_threshold=0.01 10 epochs training_supervised_ws
loss_threshold = 0.061 best for small large use_finecam_only=False loss_threshold=0.061 10 epochs training_supervised_ws
"""

import argparse
import split_data
import train_ViT_classification
import build_fine_cams
import train_CNN_decoder
import build_masks
import train_supervised_gt
import train_supervised_ws
import evaluation
from utils.utils import set_seed
from torchvision import transforms



def main(train=False, vit_model="tiny", decoder_size="medium", seed=42, loss_threshold=0.35, epochs=4, gt_ratio=0.0, use_finecam_only=False, train_full_gt=False):
    set_seed(seed)

    if train:
        print(f"seed: {seed}")
        print(f"loss_threshold: {loss_threshold}")
        print(f"use_finecam_only: {use_finecam_only}")
        print(f"train_full_gt: {train_full_gt}")
    print("=" * 50)
    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    if train:
        print("Starting training process...")
        print("=" * 50)

        print("Splitting data into train and test sets...")
        split_data.main(seed=seed)
        print("=" * 50)

        print("Training ViT model for image classification...")
        train_ViT_classification.main(seed=seed, vit_model=vit_model, transform=transform)
        print("=" * 50)

        print("Building FineCAMs...")
        build_fine_cams.main(vit_model=vit_model, transform=transform)
        print("=" * 50)

        print("Training CNN decoder...")
        train_CNN_decoder.main(seed=seed, decoder_size=decoder_size, vit_model=vit_model, transform=transform)
        print("=" * 50)

        print("Building masks...")
        build_masks.main(vit_model=vit_model, decoder_size=decoder_size, loss_threshold=loss_threshold, use_finecam_only=use_finecam_only, transform=transform, plot=False)
        print("=" * 50)

        if train_full_gt:
            print("Training supervised models...")
            print("Training Supervised Model with the Ground Truth Labels model...")
            train_supervised_gt.main(seed=seed)
            print("=" * 50)

        print("Training Supervised Model with the Pseudo Masks model...")
        train_supervised_ws.main(vit_model=vit_model, decoder_size=decoder_size, seed=seed, transform=transform, epochs=epochs, gt_ratio=gt_ratio)
        
        print("=" * 50)

    print("Testing results...")
    evaluation.main(vit_model=vit_model, decoder_size=decoder_size, gt_ratio=gt_ratio, plot=False)
    print("=" * 50)

    print("All processes completed successfully.")


def get_model_paths(vit_model, decoder_size, seed):
    """Generate consistent model paths based on configuration."""
    return {
        "vit": f"models/vit_{vit_model}_seed{seed}.pth",
        "decoder": f"models/decoder_{vit_model}_{decoder_size}_seed{seed}.pth",
        "decoder_bestloss": f"models/decoder_{vit_model}_{decoder_size}_seed{seed}_bestloss.pth",
        "segmentation": f"models/segmentation_{vit_model}_{decoder_size}_seed{seed}.pth",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument("--train", action="store_true", help="Run training process")
    
    # Model configuration
    parser.add_argument("--vit_model", type=str, default="tiny", choices=["tiny", "small"],
                        help="ViT model size (default: tiny)")
    parser.add_argument("--decoder_size", type=str, default="medium", choices=["small", "medium", "large"],
                        help="Decoder size (default: medium)")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--loss_threshold", type=float, default=0.35,
                        help="Loss threshold for mask building (default: 0.35)")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of epochs for supervised training (default: 4)")
    parser.add_argument("--gt_ratio", type=float, default=0.0,
                        help="Ground truth ratio (default: 0.0)")
    
    # Flags
    parser.add_argument("--use_finecam_only", action="store_true",
                        help="Use FineCAM only without refinement")
    parser.add_argument("--train_full_gt", action="store_true",
                        help="Train with full ground truth")
    
    args = parser.parse_args()
    
    print("configurations:")
    print(f"  vit_model: {args.vit_model}")
    print(f"  decoder_size: {args.decoder_size}")
    print(f"  gt_ratio: {args.gt_ratio}")
    print(f"  seed: {args.seed}")
    print(f"  loss_threshold: {args.loss_threshold}")
    print(f"  epochs: {args.epochs}")
    print(f"  use_finecam_only: {args.use_finecam_only}")
    print(f"  train_full_gt: {args.train_full_gt}")
    
    main(
        train=args.train,
        vit_model=args.vit_model,
        decoder_size=args.decoder_size,
        seed=args.seed,
        loss_threshold=args.loss_threshold,
        epochs=args.epochs,
        gt_ratio=args.gt_ratio,
        use_finecam_only=args.use_finecam_only,
        train_full_gt=args.train_full_gt
    )
