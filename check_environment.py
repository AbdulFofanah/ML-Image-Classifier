# Environment check
import sys
import torch
import torchvision
import torchinfo
import tqdm

def main():
    print(
        f" Package Check \n"
        f"Python version: {sys.version}\n"
        f"PyTorch version: {torch.__version__}\n"
        f"Torchvision version: {torchvision.__version__}\n"
        f"torchinfo: successfully imported\n"
        f"tqdm: successfully imported"
    )

    print(
        f"CUDA and GPU Check\n"
        f"CUDA available: {torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}\n"
            f"CUDA version used by PyTorch: {torch.version.cuda}"
        )
    else:
        print("GPU: No GPU found")

if __name__ == "__main__":
    main()