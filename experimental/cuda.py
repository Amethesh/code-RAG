import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Using GPU: {device_name} (CUDA device {current_device})")
    else:
        print("CUDA is not available. Running on CPU.")


check_cuda()

print(torch.cuda.device_count())
print (torch.cuda.get_device_name(0))
