import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from Models.ARSR.ARSR import ARSRNetwork

# Training hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARN_RATE = 0.0002
COLOR_SPACE = 'rgb'

# Define the RGB to YUV conversion matrix and its inverse
rgb_to_yuv = torch.tensor([
    [0.299, 0.587, 0.114],
    [-0.14713, -0.28886, 0.436],
    [0.615, -0.51499, -0.10001]
])

yuv_to_rgb = torch.inverse(rgb_to_yuv)


def yuv_to_rgb_batch(yuv_batch):
    yuv_batch = yuv_batch.permute(0, 2, 3, 1)
    rgb_batch = torch.matmul(yuv_batch, yuv_to_rgb.T)
    return rgb_batch.permute(0, 3, 1, 2)


def tensor_to_image(tensor):
    return transforms.ToPILImage()(tensor)


def normalize_tensor_image(rgb_tensor):
    rgb_min = rgb_tensor.min()
    rgb_max = rgb_tensor.max()
    rgb_tensor = (rgb_tensor - rgb_min) / (rgb_max - rgb_min)
    return rgb_tensor


def plot_images(low_res, inference, truths, title):
    low_res = low_res.cpu()
    inference = inference.cpu()
    truths = truths.cpu()

    if COLOR_SPACE == 'yuv':
        low_res = yuv_to_rgb_batch(low_res)
        inference = yuv_to_rgb_batch(inference)
        truths = yuv_to_rgb_batch(truths)

    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    for i in range(3):
        axs[i, 0].set_title('Low Res')
        axs[i, 0].imshow(tensor_to_image(normalize_tensor_image(low_res[i])))
        axs[i, 0].axis('off')

        axs[i, 1].set_title('Upscaled')
        axs[i, 1].imshow(tensor_to_image(inference[i]))
        axs[i, 1].axis('off')

        axs[i, 2].set_title('Normalized')
        axs[i, 2].imshow(tensor_to_image(normalize_tensor_image(inference[i])))
        axs[i, 2].axis('off')

        axs[i, 3].set_title('Truth')
        axs[i, 3].imshow(tensor_to_image(normalize_tensor_image(truths[i])))
        axs[i, 3].axis('off')
    plt.tight_layout()
    plt.savefig(title)
    plt.close()


def model_inference_step(model, dataloader, device, criterion, optimizer=None):
    """Run one epoch of training or evaluation"""
    running_loss = 0.0

    for batch in dataloader:
        low_res, hi_res_truth = batch
        low_res = low_res.to(device)
        hi_res_truth = hi_res_truth.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        inference = model(low_res)
        loss = criterion(inference, hi_res_truth)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader.dataset)


def train_model(model, train_dataloader, test_dataloader, optimizer,
                criterion, device, num_epochs=NUM_EPOCHS):
    """Main training loop"""

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = model_inference_step(
            model, train_dataloader, device, criterion, optimizer
        )

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_loss = model_inference_step(
                model, test_dataloader, device, criterion
            )

        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/test", test_loss, epoch + 1)

        print(f'Epoch {epoch + 1:>{3}} | Train loss: {train_loss:.8f} | '
              f'Test Loss: {test_loss:.8f}', flush=True)

        # Generate and save sample images
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                low_res, hi_res_truth = next(iter(test_dataloader))
                low_res = low_res.to(device)
                hi_res_truth = hi_res_truth.to(device)
                inference = model(low_res)
                plot_images(low_res, inference, hi_res_truth,
                            f"epoch_results/epoch{epoch + 1}.png")

    writer.flush()
    return model


def sec_to_human(seconds):
    """Convert seconds to human-readable format"""
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hours, minutes, seconds)


if __name__ == '__main__':
    start_time = time.time()
    print(f"Starting training at {time.ctime()}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} [torch {torch.__version__}]')
    print(f'Python version: {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')

    # Create model and move to device
    model = ARSRNetwork(
        scale_factor=2,
        in_channels=3  # Specify 3 channels for RGB input
    ).to(device)

    # Setup optimizer and loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    # Load dataset
    seed = 42
    torch.manual_seed(seed)

    from low_hi_res_dataset import SR_tensor_dataset

    full_dataset = SR_tensor_dataset(
        high_res_tensors_path='../data/data/high_res_tensors_10k.pt',
        low_res_tensors_path='../data/data/low_res_tensors_10k.pt'
    )

    # Split dataset
    train_size = int(0.85 * len(full_dataset))
    valid_size = int(0.10 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f'Total samples: {len(full_dataset)}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(valid_dataset)}')
    print(f'Test samples: {len(test_dataset)}')

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Train model
    model = train_model(
        model, train_dataloader, valid_dataloader,
        optimizer, criterion, device
    )

    # Save final model
    torch.save(model.state_dict(), './saved_weights/arsr_final.pth')

    # Print training time
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training completed in {sec_to_human(duration)}")