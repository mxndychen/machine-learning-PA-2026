import torch
import torch.nn as nn
import torch.optim as optim

#file has provided dataloaders
from data import train_loader, val_loader

'''
RULES FROM KAGGLE:

-Net family only. Standard U-Net, U-Net with custom encoders/decoders, U-Net++, attention U-Net, residual U-Net, etc. are all allowed as long as the overall encoder–decoder + skip-connection architecture is preserved.
Encoder backbones from torchvision (ResNet, EfficientNet, VGG, etc.) initialized with ImageNet classification pretrained weights are allowed.
3. Forbidden
The following are not allowed under any circumstances:

❌ Segmentation foundation models (SAM, SAM2, Mask2Former, OneFormer, MaskFormer, SegFormer, etc.)
❌ Vision-language pretrained encoders (CLIP, DINO, DINOv2, BEiT, MAE, etc.)
❌ Any pretrained weights other than ImageNet classification weights
❌ External segmentation datasets (COCO, ADE20K, Pascal VOC, Cityscapes, LVIS, etc.)
❌ External pet/animal datasets beyond Oxford-IIIT Pet
❌ Test-time use of any pretrained segmentation model (e.g. running SAM on test images and using its output)

'''

'''
Acknowledgements/help:

Looked to github.com/milesial/pytorch-unet for help and also from VS Code Copilot assistant when coding.

When testing python3 main.py and UNET I made, I ran into issues with a typeError so I used CoPilot to help me troubleshoot --> said to change to segmentation in data.py

'''

'''
REMINDERS WHILE TESTING MAIN.PY ON MACBOOK:

- Use a smaller batch size if run into memory issues
- Keep an eye on the training/validation loss to avoid overfitting

reminders of changed:
epoch to be 1 instead of 10 CHANGE num_epochs BACK TO 10 WHEN DOING GPU / VESSL

'''

#add function so don't need to constantly keep doing reLu and conv 

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """UNet architecture for image segmentation."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3): #num_classes og: 1 but we have 3 classes??
        super().__init__()

        # define encoder (contracting path) = encoder needs to reduce image size but build depth of features
        #how to represent animal vs background

    #    self.down1 = nn.Sequential(
    #        nn.Conv2d(in_channels, 64, 
    #                  kernel_size = 3, padding = 1), 
    #                  nn.ReLU(), 
    #                  nn.Conv2d(64, 64, kernel_size = 3, padding = 1), 
    #                  nn.ReLU(inplace = True))
    #
    #     
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # define bottleneck (e.g., 512 -> 1024)
        self.bottleneck = DoubleConv(512, 1024)

        # define decoder (expanding path)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        # define final 1x1 conv (out_channels = num_classes) -> multiclass, goal has foreground, background, and boundary.
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1) #final conv should make features into class predict.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder forward, store feature maps for skip connections
        x1 = self.down1(x)
        p1 = self.pool1(x1)

        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        x3 = self.down3(p2)
        p3 = self.pool3(x3)

        x4 = self.down4(p3)
        p4 = self.pool4(x4)

        # bottleneck forward
        bottleneck = self.bottleneck(p4)

        # decoder forward, concat with skip connections
        up1 = self.up1(bottleneck)
        up1 = torch.cat([up1, x4], dim=1)  # skip connection
        conv1 = self.conv1(up1)

        up2 = self.up2(conv1)
        up2 = torch.cat([up2, x3], dim=1)  # skip connection
        conv2 = self.conv2(up2)

        up3 = self.up3(conv2)
        up3 = torch.cat([up3, x2], dim=1)  # skip connection
        conv3 = self.conv3(up3)

        up4 = self.up4(conv3)
        up4 = torch.cat([up4, x1], dim=1)  # skip connection
        conv4 = self.conv4(up4)

        # return segmentation logits via final 1x1 conv
        return self.final_conv(conv4)


class SegmentationLoss(nn.Module):
    """Loss function for segmentation (e.g., BCE / CrossEntropy / Dice / combined)."""

    def __init__(self):
        super().__init__()
        # define the loss to use
        #   - BCEWithLogitsLoss for binary segmentation
        #   - CrossEntropyLoss for multi-class segmentation
        #   - optionally combine with Dice loss

        self.loss = nn.CrossEntropyLoss() #multiclass lose

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # compute and return loss from logits and targets
        return self.loss(logits, targets)


class Trainer:
    """Training / validation loop wrapper."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, loader) -> float:
        self.model.train()
        # iterate over (images, masks) batches from loader
        #   1) move tensors to device
        #   2) optimizer.zero_grad()
        #   3) forward -> compute loss
        #   4) loss.backward() -> optimizer.step()
        #   5) accumulate and return average loss
        total_loss = 0
        #for images, masks in loader:
        for batch_idx, (images, masks) in enumerate(loader):
            #print(f"Batch {batch_idx + 1}/{len(loader)}") #testing on macbook cpu main.py, see training progress
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)


    @torch.no_grad()
    def validate(self, loader) -> float:
        self.model.eval()
        # run forward only and compute loss / metrics (IoU, Dice, etc.)
        total_loss = 0
        for images, masks in loader:
            images, masks = images.to(self.device), masks.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            total_loss += loss.item()
        return total_loss / len(loader)
    


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set hyperparameters (lr, num_epochs, num_classes, etc.)
    num_epochs = 10 

    #test main.py on macbook CPU REMEMBER TO COMMENT AND UNCOMMENT epochs = 10
    #num_epochs = 1


    learning_rate = 1e-4
    num_classes = 3

    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = SegmentationLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, criterion, optimizer, device)

    # for epoch in range(num_epochs):
    #     train_loss = trainer.train_one_epoch(train_loader)
    #     val_loss = trainer.validate(val_loader)
    #     print(f"[Epoch {epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    # # save best model checkpoint / visualize predictions / report metrics

    # torch.save(model.state_dict(), "unet_checkpoint.pth")
    # print("Model checkpoint saved as unet_checkpoint.pth")

    best_val_loss = float("inf")

    #troubleshooting ToT
    #for images, labels in train_loader:

    images, labels = next(iter(train_loader))
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)

    print("Label dtype:", labels.dtype)

        # if len(labels.shape) == 1:
        #     print("labels:", labels[:10])

    for epoch in range(num_epochs):
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        print(f"[Epoch {epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "unet_best.pth")
            torch.save(model.state_dict(), "/output/unet_best.pth")
            print("Model checkpoint saved as unet_best.pth")

if __name__ == "__main__":
    main()
