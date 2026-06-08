import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# ─── Configuration & Vocab (Must match solution.ipynb) ────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_HEIGHT = 32
IMG_WIDTH = 128

CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
BLANK_IDX = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 1-indexed
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # 40 classes (including blank)

# ─── Model Architecture (Exact structure from training) ────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pad=1, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            ConvBNReLU(1, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            ConvBNReLU(64, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Block 3
            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Block 4
            ConvBNReLU(256, 512),
            ConvBNReLU(512, 512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # Block 5
            ConvBNReLU(512, 512, kernel=2, pad=0),
        )
    def forward(self, x):
        return self.features(x)

class CRNN(nn.Module):
    def __init__(self, num_classes, rnn_hidden=256, rnn_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = CNNBackbone()
        
        self.input_proj = nn.Sequential(
            nn.Linear(512, rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.rnn = nn.LSTM(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=False,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        
    def forward(self, x):
        cnn_out = self.cnn(x)
        B, C, H, W = cnn_out.shape
        seq = cnn_out.squeeze(2)        # [B, 512, W']
        seq = seq.permute(2, 0, 1)      # [W', B, 512]
        seq = self.input_proj(seq)      # [W', B, rnn_hidden]
        rnn_out, _ = self.rnn(seq)      # [W', B, 2*rnn_hidden]
        logits = self.fc(rnn_out)       # [W', B, num_classes]
        return F.log_softmax(logits, dim=2)

# ─── Inference Utilities ───────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs, blank_idx=BLANK_IDX):
    """Decode predictions using best path decoding (greedy collapse)."""
    pred_indices = log_probs.argmax(dim=2)  # [T, B]
    pred_indices = pred_indices.permute(1, 0)  # [B, T]
    
    decoded_batch = []
    for preds in pred_indices:
        chars = []
        prev = None
        for idx in preds.tolist():
            if idx != blank_idx and idx != prev:
                chars.append(IDX2CHAR.get(idx, ''))
            prev = idx
        decoded_batch.append(''.join(chars))
    return decoded_batch

def predict_single_image(image_path, model_path='best_crnn_model.pth'):
    # Check if weights file exists
    if not os.path.exists(model_path):
        print(f"Error: Weights file '{model_path}' not found in the current directory.")
        print("Please make sure 'best_crnn_model.pth' is copied to this folder.")
        sys.exit(1)
        
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
        
    # Instantiate model
    model = CRNN(num_classes=NUM_CLASSES, rnn_hidden=256, rnn_layers=2, dropout=0.3).to(DEVICE)
    
    # Load trained weights
    print(f"Loading weights from: {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Preprocess image
    print("Preprocessing image...")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.open(image_path).convert('L')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 1, 32, 128]
    
    # Predict
    print("Running inference...")
    with torch.no_grad():
        log_probs = model(input_tensor)
        
    # Decode text
    predictions = ctc_greedy_decode(log_probs)
    return predictions[0]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py sample.png")
        sys.exit(1)
        
    img_path = sys.argv[1]
    result = predict_single_image(img_path)
    print("\n" + "="*30)
    print(f"PREDICTED TEXT: {result}")
    print("="*30)
