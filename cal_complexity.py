import torch
import torch.nn as nn
from models.builder import EncoderDecoder as segmodel
from configs.config_MFNet import config
from engine.logger import get_logger

try:
    from thop import profile
except ImportError:
    print("Please install thop first: pip install thop")
    exit()

def main():
    print("=======================================")
    print("Calculating Model Complexity for Sigma-Mask")
    print("=======================================")

    # 确保使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 实例化我们修改后的模型
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)
    
    # 将模型放入 GPU 并设为评估模式
    model = model.to(device)
    model.eval()

    # 2. 计算总参数量 (Parameters)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # 3. 计算 FLOPs (将输入的假图片也放入 GPU)
    dummy_rgb = torch.randn(1, 3, 480, 640).to(device)
    dummy_x = torch.randn(1, 3, 480, 640).to(device)

    print("Calculating FLOPs... (This may take a few seconds)")
    
    # 巧妙处理 kwargs 输入
    flops, params = profile(model, inputs=(dummy_rgb, dummy_x), verbose=False)
    
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print("=======================================")

if __name__ == '__main__':
    main()
