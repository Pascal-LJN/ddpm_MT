import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_cifar10(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化，中心化到[-1,1]
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


from torchvision.datasets import Places365
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def load_places365(batch_size=32):
    transform = Compose([
        Resize((256, 256)),  # 将图像调整为统一大小
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差进行标准化
    ])
    train_dataset = Places365(root='./data', split='train-standard', small=False, download=True, transform=transform)
    test_dataset = Places365(root='./data', split='val', small=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


from torchvision import datasets
from torch.utils.data import DataLoader


def load_dataset(dataset_name, batch_size=32, train=True):
    """ 根据给定的数据集名加载并应用预处理 """
    transform = get_transforms(dataset_name)
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'places365':
        split = 'train-standard' if train else 'val'
        dataset = datasets.Places365(root='./data', split=split, small=False, download=True, transform=transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return data_loader


import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5  # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    batch_size = 32
    cifar10_train, cifar10_test = load_cifar10(batch_size)
    places_train, places_test = load_places365(batch_size)

    # 显示CIFAR-10中的一些图像
    dataiter = iter(cifar10_train)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))

    # 显示Places365中的一些图像
    dataiter = iter(places_train)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    """下采样层，使用卷积实现"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return F.relu(self.conv(x))


class Upsample(nn.Module):
    """上采样层，使用转置卷积实现"""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(Upsample, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return F.relu(self.conv_transpose(x))


class ResidualBlock(nn.Module):
    """残差块，用于增强模型的学习能力"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)


class DDPM(nn.Module):
    def __init__(self):
        super(DDPM, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            Downsample(3, 64),  # 输入通道, 输出通道
            ResidualBlock(64),
            Downsample(64, 128),
            ResidualBlock(128),
            Downsample(128, 256),
            ResidualBlock(256)
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            Upsample(256, 128),
            ResidualBlock(128),
            Upsample(128, 64),
            ResidualBlock(64),
            Upsample(64, 3),  # 输出通道为3，与输入图像相匹配
            nn.Tanh()  # 输出范围[-1, 1]以匹配图像的标准化
        )

    def forward(self, x, t, noise=None):
        # t: 时间步, noise: 噪声
        if noise is None:
            noise = torch.randn_like(x)
        x = x + torch.sqrt(1 - t) * noise  # 注入噪声
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def sqrt(x):
    return torch.sqrt(x)


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes):
        super(ConditionalDDPM, self).__init__()
        # 基本的DDPM组件
        self.encoder = nn.Sequential(
            Downsample(3, 64),  # 输入通道, 输出通道
            ResidualBlock(64),
            Downsample(64, 128),
            ResidualBlock(128),
            Downsample(128, 256),
            ResidualBlock(256)
        )
        self.decoder = nn.Sequential(
            Upsample(256, 128),
            ResidualBlock(128),
            Upsample(128, 64),
            ResidualBlock(64),
            Upsample(64, 3),  # 输出通道为3，与输入图像相匹配
            nn.Tanh()  # 输出范围[-1, 1]以匹配图像的标准化
        )
        # 类别条件
        self.class_embedding = nn.Embedding(num_classes, 256)  # 假设256是嵌入维度

    def forward(self, x, t, labels, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        class_cond = self.class_embedding(labels)  # 类条件嵌入
        class_cond = class_cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])  # 扩展到与输入相同的维度
        x = x + torch.sqrt(1 - t) * noise + class_cond  # 将类条件添加到输入和噪声中
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def load_dataset(batch_size):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图像大小以匹配模型输入
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def train_ddpm(model, train_loader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            batch_size = data.size(0)
            # 生成随机时间步和噪声
            timesteps = torch.rand(batch_size, 1, 1, 1).to(device)
            noise = torch.randn_like(data).to(device)

            optimizer.zero_grad()
            # 模型输出
            output = model(data, timesteps, noise)
            # 计算损失：我们使用均方误差损失
            loss = F.mse_loss(output, data)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss {loss.item():.6f}')

    print('Training Complete')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDPM().to(device)
    train_loader = load_dataset(batch_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_ddpm(model, train_loader, optimizer, device)


if __name__ == '__main__':
    main()

import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips  # LPIPS库


# 初始化LPIPS


def calculate_psnr(target, output):
    target_np = target.mul(0.5).add(0.5).clamp(0, 1).numpy()  # 反归一化
    output_np = output.mul(0.5).add(0.5).clamp(0, 1).numpy()
    return psnr(target_np, output_np, data_range=1)


def calculate_ssim(target, output):
    target_np = target.mul(0.5).add(0.5).clamp(0, 1).numpy()
    output_np = output.mul(0.5).add(0.5).clamp(0, 1).numpy()
    ssim_value = ssim(target_np, output_np, multichannel=True, data_range=1)
    return ssim_value


def test_model(model, test_loader, device):
    model.eval()
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            t = torch.full((images.size(0), 1, 1, 1), 1.0, device=device)  # 最后的时间步
            outputs = model(images, t)

            # 计算评估指标
            batch_psnr = calculate_psnr(images, outputs)
            batch_ssim = calculate_ssim(images, outputs)
            batch_lpips = lpips_vgg(images, outputs).mean().item()

            total_psnr += batch_psnr
            total_ssim += batch_ssim
            total_lpips += batch_lpips

            if i % 10 == 0:  # 保存部分输出图像
                save_image(outputs, f'output_{i}.png')

        # 计算平均指标
        avg_psnr = total_psnr / len(test_loader)
        avg_ssim = total_ssim / len(test_loader)
        avg_lpips = total_lpips / len(test_loader)
        print(f'Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}, Average LPIPS: {avg_lpips:.4f}')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDPM().to(device)
    test_loader = load_dataset(batch_size=32, train=False)  # 假设有一个加载测试集的函数
    test_model(model, test_loader, device)


if __name__ == '__main__':
    main()

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class DDPM(torch.nn.Module):
    # 请根据您的模型实际结构填写
    def __init__(self):
        super(DDPM, self).__init__()
        # 初始化模型的各个层

    def forward(self, x):
        # 定义模型的前向传播
        return x  # 返回处理后的图像


def load_model(model_path, device):
    model = DDPM()  # 实例化模型
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def repair_images(model, loader, device):
    repaired_images = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            repaired = model(images)
            repaired_images.extend(repaired.cpu())  # 将修复的图像移动到CPU
    return repaired_images


def save_repaired_images(images, output_folder):
    for idx, image in enumerate(images):
        save_image(image, f'{output_folder}/repaired_image_{idx}.png')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'path_to_your_model.pth'  # 模型文件路径
    model = load_model(model_path, device)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(root='path_to_test_images', transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    repaired_images = repair_images(model, loader, device)
    save_repaired_images(repaired_images, 'output_directory')  # 指定输出文件夹


if __name__ == '__main__':
    main()

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np


def load_data(data_path, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 假设所有图像调整为32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def visualize_and_save_images(original_images, masked_images, repaired_images, filename):
    # 将图像组合成一行
    image_tensor = torch.cat((original_images, masked_images, repaired_images), dim=0)
    grid_image = make_grid(image_tensor, nrow=3)  # 每行显示3张图片：原始，屏蔽，修复
    np_image = grid_image.numpy().transpose((1, 2, 0))
    np_image = np_image * 0.5 + 0.5  # 反标准化
    plt.figure(figsize=(10, 5))
    plt.imshow(np_image)
    plt.axis('off')
    plt.show()
    save_image(grid_image, f'{filename}.png')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    data_loader = load_data('path_to_images', batch_size=1)
    # 假设model已经加载，这里只是演示
    model = None

    for batch in data_loader:
        original_images, _ = batch
        original_images = original_images.to(device)

        # 假设我们可以从模型或某个过程中获得屏蔽图像和修复图像
        masked_images = torch.randn_like(original_images)  # 这里用随机数据代替真正的屏蔽图像
        repaired_images = torch.randn_like(original_images)  # 这里用随机数据代替真正的修复图像

        # 可视化并保存图像
        visualize_and_save_images(original_images.cpu(), masked_images.cpu(), repaired_images.cpu(), 'comparison')


if __name__ == '__main__':
    main()

import pandas as pd
import matplotlib.pyplot as plt


def load_data(data_path):
    # 从CSV或其他文件格式加载数据
    return pd.read_csv(data_path)


def analyze_data(data):
    # 计算基本统计数据
    summary_stats = data.describe()
    print("Summary Statistics:\n", summary_stats)

    # 如果有评分数据，计算平均评分
    if 'rating' in data.columns:
        average_rating = data['rating'].mean()
        print("Average Rating:", average_rating)

    return summary_stats, average_rating


def visualize_data(data):
    # 绘制评分的直方图
    if 'rating' in data.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(data['rating'], bins=20, color='blue', alpha=0.7)
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    # 可视化反馈数量随时间的变化（如果数据中包含时间戳）
    if 'date' in data.columns:
        plt.figure(figsize=(10, 6))
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data['rating'].resample('M').mean().plot()
        plt.title('Average Rating Over Time')
        plt.ylabel('Average Rating')
        plt.grid(True)
        plt.show()


def main(data_path):
    data = load_data(data_path)
    summary_stats, average_rating = analyze_data(data)
    visualize_data(data)


if __name__ == '__main__':
    data_path = 'path_to_your_data_file.csv'  # 数据文件路径
    main(data_path)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes):
        super(ConditionalDDPM, self).__init__()
        # 基本的DDPM组件
        self.encoder = nn.Sequential(
            Downsample(3, 64),  # 输入通道, 输出通道
            ResidualBlock(64),
            Downsample(64, 128),
            ResidualBlock(128),
            Downsample(128, 256),
            ResidualBlock(256)
        )
        self.decoder = nn.Sequential(
            Upsample(256, 128),
            ResidualBlock(128),
            Upsample(128, 64),
            ResidualBlock(64),
            Upsample(64, 3),  # 输出通道为3，与输入图像相匹配
            nn.Tanh()  # 输出范围[-1, 1]以匹配图像的标准化
        )
        # 类别条件
        self.class_embedding = nn.Embedding(num_classes, 256)  # 假设256是嵌入维度

    def forward(self, x, t, labels, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        class_cond = self.class_embedding(labels)  # 类条件嵌入
        class_cond = class_cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])  # 扩展到与输入相同的维度
        x = x + torch.sqrt(1 - t) * noise + class_cond  # 将类条件添加到输入和噪声中
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_dynamic_timestep(epoch, max_epochs, initial_t=1.0, final_t=0.01):
    # 线性衰减
    t = initial_t - (initial_t - final_t) * (epoch / max_epochs)
    return max(t, final_t)  # 确保t不会低于final_t


import torch
import torch.nn as nn
import torch.optim as optim


def train_ddpm(model, train_loader, optimizer, device, epochs=10, initial_t=1.0, final_t=0.01):
    model.train()
    for epoch in range(epochs):
        current_t = get_dynamic_timestep(epoch, epochs, initial_t, final_t)
        for data, _ in train_loader:
            data = data.to(device)
            noise = torch.randn_like(data).to(device)

            optimizer.zero_grad()
            output = model(data, current_t, noise)
            loss = nn.MSELoss()(output, data)  # 假设使用MSE损失函数
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, Current t: {current_t:.4f}')


import torch
import torch.nn as nn
import torch.optim as optim


def train_ddpm(model, train_loader, optimizer, device, epochs=10, initial_t=1.0, final_t=0.01):
    model.train()
    for epoch in range(epochs):
        current_t = get_dynamic_timestep(epoch, epochs, initial_t, final_t)
        for data, _ in train_loader:
            data = data.to(device)
            noise = torch.randn_like(data).to(device)

            optimizer.zero_grad()
            output = model(data, current_t, noise)
            loss = nn.MSELoss()(output, data)  # 假设使用MSE损失函数
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, Current t: {current_t:.4f}')


class DDPM(nn.Module):
    # 这里应该是您的DDPM模型的代码，包含encoder和decoder
    pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDPM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = None  # 假设已经定义了数据加载器

    train_ddpm(model, train_loader, optimizer, device, epochs=50)


if __name__ == '__main__':
    main()

import torch
import torch.nn.functional as F


def advanced_loss(output, target, noise, noise_weight=0.1):
    """
    计算高级损失，包括重建误差和噪声复原误差。
    :param output: 模型的输出图像
    :param target: 原始图像（未加噪声）
    :param noise: 应用于原始图像的噪声
    :param noise_weight: 噪声重建部分的权重
    :return: 总损失值
    """
    # 重建损失，可以使用MSE或者其他更适合图像的损失
    reconstruction_loss = F.mse_loss(output, target)

    # 噪声复原损失
    noise_estimation = output - target  # 噪声估计
    noise_loss = F.mse_loss(noise_estimation, noise)

    # 总损失
    total_loss = reconstruction_loss + noise_weight * noise_loss
    return total_loss


def train_ddpm(model, train_loader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for data in train_loader:
            original_data = data.clone().to(device)  # 原始数据
            noise = torch.randn_like(data).to(device)  # 生成噪声
            noisy_data = original_data + noise  # 创建含噪声的数据

            data, noisy_data = data.to(device), noisy_data.to(device)
            optimizer.zero_grad()
            output = model(noisy_data)  # 模型预测
            loss = advanced_loss(output, original_data, noise)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}')


import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 假设已经定义了DDPM, train_dataset, val_dataset
from your_model_file import DDPM, train_dataset, val_dataset


def objective(trial):
    # 提议超参数
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型和优化器
    model = DDPM().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 简化的训练和验证循环
    for epoch in range(10):  # 使用更少的epoch进行快速演示
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, data)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                output = model(data)
                val_loss += nn.MSELoss()(output, data).item()
        val_loss /= len(val_loader)

    return val_loss


# 创建一个Optuna研究对象并执行优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# 最佳超参数
print("Best hyperparameters: ", study.best_trial.params)
