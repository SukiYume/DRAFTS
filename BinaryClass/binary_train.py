import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')

import torch, torchvision
from timm.scheduler import CosineLRScheduler
from binary_model import SPPResNet, BinaryNet, random_resize
from binary_data import BurstDataset, get_train_val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed   = 3407
# torch.manual_seed(seed)


def plot_scheduler(epoches):

    model     = torchvision.models.resnet18()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = CosineLRScheduler(optimizer, t_initial=num_epoches-5, cycle_decay=0.5, cycle_limit=1, lr_min=1e-6, warmup_t=4, warmup_lr_init=1e-5)

    a = []
    for epoch in range(epoches):
        scheduler.step(epoch + 1)
        a.append(optimizer.param_groups[0]['lr'])

    plt.plot(a)
    plt.savefig('logs/scheduler.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    finetune      = False
    num_epoches   = 50
    num_classes   = 2
    base_model    = sys.argv[1] # 'resnet18'
    model_type    = sys.argv[2] # 'BinaryNet'
    fix_size      = eval(sys.argv[3]) if len(sys.argv) > 3 else True

    ## 根据模型种类构建不同模型
    if model_type == 'SPPResNet':
        save_path = './logs_spp_{}'.format(base_model)
        model     = SPPResNet(base_model, num_classes=num_classes).to(device)
    elif model_type == 'BinaryNet':
        save_path = './logs_res_{}'.format(base_model)
        model     = BinaryNet(base_model, num_classes=num_classes).to(device)

    if fix_size:
        save_path += '_fix/'
        batch_size = 16
    else:
        save_path += '_ran/'
        batch_size = 16

    if not os.path.exists(save_path): os.makedirs(save_path)

    ## 加载预训练模型
    if finetune:
        model.load_state_dict(torch.load('./logs/best_model_fix_resnet18.pth'))
        model.eval()

    ## 加载数据集
    data_path     = './Data/'
    train_data, val_data = get_train_val(data_path)
    train_data    = BurstDataset(train_data, val=False)
    val_data      = BurstDataset(val_data, val=True)
    train_loader  = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader    = torch.utils.data.DataLoader(val_data,   batch_size=batch_size, shuffle=True)

    ## 设置损失函数和优化器
    ## 当num_classes=1时，使用 BCEWithLogitsLoss + sigmoid，label需要y = torch.tensor([y], dtype=torch.float32)，correct = torch.eq(torch.gt(outputs, 0.5), labels)
    ## 当num_classes=2时，使用 CrossEntropyLoss + softmax，label需要y = torch.tensor(y, dtype=torch.long)，correct = torch.eq(torch.max(torch.softmax(outputs, dim=1), dim=1)[1], labels)
    criterion     = torch.nn.CrossEntropyLoss()
    optimizer     = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=1e-5)
    scheduler     = CosineLRScheduler(optimizer, t_initial=num_epoches-5, cycle_decay=0.5, cycle_limit=1, lr_min=1e-6, warmup_t=4, warmup_lr_init=1e-5)

    ## 训练
    log_data, loss_min = [], pow(10, 10)
    for epoch in range(num_epoches):
        epoch_lr = optimizer.param_groups[0]['lr']
        print('Epoch {}/{} with LR {:.6f}'.format(epoch + 1, num_epoches, epoch_lr))

        ## 训练
        with tqdm(total=len(train_loader), dynamic_ncols=True, ascii=True, desc='Epoch {}/{}'.format(epoch + 1, num_epoches), bar_format='{desc:16}{percentage:3.0f}%|{bar:20}{r_bar}') as pbar:
            model.train()
            train_loss, train_acc = [], []
            for i, (inputs, labels) in enumerate(train_loader, 0):
                ## 是否固定输入大小为512x512训练
                if not fix_size:
                    inputs     = random_resize(inputs)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs  = model(inputs)
                loss     = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # predicted  = torch.eq(torch.gt(outputs.sigmoid(), 0.5), labels)
                predicted = torch.eq(torch.max(outputs, dim=1)[1], labels)
                acc = torch.sum(predicted).item() / predicted.shape[0]
                train_loss.append(loss.data.item())
                train_acc.append(acc)

                pbar.set_postfix({'loss': loss.item(), 'acc': acc})
                pbar.update()

        ## 验证
        with tqdm(total=len(val_loader), dynamic_ncols=True, ascii=True, desc='Val Epoch {}/{}'.format(epoch + 1, num_epoches), bar_format='{desc:16}{percentage:3.0f}%|{bar:20}{r_bar}') as pbar:
            model.eval()

            val_loss, val_acc = [], []
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(val_loader, 0):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs   = model(inputs)
                    loss      = criterion(outputs, labels)

                    # predicted  = torch.eq(torch.gt(outputs.sigmoid(), 0.5), labels)
                    predicted = torch.eq(torch.max(outputs, dim=1)[1], labels)
                    acc = torch.sum(predicted).item() / predicted.shape[0]
                    val_loss.append(loss.item())
                    val_acc.append(acc)

                    pbar.set_postfix({'loss': loss.item(), 'acc': acc})
                    pbar.update()

        train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc = np.mean(train_loss), 100 * np.mean(train_acc), np.mean(val_loss), 100 * np.mean(val_acc)
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%\n'.format(epoch+1, num_epoches, train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc))

        ## 检查点
        if val_epoch_loss <= loss_min:
            loss_min = val_epoch_loss
            counter  = 0
            print('loss is min')
            print('save model...')
            torch.save(model.state_dict(), save_path + 'best_model.pth')
            torch.save(
                model.state_dict(),
                save_path + 'Epoch{:0>3d}_Tloss{:.3f}_Tacc{:.3f}_Vloss{:.3f}_Vacc{:.3f}.pth'.format(epoch + 1, train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc)
            )
            print("done\n")
        else:
            print('loss is not min\n')

        ## 调整学习率
        scheduler.step(epoch + 1)
        log_data.append([epoch_lr, train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc])

    np.save('{}/logs.npy'.format(save_path), log_data)
