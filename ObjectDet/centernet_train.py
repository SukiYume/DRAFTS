import os, re, cv2, json, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

plt.style.use('default')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from timm.scheduler import CosineLRScheduler
from centernet_data import BurstDataset
from centernet_model import centernet, centerloss
from centernet_utils import get_res, denormalize
from sklearn.model_selection import train_test_split


def load_train_cat(train_path):

    data              = pd.read_csv(train_path)
    data              = data.loc[data.save_name!='-1'].reset_index(drop=True)
    data['image_id']  = data.save_name.str.replace('.npy', '', regex=False) + '__' + data.freq_slice.astype(str) + '.npy'

    ## 标记时标记的是时间和色散，中心点C与左下点L，L的时间小于C，L的色散大于C
    data[['x', 'y']]  = data.loc[:, ['time_center', 'dm_center']]
    ## 计算宽度，乘2
    data['w']         = np.min([data.time_center - data.time_left, 8192 - data.time_center - 1], axis=0) * 2
    data['h']         = np.min([data.dm_left - data.dm_center, data.dm_center - 1.05], axis=0) * 2
    data              = data.loc[:, ['image_id', 'x', 'y', 'w', 'h']]
    ## 将原本没有标记的数据设为-1
    data.loc[(data.x<=0)&(data.y<=0), ['x', 'y', 'w', 'h']] = [-1, -1, -1, -1]
    # data = data.loc[data.x!=-1].reset_index(drop=True)

    image_ids         = data['image_id'].unique()
    train_id, test_id = train_test_split(image_ids, test_size=0.2)

    return data, train_id, test_id


if __name__ == '__main__':

    input_size   = 512
    model_scale  = 4
    batch_size   = 4
    epochs       = 100
    patience     = 100
    data_path    = './Data/'

    # backbone     = 'resnet18'
    backbone     = sys.argv[1]
    log_dir      = './logs_{}/'.format(backbone)


    train_df, train_id, val_id = load_train_cat('data_label.txt')
    train_data   = BurstDataset(train_id, train_df, val=False)
    val_data     = BurstDataset(val_id, train_df, val=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # 展示
    # img, hm, regr = train_data[0]
    # plt.imshow(img.transpose([1, 2, 0]))
    # plt.show()
    # img.std()
    # plt.imshow(hm)
    # plt.show()

    if True:
        if os.path.exists(log_dir + 'best_model.pth'):
            model = centernet(model_name=backbone).to(device)
            model.load_state_dict(torch.load(log_dir + 'best_model.pth'))
            model.eval()
            print('load model done')
        else:
            model = centernet(model_name=backbone).to(device)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        optimizer = torch.optim.Adam(model.parameters())
        scheduler = CosineLRScheduler(optimizer, t_initial=epochs, lr_min=1e-5, warmup_t=5, warmup_lr_init=1e-5, warmup_prefix=True)

        logs, logs_eval = [], []
        loss_min = pow(10, 10)

        for epoch in range(epochs):

            print('Epoch {}/{} with LR {:.5f}'.format(epoch + 1, epochs, optimizer.param_groups[0]['lr']))
            running_loss, running_cls, running_reg = 0, 0, 0
            running_loss_val, running_cls_val, running_reg_val = 0, 0, 0
            tqdm_train = tqdm(train_loader, dynamic_ncols=True, ascii=True)

            model.train()
            for idx_train, (img, targ) in enumerate(tqdm_train):
                img, targ_gt = img.to(device), targ.to(device)
                optimizer.zero_grad()
                hm, wh, offset = model(img)
                pred = torch.cat((hm, wh, offset), 1)
                loss, cls_loss, reg_loss = centerloss(pred, targ_gt)

                running_loss += loss
                running_cls  += cls_loss
                running_reg  += reg_loss
                loss.backward()
                optimizer.step()
                tqdm_train.set_description(f'Train [l={running_loss / (idx_train + 1):.3f}][c={running_cls / (idx_train + 1):.4f}][r={running_reg / (idx_train +    1):.4f}]')

            tqdm_val = tqdm(val_loader, dynamic_ncols=True, ascii=True)
            model.eval()
            with torch.no_grad():
                for idx_val, (img, targ) in enumerate(tqdm_val):
                    img, targ_gt = img.to(device), targ.to(device)
                    hm, wh, offset = model(img)
                    pred = torch.cat((hm, wh, offset), 1)
                    loss, cls_loss, reg_loss = centerloss(pred, targ_gt)
                    running_loss_val += loss
                    running_cls_val  += cls_loss
                    running_reg_val  += reg_loss
                    tqdm_val.set_description(f'Val [l={running_loss_val / (idx_val + 1):.3f}][c={running_cls_val / (idx_val + 1):.4f}][r={running_reg_val / (idx_val + 1):.4f}]')

            train_loss = running_loss / len(train_loader)
            val_loss = running_loss_val / len(val_loader)
            print('Epoch {}/{}, TrainLoss {:.5f}, ValLoss {:.5f}'.format(epoch + 1, epochs, train_loss, val_loss))

            if val_loss <= loss_min:
                loss_min = val_loss
                counter  = 0
                print('loss is min')
                print('save model...')
                torch.save(model.state_dict(), log_dir + 'best_model.pth')
                torch.save(model.state_dict(), log_dir + 'Epoch{:0>2d}_TLoss{:.3f}_VLoss{:.3f}.pth'.format(epoch + 1, train_loss, val_loss))
                print("done\n")
            else:
                counter += 1
                print('loss is not min\n')
                if counter >= patience:
                    break

            # save logs
            log_epoch = {
                'epoch': epoch + 1,
                'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                'loss_train': running_loss.item() / len(train_loader),
                'cls_train': running_cls.item() / len(train_loader),
                'reg_train': running_reg.item() / len(train_loader),
                'loss_val': running_loss_val.item() / len(val_loader),
                'cls_val': running_cls_val.item() / len(val_loader),
                'reg_val': running_reg_val.item() / len(val_loader),
            }
            logs.append(log_epoch)
            with open(log_dir + 'logs_{}.json'.format(backbone), 'w') as f:
                json.dump(logs, f)
            scheduler.step(epoch + 1)


    if False:

        model = centernet(model_name=backbone).to(device)
        model.load_state_dict(torch.load(log_dir + 'best_model.pth'))
        model.eval()

        # 展示
        for id in range(15, 30):
            img, targ = val_data[id]
            hm_gt, wh_gt, reg_gt, reg_mask = targ[0][np.newaxis, np.newaxis], targ[1:3][np.newaxis], targ[3:5][np.  newaxis], targ[5][np.newaxis, np.newaxis]

            model.eval()
            with torch.no_grad():
                hm, wh, offset = model(torch.from_numpy(img).to(device).float().unsqueeze(0))
            top_conf, top_boxes = get_res(hm, wh, offset, 0.2)

            plt.imshow(hm.cpu().numpy().squeeze(0).squeeze(0) > 0.2)
            plt.show()

            img  = np.array(img).transpose(1, 2, 0)
            img  = denormalize(img)
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # print ground truth boxes
            _, top_boxes_gt = get_res(torch.tensor(hm_gt).to(device), torch.tensor(wh_gt).to(device), torch.tensor  (reg_gt). to(device), 0.2)
            if top_boxes_gt is not None:
                for box in top_boxes_gt:
                    left_x, left_y, right_x, right_y = box.astype(np.int64)
                    cv2.rectangle(img, (left_x, left_y), (right_x, right_y), (0, 0, 220), 1)

            # print predicted boxes
            if top_boxes is not None:
                for box in top_boxes:
                    left_x, left_y, right_x, right_y = box.astype(np.int64)

                    center_time, center_dm = (left_x + right_x) / 2, (left_y + right_y) / 2
                    left_time, left_dm = left_x, center_dm + (right_y - left_y) / 2
                    center_time, center_dm, left_time, left_dm = center_time * 16, center_dm * 2, left_time * 16, left_dm * 2

                    print(center_time, center_dm, left_time, left_dm)
                    cv2.rectangle(img, (left_x, left_y), (right_x, right_y), (220, 0, 0), 1)

            plt.imshow(img)
            plt.show()