# encoding: utf8
# @Author: qiaoyongchao
# @Email: xiaoxia0722qyc@163.com
import json
import shutil

from torch.utils.tensorboard import SummaryWriter
from src.train import *
from src.dataset import *

import os
import time
import datetime
import argparse
from loguru import logger

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision


parser = argparse.ArgumentParser(description="DFS-GAN")

parser.add_argument("--config", help='配置文件(可指定参数优先于配置文件参数)', default='./configs/bird.yml')
parser.add_argument("--mode", help='代码运行模式(train或者test)', choices=['test', 'train'])
parser.add_argument("--epochs", help="训练时的轮数", type=int)
parser.add_argument("--batch_size", help="训练时每步的批次", type=int)
parser.add_argument("--test_interval", help="测试频率", type=int)
parser.add_argument("--local_rank", default=-1, type=int, help="pytorch多gpu训练时的节点")
parser.add_argument("--gpus", help='需要使用的gpu/cpu')
parser.add_argument("--output", help='保存文件目录')

args = parser.parse_args()

configs = get_config(args)

many_gpus = False
local_rank = 0
gpus = False
if configs['gpus'] in ('cpu', '-1') or not torch.cuda.is_available():
    device = torch.device('cpu')
elif ',' not in configs['gpus']:
    device = torch.device(int(configs['gpus']))
    gpus = True
else:
    local_rank = int(configs['local_rank'])
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    many_gpus = True
    gpus = True

if not os.path.exists(configs['output']):
    logger.warning("输出目录{}不存在, 开始创建...".format(configs['output']))
    os.makedirs(configs['output'])

logger.add(os.path.join(configs['output'], 'log_{}_{}.log'.format(configs['mode'], datetime.datetime.now().strftime("%Y%m%d%H%M%S"))))

logger.info("配置文件名为:{}, 路径为:{}, 配置如下:\n{}".format(configs['config_name'], args.config, configs))

configs['many_gpus'] = many_gpus

logger.info("gpu是否可用:{}, gpu是否使用:{}, 多gpu是否使用:{}, gpu使用信息:{}".format(torch.cuda.is_available(), gpus, many_gpus, device))

logger.info("初始化数据集....")

train_dataset, val_dataset = prepare_datasets(configs)

logger.info("训练数据大小:{}, 测试数据大小:{}".format(len(train_dataset), len(val_dataset)))

logger.info("初始化dataloader...")
train_sampler = None
if many_gpus:
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], drop_last=True, num_workers=configs['num_workers'], sampler=train_sampler)
else:
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], drop_last=True, num_workers=configs['num_workers'], shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=configs['batch_size'], num_workers=configs['num_workers'])

trainer = Trainer(configs, device, logger)

img_save_dir = None
if configs['save_image']:
    img_save_dir = os.path.join(configs['output'], 'imgs')
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    elif configs['resume_epoch'] == 1:
        shutil.rmtree(img_save_dir)
        os.makedirs(img_save_dir)

if configs['params'] and (not many_gpus or local_rank == 0):
    params_dict = trainer.get_params()
    logger.info("模型的参数量为:{:.2f}M\tflops:{:.2f}G".format(params_dict['all']['params'] / 1e6, params_dict['all']['flops'] / 1e9))
    logger.info("text_encoder模型的参数量为:{:.2f}M\tflops:{:.2f}G".format(params_dict['text_encoder']['params'] / 1e6, params_dict['text_encoder']['flops'] / 1e9))
    logger.info("generator模型的参数量为:{:.2f}M\tflops:{:.2f}G".format(params_dict['generator']['params'] / 1e6, params_dict['generator']['flops'] / 1e9))

writer = None
fixed_img = None
fixed_sent = None
fixed_z = None

if configs['tensorboard']:
    logger.info("创建tensorboard writer...")
    writer = SummaryWriter(os.path.join(configs['output'], 'logs'))
    fixed_img, fixed_sent, fixed_z, fixed_caption = trainer.get_fix_data(train_dataloader, val_dataloader)
    fixed_grid = make_grid(fixed_img.cpu(), nrow=8, normalize=True)
    writer.add_image('fixed images', fixed_grid, 0)
    if img_save_dir:
        img_name = 'z.png'
        img_save_path = os.path.join(img_save_dir, img_name)
        torchvision.utils.save_image(fixed_img.data, img_save_path, nrow=8, normalize=True)
        with open(os.path.join(img_save_dir, 'captions.txt'), 'w', encoding='utf8') as f:
            for cap in fixed_caption:
                f.write("{}\n".format(cap))

start_epoch = 1
loss_list = []
best_fid = 1000
if configs['resume_epoch'] != 1:
    start_epoch = configs['resume_epoch'] + 1
    path = os.path.join(configs['resume_model_path'], 'state_epoch_%03d.pth' % (configs['resume_epoch']))
    trainer.load_model_opt(path)
    with open(os.path.join(configs['resume_model_path'], 'loss.json'), 'r', encoding='utf8') as f:
        loss_list = json.load(f)
    loss_list = loss_list[:configs['resume_epoch']]
    best_fid = min([c['fid'] for c in loss_list if 'fid' in c])
    if os.path.join(configs['resume_model_path'], 'imgs') != img_save_dir:
        shutil.copytree(os.path.join(configs['resume_model_path'], 'imgs'), img_save_dir, dirs_exist_ok=True)
    if os.path.exists(os.path.join(configs['resume_model_path'], 'imgs/captions.txt')):
        new_fixed_caption = []
        with open(os.path.join(configs['resume_model_path'], 'imgs/captions.txt'), 'r', encoding='utf8') as f:
            for line in f.readlines():
                new_fixed_caption.append(line.strip())
            fixed_caption = new_fixed_caption

if not many_gpus or configs.get('local_rank', -1) == 0:
    save_args(os.path.join(trainer.configs['output'], 'configs.yml'), trainer.configs)

logger.info("开始训练...")

torch.cuda.empty_cache()
start_time = time.time()
for epoch in range(start_epoch, configs['epochs'] + 1):
    if many_gpus and get_rank() != 0:
        train_sampler.set_epoch(epoch)
    torch.cuda.empty_cache()
    loss_dict = trainer.train(epoch, train_dataloader)
    loss_dict['epoch'] = epoch
    logger.info("train epoch:{} => ma_gp:{}, time:{}".format(epoch, sum(loss_dict['ma_gp']) / len(loss_dict['ma_gp']), loss_dict['time']))
    if not many_gpus or configs.get('local_rank', -1) == 0:
        trainer.save_img(fixed_z, fixed_sent, epoch, img_save_dir, writer)
        if writer:
            writer.add_scalar('g_loss', sum(loss_dict['g_losses']) / len(loss_dict['g_losses']), epoch)
            writer.add_scalar('d_loss', sum(loss_dict['d_losses']) / len(loss_dict['d_losses']), epoch)
            writer.add_scalar('ma_gp', sum(loss_dict['ma_gp']) / len(loss_dict['ma_gp']), epoch)
        if epoch % configs['test_interval'] == 0 or epoch == configs['epochs'] or (configs['resume_epoch'] != 1 and epoch == start_epoch):
            torch.cuda.empty_cache()
            fid = trainer.val(epoch, val_dataloader)
            loss_dict['fid'] = fid
            all_times = time.time() - start_time
            logger.info("val epoch:{} => fid:{}, 当前耗时为:{}h, 预计总耗时为:{}h, 剩余耗时为:{}h".format(epoch, fid, all_times / 3600, all_times / (epoch - start_epoch + 1) * (configs['epochs'] - start_epoch) / 3600, all_times / (epoch - start_epoch + 1) * (configs['epochs'] - epoch) / 3600))
            if writer:
                writer.add_scalar('FID', fid, epoch)
            if fid <= best_fid:
                logger.info("epoch:{}, new fid:{}, old fid:{}, 保存最优模型".format(epoch, fid, best_fid))
                trainer.save_model('best', epoch)
                best_fid = fid

        if epoch % configs['save_interval'] == 0 or epoch == configs['epochs']:
            logger.info("epoch:{}, 保存模型...".format(epoch))
            trainer.save_model('epoch', epoch)

        logger.info("保存最新模型...")
        trainer.save_model('latest', epoch)

        loss_list.append(loss_dict)

        with open(os.path.join(configs['output'], 'loss.json'), 'w', encoding='utf8') as f:
            json.dump(loss_list, f, ensure_ascii=False, indent=4)


