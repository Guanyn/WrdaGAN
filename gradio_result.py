# !/bin/bash

# @Author: qiaoyongchao
# @Email: qiaoyc2023@163.com


from src.models import *
from src.utils import *

import gradio as gr

import os
import argparse
from loguru import logger
import time
import clip

parser = argparse.ArgumentParser(description='计算测试集的clip match score')

parser.add_argument('--config', default='./configs/coco.yml')
parser.add_argument('--config2', default='./configs/bird.yml')
parser.add_argument('--load_dir', default='./output')
parser.add_argument("--mode", help='代码运行模式(train或者test)', choices=['test', 'train'], default='test')


parser.add_argument("--log", help='日志保存文件', default='./service.log')
parser.add_argument("--port", help='端口号', default=7860, type=int)

args = parser.parse_args()

configs = get_config(args)

logger.add(configs['log'])
logger.info("日志文件为:{}".format(configs['log']))

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device(0)

clip_model, preprocess = clip.load("ViT-B/32", device=device)

text_encoder_coco = CLIP_TXT_ENCODER(clip_model)
netS_coco = SemanticCorrection(clip_model, configs['model']['embedding_dim'], configs['ch_size']).to(device)
netG_coco = NetG(configs['nf'], configs['z_dim'], configs['model']['embedding_dim'], configs['imsize'], configs['ch_size']).to(device)

checkpoint_coco = torch.load(os.path.join(configs['load_dir'], 'coco/state_best.pth'), map_location=torch.device('cpu'))
netG_coco = load_model_weights(netG_coco, checkpoint_coco['model']['netG'], False)
netS_coco = load_model_weights(netS_coco, checkpoint_coco['model']['netS'], False)

netG_coco = netG_coco.to(device)
netS_coco = netS_coco.to(device)

text_encoder_bird = CLIP_TXT_ENCODER(clip_model)
netS_bird = SemanticCorrection(clip_model, configs['config2']['model']['embedding_dim'], configs['config2']['ch_size']).to(device)
netG_bird = NetG(configs['config2']['nf'], configs['config2']['z_dim'], configs['config2']['model']['embedding_dim'], configs['config2']['imsize'], configs['config2']['ch_size']).to(device)

checkpoint_bird = torch.load(os.path.join(configs['load_dir'], 'bird/state_best.pth'), map_location=torch.device('cpu'))
netG_bird = load_model_weights(netG_bird, checkpoint_bird['model']['netG'], False)
netS_bird = load_model_weights(netS_bird, checkpoint_bird['model']['netS'], False)

netG_bird = netG_bird.to(device)
netS_bird = netS_bird.to(device)


def predict(text, model, noise=None):
    if model == 'coco':
        net_G = netG_coco
        text_encoder = text_encoder_coco
        net_S = netS_coco
    else:
        net_G = netG_bird
        text_encoder = text_encoder_bird
        net_S = netS_bird

    start = time.time()
    if not noise or len(noise) != configs['z_dim']:
        noise = torch.randn(1, configs['z_dim']).to(device)
    sent_token = clip.tokenize(text, truncate=True)
    sent_token = sent_token.to(device)

    sent_emb, words_embs = text_encoder(sent_token)

    fake = net_G(noise, sent_emb)
    fake = net_S(fake, sent_emb)
    end = time.time()
    print(end - start)

    fake = torch.nn.functional.interpolate(fake * 0.5 + 0.5, size=(256, 256))

    res = fake.cpu().data.numpy()[0].astype(np.float32)

    res = res.transpose(1, 2, 0)
    return res


title = "智能创想家：Atlas赋能的多模态“广告设计师”"

examples = [
    ['two people playing soccer on a green field', 'coco'],
    ['A group of people riding boards in the ocean.', 'coco'],
    ['A small plane that is parked in a hanger.', 'coco'],
    ["this bird has a brown crown as well as a yellow belly.", 'bird'],
    ["this bird is medium size with beautiful cream colored feathers with light brown mingled in.", 'bird'],
    ["a small bird with green wings with a short beak.", 'bird'],
    ["这是一种黑色的小鸟，它的头和它的小身体成正比，喙向下弯曲。", 'bird'],
    ["这只白色的小鸟有点胖，有一个中等大小的灰色喙。", 'bird'],
    ['一群斑马站在一块岩石地上。', 'coco'],
    ['一只长颈鹿在草地上行走。', 'coco'],
    ['一个穿着棒球服的年轻人正在扔棒球。', 'coco'],
    ['一个女人骑着摩托车在绿色的田野上。', 'coco']
]

demo = gr.Interface(
    fn=predict,
    inputs=["text", gr.Radio(['coco', 'bird'])],
    outputs=[gr.Image(height=256, width=256)],
    title=title,
    examples=examples
)
demo.launch(server_name="0.0.0.0", server_port=args.port)

