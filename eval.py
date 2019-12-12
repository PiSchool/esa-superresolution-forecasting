import argparse
import torch
from PIL import Image
from torchvision import transforms
from data.cams import CAMSDataset
import os
from utils import save_tensors

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--results_dir', type=str, required=True, help='directory for saving results')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--batch_size', type=int, default=8, help='batch size used for training')
parser.add_argument('--seqlen', type=int, default=10, help='sequence length')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)
transform = transforms.Compose([transforms.CenterCrop(256), transforms.Resize(128), transforms.ToTensor()])
full_dataset = CAMSDataset('data/NO2/NO2S5/seqs', transform, seqlen=opt.seqlen)

kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':True}
dataloader = {'predict': torch.utils.data.DataLoader(full_dataset, batch_size=opt.batch_size , shuffle=False,**kwargs)}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = torch.load(opt.model)
print("number of parameters: {}".format(count_parameters(model)))
model.eval()

out_mean = []
target_mean = []
for idx, batch in enumerate(dataloader['predict']):
    with torch.no_grad():
        (input, target) = torch.split(batch, (opt.seqlen-2, 2), dim=1)
        target_fn = os.path.join(opt.results_dir, 'target' + str(idx) + '.png')
        out_fn = os.path.join(opt.results_dir, 'pred' + str(idx) + '.png')
        if opt.cuda:
            model = model.cuda()
            input = input.cuda()
            target = target.cuda()
        out = model(input)
        out_mean.append(torch.mean(out[0]))
        target_mean.append(torch.mean(target))
        full_seq = torch.cat((input, out), dim=1)
        full_target_seq = torch.cat((input, target), dim=1)
        save_tensors(full_seq, out_fn)
        save_tensors(full_target_seq, target_fn)


        # out = out.cpu()
        # out_mean.append(torch.mean(out))
        # target = target[:, 0].cpu()
        # target_mean.append(torch.mean(target))
        #
        # out_img = out[0].detach().numpy()
        # out_img *= 255.0
        # out_img = out_img.clip(0, 255)
        # out_img = Image.fromarray(np.uint8(out_img[0]), mode='L')
        #
        # out_img.save(out_fn)
        # print('output image saved to ', out_fn)
        #
        #
        # out_target = target[0].detach().numpy()
        # out_target *= 255.0
        # out_target = out_target.clip(0, 255)
        # out_target = Image.fromarray(np.uint8(out_target[0]), mode='L')
        #
        # out_target.save(target_fn)
        # print('output image saved to ', target_fn)

print("Output average: {}".format(sum(out_mean) / len(out_mean)))
print("Target average: {}".format(sum(target_mean) / len(target_mean)))
