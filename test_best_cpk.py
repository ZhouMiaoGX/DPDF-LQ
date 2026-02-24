import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, results_recorder, dict_to_namespace
from tensorboardX import SummaryWriter
from models.Gate_fusion import build_model as build_new
from core.metric import MetricsTop
import csv
import shutil
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"device: {device}")
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='configs/mosei.yaml')
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--gpu_id', type=int, default=0)
opt = parser.parse_args()
print(opt)

with open(opt.config_file) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
args = dict_to_namespace(args)
print(args)

def main():
    # model path
    test_path = "/root/autodl-tmp/DPDF-LQ/best_cpk/Non0_acc_2_0.8621_epoch_4_seed_526.pth"
    print("model_test_path:", test_path)

    model = build_new(args).to(device)
    
    if os.path.exists(test_path):
        checkpoint = torch.load(test_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {test_path}")
    else:
        print(f"Model weights file not found at {test_path}")
        return

    dataLoader = MMDataLoader(args)
    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(args.dataset.datasetName)

    evaluate(model, dataLoader['test'], loss_fn, metrics)


def evaluate(model, eval_loader, loss_fn, metrics):
    test_pbar = tqdm(enumerate(eval_loader))
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval() 
    with torch.no_grad(): 
        for cur_iter, data in test_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)

            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output= model(img,audio,text)

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

        pred, true = torch.cat(y_pred), torch.cat(y_true)

        test_results = metrics(pred, true)
        print("Test Results:")
        for key, value in test_results.items():
            print(f"{key}: {value:.4f}")

        best_results=test_results
        with open('result.txt', 'a') as f:
            has0_acc_2 = best_results.get('Has0_acc_2', 0.0) * 100
            non0_acc_2 = best_results.get('Non0_acc_2', 0.0) * 100
            has0_f1_score = best_results.get('Has0_F1_score', 0.0) * 100
            non0_f1_score = best_results.get('Non0_F1_score', 0.0) * 100
            mult_acc_5 = best_results.get('Mult_acc_5', 0.0) * 100
            mult_acc_7 = best_results.get('Mult_acc_7', 0.0) * 100
            mae = round(best_results.get('MAE', 0.0), 3)
            corr = round(best_results.get('Corr', 0.0), 3)

            f.write(f"{has0_acc_2:.2f}/{non0_acc_2:.2f} {has0_f1_score:.2f}/{non0_f1_score:.2f} {mult_acc_5:.2f} {mult_acc_7:.2f} {mae:.3f} {corr:.3f}\n")


if __name__ == '__main__':
    main()