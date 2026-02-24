import os
import torch
import argparse
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, results_recorder, dict_to_namespace
from tensorboardX import SummaryWriter
from models.Gate_fusion import build_model
from core.metric import MetricsTop
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='configs/mosi.yaml')
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--gpu_id', type=int, default=0)
opt = parser.parse_args()
print(opt)

with open(opt.config_file) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
args = dict_to_namespace(args)
print(args)

seed = args.base.seed if opt.seed == -1 else opt.seed
gpu_id = args.base.gpu_id if opt.gpu_id == -1 else opt.gpu_id

print('-----------------args-----------------')
print(args)
print('-------------------------------------')

gpu_id = str(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Device: {device} ({gpu_id})")


def main():
    print(f"seed{seed}")
    setup_seed(seed)
    log_path = os.path.join(".", "log", args.base.project_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path="/root/tf-logs"

    save_path = os.path.join(args.base.ckpt_root, args.base.project_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = build_model(args).to(device)

    dataLoader = MMDataLoader(args)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.base.lr,
                                  weight_decay=args.base.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, args)

    loss_fn = torch.nn.MSELoss()

    metrics_fn = MetricsTop().getMetics(args.dataset.datasetName)

    training_results_recorder = results_recorder()
    validation_results_recorder = results_recorder()
    test_results_recorder = results_recorder()

    writer = SummaryWriter(logdir=log_path)

    best_non0_acc_2 = 0.0
    best_epoch = 0
    best_results = {}
    best_cpk_path = ""

    best_cpk_dir = "./best_cpk"
    os.makedirs(best_cpk_dir, exist_ok=True)

    for epoch in range(1, args.base.n_epochs + 1):
        training_ret = train(model, dataLoader['train'], optimizer, loss_fn, metrics_fn)
        validation_ret = evaluate(model, dataLoader['valid'], loss_fn, metrics_fn)
        test_ret = evaluate(model, dataLoader['test'], loss_fn, metrics_fn)

        training_results_recorder.update(training_ret['results'], epoch)
        validation_results_recorder.update(validation_ret['results'], epoch)
        test_results_recorder.update(test_ret['results'], epoch)
        best_validation_results = validation_results_recorder.get_best_results()
        best_test_results = test_results_recorder.get_best_results()

        current_non0_acc_2 = test_ret['results'].get('Non0_acc_2', 0.0)
        if current_non0_acc_2 > best_non0_acc_2:
            # delete previous best checkpoint
            if os.path.exists(best_cpk_path):
                os.remove(best_cpk_path)

            best_non0_acc_2 = current_non0_acc_2
            best_epoch = epoch
            best_results = test_ret['results']

            # save best model
            cpk_name = f"Non0_acc_2_{best_non0_acc_2:.4f}_epoch_{best_epoch}_seed_{seed}.pth"
            best_cpk_path = os.path.join(best_cpk_dir, cpk_name)
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Non0_acc_2': best_non0_acc_2,
                'seed': seed
            }, best_cpk_path)

        print(f'\n----------------- Results Epoch {epoch} -----------------')
        print(f'Learning Rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        print(f'Training Results: {training_ret["results"]}')
        print(f'Validation Results: {validation_ret["results"]}')
        print(f'Test Results: {test_ret["results"]}\n')
        print(f'Best Validation Results across All Epochs: {best_validation_results["best_results_all_epochs"]}')
        print(f'Best Validation Results of One Epoch: {best_validation_results["best_results_one_epoch"]}\n')
        print(f'Best Test Results across All Epochs: {best_test_results["best_results_all_epochs"]}')
        print(f'Best Test Results of One Epoch: {best_test_results["best_results_one_epoch"]}')
        print('----------------------------------------------------------\n')

        writer.add_scalar('train/MAE', training_ret['loss_recorder'].value_avg, epoch)
        writer.add_scalar('valid/MAE', validation_ret['loss_recorder'].value_avg, epoch)
        writer.add_scalar('test/MAE', test_ret['loss_recorder'].value_avg, epoch)
        
        # Note: For MOSI dataset, the best model parameters are achieved at epoch 31; for MOSEI, at epoch 4.
        # Consider implementing early stopping based on validation performance.
        scheduler_warmup.step()
        if epoch>99:
            break

    # add to best.txt
    with open('best.txt', 'a') as f:
        f.write(f'Seed: {seed}, Epoch: {best_epoch}, Results: {best_results}\n')

    writer.close()


def train(model, data_loader, optimizer, loss_fn, metrics_fn):
    loss_recorder = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    for cur_iter, data in enumerate(data_loader):
        img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        output = model(img, audio, text)

        loss = loss_fn(output, label)

        loss_recorder.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics_fn(pred, true)

    return {'results': results, 'loss_recorder': loss_recorder}


def evaluate(model, data_loader, loss_fn, metrics_fn):
    loss_recorder = AverageMeter()
    y_pred, y_true = [], []

    model.eval()

    for cur_iter, data in enumerate(data_loader):
        img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        with torch.no_grad():
            output = model(img, audio, text)

        loss = loss_fn(output, label)

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        loss_recorder.update(loss.item(), batchsize)

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics_fn(pred, true)

    return {'results': results, 'loss_recorder': loss_recorder}


if __name__ == '__main__':
    main()