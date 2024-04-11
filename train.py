import logging
import os
import sys
import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from dataloader.MicroLens100k.dataset import MyData, custom_collate_fn
from model.MicroLens100k.MMRA import Model
import random
from functools import partial

BLUE = '\033[94m'
ENDC = '\033[0m'


def seed_init(seed):

    seed = int(seed)

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True


def print_init_msg(logger, args):

    logger.info(BLUE + 'Random Seed: ' + ENDC + f"{args.seed} ")

    logger.info(BLUE + 'Device: ' + ENDC + f"{args.device} ")

    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_id} ")

    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")

    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")

    logger.info(BLUE + "Optimizer: " + ENDC + f"{args.optim}(lr = {args.lr})")

    logger.info(BLUE + "Total Epoch: " + ENDC + f"{args.epochs} Turns")

    logger.info(BLUE + "Early Stop: " + ENDC + f"{args.early_stop_turns} Turns")

    logger.info(BLUE + "Batch Size: " + ENDC + f"{args.batch_size}")

    logger.info(BLUE + "Number of retrieved items used in this training: " + ENDC + f"{args.num_of_retrieved_items}")

    logger.info(BLUE + "Alpha: " + ENDC + f"{args.alpha}")

    logger.info(BLUE + "Number of frames: " + ENDC + f"{args.frame_num}")

    logger.info(BLUE + "Training Starts!" + ENDC)


def make_saving_folder_and_logger(args):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder_name = f"train_{args.model_id}_{args.dataset_id}_{args.metric}_{timestamp}"

    father_folder_name = args.save

    if not os.path.exists(father_folder_name):

        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)

    os.mkdir(folder_path)

    os.mkdir(os.path.join(folder_path, "trained_model"))

    logger = logging.getLogger()

    logger.handlers = []

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')

    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    return father_folder_name, folder_name, logger


def delete_model(father_folder_name, folder_name, min_turn):

    model_name_list = os.listdir(f"{father_folder_name}/{folder_name}/trained_model")

    for i in range(len(model_name_list)):

        if model_name_list[i] != f'model_{min_turn}.pth':

            os.remove(os.path.join(f'{father_folder_name}/{folder_name}/trained_model', model_name_list[i]))


def force_stop(msg):

    print(msg)

    sys.exit(1)


def delete_special_tokens(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:

        content = file.read()

    content = content.replace(BLUE, '')

    content = content.replace(ENDC, '')

    with open(file_path, 'w', encoding='utf-8') as file:

        file.write(content)


def train_val(args):

    father_folder_name, folder_name, logger = make_saving_folder_and_logger(args)

    device = torch.device(args.device)

    custom_collate_fn_partial = partial(custom_collate_fn, num_of_retrieved_items=args.num_of_retrieved_items,
                                        num_of_frames=args.frame_num)

    train_data = MyData(os.path.join(args.dataset_path, args.dataset_id, 'train.pkl'))

    valid_data = MyData(os.path.join(os.path.join(args.dataset_path, args.dataset_id, 'valid.pkl')))

    train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=custom_collate_fn_partial)

    valid_data_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, collate_fn=custom_collate_fn_partial)

    model = Model(feature_dim=args.feature_dim, alpha=args.alpha, frame_num=args.frame_num)

    model = model.to(device)

    if args.loss == 'BCE':

        loss_fn = torch.nn.BCELoss()

    elif args.loss == 'MSE':

        loss_fn = torch.nn.MSELoss()

    else:

        force_stop('Invalid parameter loss!')

    loss_fn.to(device)

    if args.optim == 'Adam':

        optim = Adam(model.parameters(), args.lr)

    elif args.optim == 'SGD':

        optim = SGD(model.parameters(), args.lr)

    else:

        force_stop('Invalid parameter optim!')

    min_total_valid_loss = 1008611

    min_turn = 0

    print_init_msg(logger, args)

    for i in range(args.epochs):

        logger.info(f"-----------------------------------Epoch {i + 1} Start!-----------------------------------")

        min_train_loss, total_valid_loss = run_one_epoch(model, loss_fn, optim, train_data_loader, valid_data_loader,
                                                         device)

        logger.info(f"[ Epoch {i + 1} (train) ]: loss = {min_train_loss}")

        logger.info(f"[ Epoch {i + 1} (valid) ]: total_loss = {total_valid_loss}")

        if total_valid_loss < min_total_valid_loss:
            min_total_valid_loss = total_valid_loss

            min_turn = i + 1

        logger.critical(
            f"Current Best Total Loss comes from Epoch {min_turn} , min_total_loss = {min_total_valid_loss}")

        torch.save(model, f"{father_folder_name}/{folder_name}/trained_model/model_{i + 1}.pth")

        logger.info("Model has been saved successfully!")

        if (i + 1) - min_turn > args.early_stop_turns:
            break

    delete_model(father_folder_name, folder_name, min_turn)

    logger.info(BLUE + "Training is ended!" + ENDC)

    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")


def run_one_epoch(model, loss_fn, optim, train_data_loader, valid_data_loader, device):

    model.train()

    min_train_loss = 1008611

    for batch in tqdm(train_data_loader, desc='Training Progress'):

        batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

        visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding, \
            retrieved_textual_feature_embedding, retrieved_label, label = batch

        output = model.forward(visual_feature_embedding, textual_feature_embedding, similarity,
                               retrieved_visual_feature_embedding,
                               retrieved_textual_feature_embedding, retrieved_label)

        loss = loss_fn(output, label)

        optim.zero_grad()

        loss.backward()

        optim.step()

        if min_train_loss > loss:

            min_train_loss = loss

    model.eval()

    total_valid_loss = 0

    with torch.no_grad():

        for batch in tqdm(valid_data_loader, desc='Validating Progress'):

            batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]

            visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding, \
                retrieved_textual_feature_embedding, retrieved_label, label = batch

            output = model.forward(visual_feature_embedding, textual_feature_embedding, similarity,
                                   retrieved_visual_feature_embedding, retrieved_textual_feature_embedding,
                                   retrieved_label)

            output = output.to('cpu')

            label = label.to('cpu')

            output = np.array(output)

            label = np.array(label)

            MAE = mean_absolute_error(label, output)

            nMSE = np.mean(np.square(output - label)) / (label.std() ** 2)

            loss = MAE + nMSE

            total_valid_loss += loss

    return min_train_loss, total_valid_loss


def main(args):

    seed_init(args.seed)

    train_val(args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=str, default='2024', help='Seed for reproducibility')

    parser.add_argument('--device', type=str, default='cuda:0', help='Device for training')

    parser.add_argument('--metric', type=str, default='MSE', help='Metric for evaluation')

    parser.add_argument('--save', type=str, default='train_results', help='Directory to save results')

    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

    parser.add_argument('--early_stop_turns', type=int, default=20, help='Number of turns for early stopping')

    parser.add_argument('--loss', type=str, default='MSE', help='Loss function for training')

    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer for training')

    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')

    parser.add_argument('--dataset_id', type=str, default='MicroLens-100k', help='Dataset identifier')

    parser.add_argument('--dataset_path', type=str, default='data', help='Path to the dataset')

    parser.add_argument('--model_id', type=str, default='MMRA', help='Model id')

    parser.add_argument('--feature_num', type=int, default=2, help='Number of features')

    parser.add_argument('--num_of_retrieved_items', type=int, default=10, help='Number of retrieved items, hyper-parameter')

    parser.add_argument('--feature_dim', type=int, default=768, help='Dimension of features')

    parser.add_argument('--label_dim', type=int, default=1, help='Dimension of labels')

    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha, hyper-parameter')

    parser.add_argument('--frame_num', type=int, default=10, help='Number of frames, hyper-parameter')

    args_ = parser.parse_args()

    main(args_)
