import argparse
import random
import time
import numpy as np
import torch
import os
import sys
from torch import nn, optim
from tqdm.notebook import tqdm, trange

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.ibm_backend import IBMBackend
from src.datasets import get_dataloaders, get_datasets
from src.utils import plot_images
from src.model_utils import get_model
from src.utils import *
from src.engine import train, evaluate, epoch_time

# python3 src/train.py --epochs 10 --batch_size 256 --data_dir input/Data --output_dir outputs/ --lr 0.001

parser = argparse.ArgumentParser(
    description="Train Hybrid Quantum Convolutional Neural Networks"
)
parser.add_argument("--epochs", type=int, help="epochs")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--data_dir", type=str, help="data directory")
parser.add_argument("--output_dir", type=str, help="output directory")
parser.add_argument("--model", type=str, help="model name", default="alexnet_qnn")
parser.add_argument("--num_workers", type=int, help="number of workers", default=4)
parser.add_argument("--pin_memory", type=bool, help="pin memory", default=True)
parser.add_argument("--pretrained", type=bool, help="pretrained", default=False)
parser.add_argument("--train_ratio", type=float, help="train ratio", default=0.8)
parser.add_argument("--val_ratio", type=float, help="val ratio", default=0.1)
parser.add_argument("--qnn", type=bool, help="quantum neural network", default=False)
parser.add_argument("--lr", type=float, help="learning rate", default=-1)
parser.add_argument("--seed", type=float, help="seed", default=42)
parser.add_argument(
    "--backend", type=str, help="backend", default="ibmq_qasm_simulator"
)
parser.add_argument("--device", type=str, help="device", default="cuda")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    print(f'\nIs training a QNN: {args.qnn}\n')

    backend = None
    if args.qnn:
        print("Loading IBM backend...")
        ibm_backend = IBMBackend()
        provider = ibm_backend.get_provider()
        backend = ibm_backend.get_backend(backend_name=args.backend)

    print("Setting seed...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    print("Loading dataset...")
    train_data, test_data, valid_data = get_datasets(
        args.data_dir,
        os.path.join(args.data_dir, "../train"),
        os.path.join(args.data_dir, "../test"),
        os.path.join(args.data_dir, "../val"),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print("Loading dataloaders...")
    train_iterator, test_iterator, valid_iterator = get_dataloaders(
        train_data,
        test_data,
        valid_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    if not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)

    classes = os.listdir(os.path.join(args.data_dir, "../train"))
    if not os.path.exists(os.path.join(args.output_dir, "data_sample.png")):
        print("Plotting data sample. Saving to output directory...")
        images, labels = train_iterator.__iter__().__next__()
        plot_images(images, labels, classes, args.output_dir)

    START_LR = 1e-7
    END_LR = 10
    NUM_ITER = 100
    device = torch.device(args.device)
    model = get_model(args.model, len(classes), backend=backend, is_qnn=args.qnn)
    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    if args.lr <= 0:
        print("Finding learning rate...")
        lr_finder = LRFinder(model, optimizer, criterion, args.output_dir, device)
        lrs, losses = lr_finder.range_test(train_iterator, END_LR, NUM_ITER)

        print("Plotting learning rate finder. Saving to output directory...")
        plot_lr_finder(lrs, losses, args.output_dir, skip_start=30, skip_end=30)
        print("Getting best learning rate...")
        found_lr = 1e-3
        print(f"Found learning rate: {found_lr}")
        optimizer = optim.Adam(model.parameters(), lr=found_lr)
    else:
        print(f"Using learning rate: {args.lr}")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Setting up scheduler...")
    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = STEPS_PER_EPOCH * args.epochs
    MAX_LRS = [p["lr"] for p in optimizer.param_groups]
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LRS, total_steps=TOTAL_STEPS
    )

    print("Training...")
    best_valid_loss = float("inf")
    for epoch in trange(args.epochs, desc="Epochs"):
        start_time = time.monotonic()

        train_loss, train_acc = train(
            model, train_iterator, optimizer, criterion, device
        )
        valid_loss, valid_acc = evaluate(
            model, valid_iterator, criterion, device
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, f"{args.model}.pt")
            )

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

    model.load_state_dict(torch.load(os.path.join(args.output_dir, f"{args.model}.pt")))
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")

    images, labels, probs = get_predictions(model, test_iterator, device)

    pred_labels = torch.argmax(probs, 1)
    plot_confusion_matrix(labels, pred_labels, classes, args.output_dir)

    corrects = torch.eq(labels, pred_labels)
    incorrect_examples = []
    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if not correct:
            incorrect_examples.append((image, label, prob))

    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)
    corrects = torch.eq(labels, pred_labels)
    plot_most_incorrect(incorrect_examples, classes, args.output_dir)
