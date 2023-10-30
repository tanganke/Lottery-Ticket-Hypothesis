from _common import *

log = logging.getLogger(__name__)

import copy
import pickle

from src import utils
from src.utils import timeit_context


def load_data(
    dataset: Literal["mnist", "cifar10", "fashionmnist", "cifar100"],
    batch_size: int,
    data_root: str = DATA_DIR,
    download: bool = True,
):
    if data_root is None:
        data_root = DATA_DIR
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_root, train=True, download=download, transform=transform)
        test_dataset = datasets.MNIST(data_root, train=False, download=download, transform=transform)
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(data_root, train=True, download=download, transform=transform)
        test_dataset = datasets.CIFAR10(data_root, train=False, download=download, transform=transform)
    elif dataset == "fashionmnist":
        train_dataset = datasets.FashionMNIST(data_root, train=True, download=download, transform=transform)
        test_dataset = datasets.FashionMNIST(data_root, train=False, download=download, transform=transform)
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(data_root, train=True, download=download, transform=transform)
        test_dataset = datasets.CIFAR100(data_root, train=False, download=download, transform=transform)
    # If you want to add extra datasets paste here
    else:
        print("\nWrong Dataset choice \n")
        exit(1)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    # train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    return train_loader, test_loader


def load_model(
    dataset: Literal["mnist", "cifar10", "fashionmnist", "cifar100"],
    arch_type: Literal["fc1", "lenet5", "alexnet", "vgg16", "resnet18", "densenet122"],
    device: torch.device = None,
) -> nn.Module:
    if dataset == "mnist" or dataset == "fashionmnist":
        from src.models.mnist import AlexNet, LeNet5, fc1, resnet, vgg
    elif dataset == "cifar10":
        from src.models.cifar10 import AlexNet, LeNet5, densenet, fc1, resnet, vgg
    elif dataset == "cifar100":
        from src.models.cifar100 import AlexNet, LeNet5, fc1, resnet, vgg
    # If you want to add extra datasets paste here
    else:
        log.error("Wrong Dataset choice")
        exit(1)

    if arch_type == "fc1":
        model = fc1.fc1()
    elif arch_type == "lenet5":
        model = LeNet5.LeNet5()
    elif arch_type == "alexnet":
        model = AlexNet.AlexNet()
    elif arch_type == "vgg16":
        model = vgg.vgg16()
    elif arch_type == "resnet18":
        model = resnet.resnet18()
    elif arch_type == "densenet121":
        model = densenet.densenet121()
    # If you want to add extra model paste here
    else:
        log.error("Wrong Model choice")
        exit(1)

    if device is not None:
        model = model.to(device=device)

    return model


def original_initialization(model: nn.Module, mask, initial_state_dict):
    for name, param in model.named_parameters():
        if "weight" in name:
            param.data = torch.from_numpy(mask[name] * initial_state_dict[name].cpu().numpy()).to(param.device)
        if "bias" in name:
            param.data = initial_state_dict[name]


def weight_init(m: nn.Module):
    """
    Initializes the weights of the given module using Xavier or normal distribution.

    Args:
        m (nn.Module): The module to initialize.

    Usage:
        >>> model = Model()
        >>> model.apply(weight_init)
    """
    from torch.nn import init

    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def prune_by_percentile(
    model: nn.Module,
    mask: Dict[str, Optional[Tensor]],
    percent: float,
):
    # Calculate percentile value
    for name, param in model.named_parameters():
        # We do not prune bias term
        if "weight" in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[name])

            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(param.device)
            mask[name] = new_mask


def make_mask(model: nn.Module) -> Dict[str, Optional[Tensor]]:
    mask = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            mask[name] = np.ones_like(param.data.cpu().numpy())
        else:
            mask[name] = None
    return mask


def test(model: nn.Module, test_loader, *, debug: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

            if debug:
                break  # early exit if debug
        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)

    return accuracy


def train(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion,
    *,
    debug: bool = False,
):
    eps = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if "weight" in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < eps, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()

        if debug:
            break
    return train_loss.item()


@hydra.main(str(CONFIG_DIR), "main", None)
def main(args: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reinit = True if args.prune_type == "reinit" else False

    save_dir = RESULTS_DIR / "saves" / args.arch_type / args.dataset
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = RESULTS_DIR / "plots" / "lt" / args.arch_type / args.dataset
    os.makedirs(plot_dir, exist_ok=True)
    dump_dir = RESULTS_DIR / "dumps" / "lt" / args.arch_type / args.dataset
    os.makedirs(dump_dir, exist_ok=True)

    with timeit_context("load dataset"):
        train_loader, test_loader = load_data(args.dataset, args.batch_size, args.data_root)
    with timeit_context("load model"):
        model = load_model(args.dataset, args.arch_type, device=device)

    # Weight Initialization
    model.apply(weight_init)

    # Copying and Saving Initial State
    with timeit_context("copying and saving initial state"):
        initial_state_dict = copy.deepcopy(model.state_dict())
        torch.save(model, save_dir / f"initial_state_dict_{args.prune_type}.pth")

    mask = make_mask(model)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()  # Default was F.nll_loss

    # Layer Looper
    for name, param in model.named_parameters():
        log.info(f"{name}: {param.size()}")

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestacc = 0.0
    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    step = 0
    all_loss = np.zeros(args.num_epochs, float)
    all_accuracy = np.zeros(args.num_epochs, float)

    for iter_idx in range(args.start_iter, ITERATION):
        if iter_idx != 0:
            prune_by_percentile(model, mask, args.prune_percent)
            if reinit:
                model.apply(weight_init)
                for name, param in model.named_parameters():
                    if "weight" in name:
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[name]).to(param.device)
            else:
                original_initialization(model, mask, initial_state_dict)
        print(f"\n--- Pruning Level [{iter_idx}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[iter_idx] = comp1

        for epoch_idx in (pbar := tqdm(range(args.num_epochs))):
            # Frequency for Testing
            if epoch_idx % args.valid_freq == 0:
                accuracy = test(model, test_loader, debug=args.debug)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model, save_dir / f"{iter_idx}_model_{args.prune_type}.pth")

            # Training
            loss = train(model, train_loader, optimizer, criterion, debug=args.debug)
            all_loss[epoch_idx] = loss
            all_accuracy[epoch_idx] = accuracy

            # Frequency for Printing Accuracy and Loss
            if epoch_idx % args.print_freq == 0:
                pbar.set_description(
                    f"Train Epoch: {epoch_idx}/{args.num_epochs} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%"
                )

            if args.debug:
                break

        # writer.add_scalar("Accuracy/test", best_accuracy, comp1)
        bestacc[iter_idx] = best_accuracy

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        # NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        # NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1, (args.num_epochs) + 1), 100 * (all_loss - np.min(all_loss)) / np.ptp(all_loss).astype(float), c="blue", label="Loss")
        plt.plot(np.arange(1, (args.num_epochs) + 1), all_accuracy, c="red", label="Accuracy")
        plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})")
        plt.xlabel("Iterations")
        plt.ylabel("Loss and Accuracy")
        plt.legend()
        plt.grid(color="gray")
        plt.savefig(plot_dir / f"{args.prune_type}_LossVsAccuracy_{comp1}.png", dpi=1200)
        plt.close()

        # Dump Plot values
        all_loss.dump(dump_dir / f"{args.prune_type}_all_loss_{comp1}.dat")
        all_accuracy.dump(dump_dir / f"{args.prune_type}_all_accuracy_{comp1}.dat")

        # Dumping mask
        with open(dump_dir / f"{args.prune_type}_mask_{comp1}.pkl", "wb") as fp:
            pickle.dump(mask, fp)

        # Making variables into 0
        best_accuracy = 0
        all_loss = np.zeros(args.num_epochs, float)
        all_accuracy = np.zeros(args.num_epochs, float)

    # Dumping Values for Plotting
    comp.dump(dump_dir / f"{args.prune_type}_compression.dat")
    bestacc.dump(dump_dir / f"{args.prune_type}_bestaccuracy.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestacc, c="blue", label="Winning tickets")
    plt.title(f"Test Accuracy vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("test accuracy")
    plt.xticks(a, comp, rotation="vertical")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(color="gray")
    plt.savefig(plot_dir / f"{args.prune_type}_AccuracyVsWeights.png", dpi=1200)
    plt.close()


if __name__ == "__main__":
    main()
