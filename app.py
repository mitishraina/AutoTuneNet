# from benchmarks.common.model import CNN
# from benchmarks.common.dataset import get_mnist_loaders
# from benchmarks.common.train_utils import train_model
# import torch


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNN().to(device)
# train_loader, val_loader = get_mnist_loaders()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# history = train_model(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     device,
#     epochs=1
# )

# print(history)