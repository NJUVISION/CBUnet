import torch


def save_checkpoint(model, optimizer, epoch, save_path):
    print(save_path)
    torch.save(dict(
        epoch=epoch,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict()
    ), save_path)
