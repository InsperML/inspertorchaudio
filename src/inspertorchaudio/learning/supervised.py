"""
Procedures for supervised learning experiments.
"""

# pylint: disable=missing-docstring
import mlflow
import mlflow.pytorch
import torch
import torch.nn.functional as F
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from tqdm import tqdm


def eval_step(model, x, y, loss_fn=F.cross_entropy):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        acc = (y_pred.argmax(dim=1) == y).float().mean().item()
    return loss.item(), acc


def eval_epoch(model, dataloader, use_cuda=False):
    total_loss = 0.0
    total_acc = 0.0
    for x, y in tqdm(dataloader, desc='Evaluating'):
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        else:
            x, y = x.cpu(), y.cpu()
        loss, acc = eval_step(model, x, y)
        total_loss += loss
        total_acc += acc
    return total_loss / len(dataloader), total_acc / len(dataloader)


def train_step(model, optimizer, x, y, loss_fn=F.cross_entropy):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(model, optimizer, dataloader, use_cuda=False):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(dataloader, desc='Training'):
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        else:
            x, y = x.cpu(), y.cpu()
        loss = train_step(model, optimizer, x, y)
        total_loss += loss
    return total_loss / len(dataloader)


def train_with_mlflow(experiment_name, description, *args, **kwargs):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(log_system_metrics=True, description=description):
        train(*args, **kwargs)


def train(
    model,
    optimizer,
    train_dataloader,
    eval_dataloader,
    epochs=10,
    use_cuda='auto',
    use_eval=True,
    use_mlflow=False,
    model_name='BestModel',
    patience_for_stop=5,
    lr_scheduler=None,
):
    # if model is in cuda:
    if use_cuda == 'auto':
        device_type = next(model.parameters()).device.type
        use_cuda = device_type == 'cuda'
    elif not isinstance(use_cuda, bool):
        raise ValueError(
            'use_cuda must be "auto", True, or False. '
            f'Got {use_cuda} instead.'
        )
    print(f'Using CUDA: {use_cuda}')

    last_acc = 0.0
    epochs_with_no_improvement = 0
    lr = optimizer.param_groups[0]['lr']
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        try:
            avg_train_loss = train_epoch(
                model,
                optimizer,
                train_dataloader,
                use_cuda,
            )
            print(f'Average Train Loss: {avg_train_loss:.4f}')
            if use_eval:
                avg_val_loss, avg_val_acc = eval_epoch(
                    model,
                    eval_dataloader,
                    use_cuda,
                )
                print(f'Average Validation Loss: {avg_val_loss:.4f}, '
                      f'Average Validation Accuracy: {avg_val_acc:.4f}')
            else:
                avg_val_loss, avg_val_acc = None, None
        except KeyboardInterrupt:
            print('Training interrupted by user.')
            break

        if use_eval and lr_scheduler is not None:
            lr_scheduler.step(avg_val_loss)
            lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate adjusted to: {lr}')

        if use_mlflow:
            log_mlflow(epoch, avg_train_loss, avg_val_loss, avg_val_acc, lr)
        if use_eval and avg_val_acc > last_acc and use_mlflow:
            input_schema = Schema([ColSpec('float', 'input_features')])
            output_schema = Schema([ColSpec('float', 'output')])
            signature = ModelSignature(
                inputs=input_schema,
                outputs=output_schema,
            )
            mlflow.pytorch.log_model(
                model,
                name='best_model',
                registered_model_name=model_name,
                signature=signature,
            )
        if avg_val_acc > last_acc:
            last_acc = avg_val_acc
            epochs_with_no_improvement = 0
        else:
            epochs_with_no_improvement += 1
            if epochs_with_no_improvement >= patience_for_stop:
                print(f'No improvement for {patience_for_stop} '
                      'epochs, stopping training.')
                break
        print(f'Epoch {epoch + 1} completed.\n')


def log_mlflow(epoch, train_loss, val_loss, val_acc, lr):
    mlflow.log_metric('train_loss', train_loss, step=epoch)
    mlflow.log_metric('val_loss', val_loss, step=epoch)
    mlflow.log_metric('val_acc', val_acc, step=epoch)
    mlflow.log_metric('learning_rate', lr, step=epoch)

