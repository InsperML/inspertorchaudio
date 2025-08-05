import inspertorchaudio.learning.supervised as supervised
import torch

def test_train_and_eval_step():
    """Test the train_step and eval_step functions with a simple model and data."""
    # Create a simple model
    model = torch.nn.Linear(10, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create dummy data
    x = torch.randn(5, 10)
    y = torch.randint(0, 3, (5,))

    assert y.shape == (5,)

    # Test train_step
    initial_params = [param.clone() for param in model.parameters()]
    loss = supervised.train_step(model, optimizer, x, y)
    assert isinstance(loss, float), "train_step should return a float loss value."
    updated_params = [param for param in model.parameters()]
    assert any(not torch.equal(ip, up) for ip, up in zip(initial_params, updated_params)), "Model parameters should be updated after train_step."

    # Test eval_step
    eval_loss, eval_acc = supervised.eval_step(model, x, y)
    assert isinstance(eval_loss, float), "eval_step should return a float loss value."
    assert isinstance(eval_acc, float), "eval_step should return a float accuracy value."
    assert 0.0 <= eval_acc <= 1.0, "Accuracy should be between 0 and 1."
    
def test_train_epoch():
    """Test the train_epoch function with a simple model and data."""
    # Create a simple model
    model = torch.nn.Linear(10, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create dummy data loader
    x = torch.randn(20, 10)
    y = torch.randint(0, 3, (20,))
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)

    # Test train_epoch
    avg_loss = supervised.train_epoch(model, optimizer, dataloader)
    assert isinstance(avg_loss, float), "train_epoch should return a float average loss value."
    assert avg_loss > 0, "Average loss should be positive."