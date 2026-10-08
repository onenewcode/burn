"""Independent PyTorch controls for issues 004, 010 and 011."""
import torch

print("torch", torch.__version__)
variance = torch.tensor([1e-8], requires_grad=True)
loss = torch.nn.GaussianNLLLoss()(torch.tensor([0.0]), torch.tensor([1.0]), variance)
loss.backward()
gradient = variance.grad.item()
assert abs(gradient / -499999500000.0 - 1.0) < 1e-6
print("004 Gaussian NLL: loss", loss.item(), "variance gradient", gradient)

group = torch.nn.GroupNorm(1, 2)(torch.tensor([[[1.0], [3.0]]]))
assert torch.allclose(group, torch.tensor([[[-1.0], [1.0]]]), atol=1e-5)
print("010 GroupNorm [1, 2, 1]:", group.tolist())

rms = torch.nn.RMSNorm(2, eps=1e-5, dtype=torch.float64)(
    torch.tensor([[1e20, 1e20]], dtype=torch.float64)
)
assert torch.allclose(rms, torch.ones_like(rms), atol=1e-12)
print("011 RMSNorm F64:", rms.tolist())
