"""Flow Matching interpolation loss and Euler sampler task."""

TASK = {
    "title": "Flow Matching",
    "difficulty": "Hard",
    "function_name": "flow_matching_loss",
    "hint": "For x_t=(1-t)x0+t*x1, the target vector field is x1-x0. Minimize MSE to that field. In sampling, repeatedly add dt * velocity_fn(x, t).",
    "tests": [
        {"name": "Straight-path velocity target", "code": "import torch\nx0 = torch.tensor([[1., 2.], [-1., 3.]])\nx1 = torch.tensor([[4., 6.], [2., 5.]])\npred = x1 - x0\nloss = {fn}(pred, x0, x1)\nassert torch.allclose(loss, torch.tensor(0.)), f'Expected zero, got {loss}'\n"},
        {"name": "Matches MSE reduction", "code": "import torch\ntorch.manual_seed(0)\nx0, x1 = torch.randn(2, 3, 4), torch.randn(2, 3, 4)\npred = torch.randn_like(x0)\nout = {fn}(pred, x0, x1)\nref = ((pred - (x1 - x0)) ** 2).mean()\nassert torch.allclose(out, ref), f'Expected {ref}, got {out}'\n"},
        {"name": "Euler sampler integrates constant field", "code": "import torch\nx = torch.zeros(2, 3)\nvelocity_fn = lambda x, t: torch.full_like(x, 2.0)\nout = flow_matching_sample(velocity_fn, x, num_steps=20)\nassert torch.allclose(out, torch.full_like(x, 2.0), atol=1e-6), f'Got {out}'\n"},
        {"name": "Gradient flow", "code": "import torch\nx0, x1 = torch.randn(2, 3, requires_grad=True), torch.randn(2, 3)\npred = torch.randn(2, 3, requires_grad=True)\n{fn}(pred, x0, x1).backward()\nassert pred.grad is not None and x0.grad is not None, 'Missing gradients'\n"},
    ],
}
