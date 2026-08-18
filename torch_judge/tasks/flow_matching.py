"""Conditional Flow Matching loss and Euler sampler task."""

TASK = {
    "title": "Conditional Flow Matching",
    "difficulty": "Hard",
    "function_name": "compute_fm_loss",
    "hint": "Sample t with shape (B, 1), form x_t=(1-t)*x0+t*x1, and regress model(x_t, t) to x1-x0. For sampling, clone x0 and repeatedly add dt * model(x, t).",
    "tests": [
        {"name": "Straight-path velocity target", "code": "import torch\nfrom torch import nn\nclass ExactField(nn.Module):\n    def forward(self, x, t): return x1 - x0\nx0 = torch.tensor([[1., 2.], [-1., 3.]])\nx1 = torch.tensor([[4., 6.], [2., 5.]])\nloss = {fn}(ExactField(), x0, x1)\nassert torch.allclose(loss, torch.tensor(0.)), f'Expected zero, got {loss}'\n"},
        {"name": "Uses interpolated inputs and time", "code": "import torch\nfrom torch import nn\nclass Recorder(nn.Module):\n    def forward(self, x, t):\n        assert x.shape == (5, 2) and t.shape == (5, 1)\n        assert torch.all((t >= 0) & (t <= 1))\n        return torch.zeros_like(x)\nx0, x1 = torch.randn(5, 2), torch.randn(5, 2)\nout = {fn}(Recorder(), x0, x1)\nassert out.ndim == 0\n"},
        {"name": "Euler sampler integrates constant field", "code": "import torch\nfrom torch import nn\nclass ConstantField(nn.Module):\n    def forward(self, x, t): return torch.full_like(x, 2.0)\nx0 = torch.zeros(2, 3)\nout = sample_ode(ConstantField(), x0, steps=20)\nassert torch.allclose(out, torch.full_like(x0, 2.0), atol=1e-6)\nassert torch.allclose(x0, torch.zeros_like(x0)), 'sample_ode must not mutate x0'\n"},
        {"name": "Gradient flows through the model", "code": "import torch\nfrom torch import nn\nclass SmallField(nn.Module):\n    def __init__(self):\n        super().__init__(); self.linear = nn.Linear(3, 2)\n    def forward(self, x, t): return self.linear(torch.cat((x, t), dim=1))\nmodel = SmallField()\nx0, x1 = torch.randn(4, 2), torch.randn(4, 2)\n{fn}(model, x0, x1).backward()\nassert model.linear.weight.grad is not None\n"},
    ],
}
