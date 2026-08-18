"""Toy DDPM loss and reverse sampler task."""

TASK = {
    "title": "Stable Diffusion Foundations — Toy DDPM",
    "difficulty": "Hard",
    "function_name": "diffusion_loss",
    "hint": "Sample an integer timestep for every x0, form x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise, and regress model(x_t, normalized_t) to the sampled noise.",
    "tests": [
        {"name": "Produces a scalar loss", "code": "import torch\nfrom torch import nn\nclass Field(nn.Module):\n    def forward(self, x, t): return self.linear(torch.cat((x, t), dim=1))\n    def __init__(self): super().__init__(); self.linear = nn.Linear(3, 2)\nmodel = Field(); out = {fn}(model, torch.randn(8, 2), torch.linspace(.99, .8, 5))\nassert out.ndim == 0\n"},
        {"name": "Normalized timesteps and gradients", "code": "import torch\nfrom torch import nn\nclass Field(nn.Module):\n    def __init__(self): super().__init__(); self.linear = nn.Linear(3, 2)\n    def forward(self, x, t):\n        assert t.shape == (6, 1) and torch.all((t >= 0) & (t <= 1))\n        return self.linear(torch.cat((x, t), dim=1))\nmodel = Field(); {fn}(model, torch.randn(6, 2), torch.linspace(.99, .8, 7)).backward()\nassert model.linear.weight.grad is not None\n"},
        {"name": "DDPM sampler preserves shape and input", "code": "import torch\nfrom torch import nn\nclass ZeroField(nn.Module):\n    def forward(self, x, t): return torch.zeros_like(x)\nx = torch.randn(4, 2); before = x.clone(); betas = torch.linspace(1e-4, .02, 5); alphas = 1 - betas; bars = torch.cumprod(alphas, 0)\nout = sample_ddpm(ZeroField(), x, betas, alphas, bars)\nassert out.shape == x.shape and torch.allclose(x, before) and torch.isfinite(out).all()\n"},
    ],
}
