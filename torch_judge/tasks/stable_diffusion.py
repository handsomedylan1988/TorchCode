"""Stable Diffusion classifier-free guidance and DDIM sampling task."""

TASK = {
    "title": "Stable Diffusion Sampling Step",
    "difficulty": "Hard",
    "function_name": "stable_diffusion_step",
    "hint": "Use CFG: eps = eps_uncond + scale * (eps_cond - eps_uncond). Estimate x0 from x_t and eps, then take the deterministic DDIM step with alpha_prev.",
    "tests": [
        {"name": "Preserves latent shape", "code": "import torch\nx_t = torch.randn(2, 4, 8, 8)\neps_c, eps_u = torch.randn_like(x_t), torch.randn_like(x_t)\nout = {fn}(x_t, eps_c, eps_u, torch.tensor(0.64), torch.tensor(0.81), 7.5)\nassert out.shape == x_t.shape, f'Shape: {out.shape}'\n"},
        {"name": "Matches deterministic DDIM with CFG", "code": "import torch\ntorch.manual_seed(0)\nx_t = torch.randn(1, 4, 3, 3)\neps_c, eps_u = torch.randn_like(x_t), torch.randn_like(x_t)\na_t, a_prev, scale = torch.tensor(0.49), torch.tensor(0.81), 2.0\nout = {fn}(x_t, eps_c, eps_u, a_t, a_prev, scale)\neps = eps_u + scale * (eps_c - eps_u)\nx0 = (x_t - (1 - a_t).sqrt() * eps) / a_t.sqrt()\nref = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps\nassert torch.allclose(out, ref, atol=1e-5), f'Max diff: {(out-ref).abs().max():.6f}'\n"},
        {"name": "Guidance scale zero uses unconditional prediction", "code": "import torch\nx_t = torch.randn(1, 4, 2, 2)\neps_c, eps_u = torch.randn_like(x_t), torch.randn_like(x_t)\na_t, a_prev = torch.tensor(0.36), torch.tensor(0.64)\nout = {fn}(x_t, eps_c, eps_u, a_t, a_prev, 0.0)\nx0 = (x_t - (1 - a_t).sqrt() * eps_u) / a_t.sqrt()\nref = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps_u\nassert torch.allclose(out, ref, atol=1e-5), 'scale=0 must use eps_uncond'\n"},
        {"name": "Gradient flow", "code": "import torch\nx_t = torch.randn(1, 4, 2, 2, requires_grad=True)\neps_c = torch.randn_like(x_t, requires_grad=True)\neps_u = torch.randn_like(x_t, requires_grad=True)\n{fn}(x_t, eps_c, eps_u, torch.tensor(0.5), torch.tensor(0.7)).sum().backward()\nassert x_t.grad is not None and eps_c.grad is not None and eps_u.grad is not None, 'Missing gradients'\n"},
    ],
}
