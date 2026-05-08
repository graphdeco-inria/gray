import pytest

from gray.imports import *
from gray.prelude import *
from tests.utils import *
from torchvision.transforms import functional as TF
from PIL import Image

@pytest.mark.parametrize("t_threshold", [0.0, 0.5])
def test_multiple_gaussians(t_threshold):
    NUM_PIXELS = 32
    target = TF.to_tensor(Image.open("tests/data/target_1.png").convert("RGB")).cuda()

    cfg = RaytracerConfig(t_threshold=t_threshold, sh=False)

    raytracer = Raytracer(cfg, 3, NUM_PIXELS, NUM_PIXELS) 

    max_alpha = raytracer.cuda_module.get_max_alpha()
    config = raytracer.cuda_module.get_config()
    config.zero_grads.fill_(False)
    exp_power = config.exp_power.item()
    alpha_threshold = config.alpha_threshold.item()
    t_threshold = config.t_threshold.item() 
    eps_forward_normalization = config.eps_forward_normalization.item()

    mu = torch.tensor([
        [0.05, 0.05, -0.25],
        [-0.05, -0.05, 0.25],
        [0.0, 0.0, 0.0],
    ], device="cuda").requires_grad_()
    log_scale = torch.tensor([
        [1.0, 0.2, 0.1],
        [0.1, 0.3, 0.3],
        [0.4, 0.4, 0.4],
    ], device="cuda").log().requires_grad_()
    color = torch.tensor([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], device="cuda").requires_grad_()
    unnormalized_rotation = torch.tensor([
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ], device="cuda").requires_grad_()
    logit_opacity = torch.tensor([
        [0.5],
        [0.7],
        [0.7],
    ], device="cuda").logit().requires_grad_()

    torch.cuda.synchronize()
    gaussians = raytracer.cuda_module.get_gaussians()
    gaussians.mean.copy_(mu.detach())
    gaussians.scale.copy_(log_scale.detach())
    gaussians.channels.copy_(color.detach())
    gaussians.rotation.copy_(unnormalized_rotation.detach())
    gaussians.opacity.copy_(logit_opacity.detach())
    torch.cuda.synchronize()
    raytracer.cuda_module.rebuild_bvh() 

    class mock_camera:
        origin = np.array([0.0, 0.0, 1.0])
        R = -np.eye(3)
        R[:, 2] *= 1
        fov_y = 1.5
        aspect = 1.0

    render = raytracer(mock_camera)
    loss = F.l1_loss(render, target)
    raytracer.backward(loss)

    fb = raytracer.cuda_module.get_framebuffer()
    fb.ray_origin
    fb.ray_direction

    R, S = build_scaling_rotation(log_scale.exp(), F.normalize(unnormalized_rotation, dim=-1)) 
    L2W = R @ S
    W2L = torch.linalg.inv(L2W)

    reference_render = torch.zeros((NUM_PIXELS, NUM_PIXELS, 3), device="cuda")
    for i in range(NUM_PIXELS):
        for j in range(NUM_PIXELS):
            ray_origin = fb.ray_origin[i, j]
            ray_direction = fb.ray_direction[i, j]
            ts = []
            for k in range(3):
                ray_origin_local = W2L[k] @ (ray_origin - mu[k])
                ray_direction_local_unnormalized = W2L[k] @ ray_direction
                norm = ray_direction_local_unnormalized.dot(ray_direction_local_unnormalized).sqrt()
                ray_direction_local = ray_direction_local_unnormalized / norm
                t_local = -torch.dot(ray_origin_local, ray_direction_local) 
                t_world = t_local / norm
                ts.append(t_world)
            sorted_indices = torch.argsort(torch.stack(ts))
            
            T = torch.tensor(1.0)
            full_T = torch.tensor(1.0)
            pixel = torch.tensor([0.0, 0.0, 0.0], device="cuda")
            remaining_color = torch.tensor([0.0, 0.0, 0.0], device="cuda")
            remaining_alpha = torch.tensor([0.0], device="cuda")
            for k in sorted_indices:
                ray_origin_local = W2L[k] @ (ray_origin - mu[k])
                ray_direction_local_unnormalized = W2L[k] @ ray_direction
                norm = ray_direction_local_unnormalized.dot(ray_direction_local_unnormalized).sqrt()
                ray_direction_local = ray_direction_local_unnormalized / norm
                t_local = -torch.dot(ray_origin_local, ray_direction_local)
                hit_local = ray_origin_local + t_local * ray_direction_local
                gaussval = torch.exp(-(hit_local.dot(hit_local)**exp_power) / (2.0 * exp_power))
                alpha = logit_opacity[k].sigmoid() * gaussval * max_alpha
                if alpha < alpha_threshold:
                    alpha = torch.tensor(0.0)
                if T > t_threshold:
                    pixel = pixel + color[k] * T * alpha
                    T = T * (1.0 - alpha)
                else:
                    alpha = alpha.detach()
                    remaining_color += color[k].detach() * alpha
                    remaining_alpha += alpha
                full_T = full_T * (1.0 - alpha)
            remaining_color_estimate = (remaining_color / remaining_alpha) if remaining_alpha > 0.0 else torch.tensor([0.0, 0.0, 0.0], device="cuda")
            remaining_T = T - full_T
            pixel = pixel + remaining_color_estimate * remaining_T
            reference_render[i, j] = pixel

    reference_render = reference_render.moveaxis(-1, 0)

    os.makedirs("tests/output", exist_ok=True)
    save_image(torch.stack([render, reference_render, (render - reference_render).abs()], dim=0), f"tests/output/mutliple_gaussians_t_threshold_{t_threshold}.png")

    assert F.l1_loss(reference_render, render) < 1e-5
    assert F.l1_loss(reference_render, render, reduction="none").amax().item() < 1e-4

    loss = F.l1_loss(reference_render, target)
    loss.backward()

    assert (mu.grad - gaussians.mean.grad).abs().amax().item() < 1e-4
    assert (unnormalized_rotation.grad - gaussians.rotation.grad).abs().amax().item() < 1e-4
    assert (log_scale.grad - gaussians.scale.grad).abs().amax().item() < 1e-4
    assert (logit_opacity.grad - gaussians.opacity.grad).abs().amax().item() < 1e-4
    assert (color.grad - gaussians.channels.grad).abs().amax().item() < 1e-4