import torch 

def build_rotation(q):
    R = torch.zeros((q.size(0), 3, 3), device=q.device, dtype=q.dtype)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, q):
    S = torch.zeros((s.shape[0], 3, 3), device=s.device, dtype=s.dtype)
    R = build_rotation(q)
    R.retain_grad()

    S[:,0,0] = s[:,0]
    S[:,1,1] = s[:,1]
    S[:,2,2] = s[:,2]

    return R, S
