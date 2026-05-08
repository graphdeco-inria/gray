
#include "../core/gaussians.h"

void adam_step(Gaussians gaussians, int step, bool zero_grads, bool update_channels, float beta_1, float beta_2,
               float epsilon, bool enable_sh, int sh_update_laziness);