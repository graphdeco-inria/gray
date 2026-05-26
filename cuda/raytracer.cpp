#include "raytracer.h"

TORCH_LIBRARY(gray, m) {
    CameraDataHolder::bind(m);
    ConfigDataHolder::bind(m);
    FramebufferDataHolder::bind(m);
    GaussianDataHolder::bind(m);
    MetaDataHolder::bind(m);
    StatsDataHolder::bind(m);
    PPLLDataHolder::bind(m);

    Raytracer::bind(m);
}
