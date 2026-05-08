import os

# * This setting is important, without it trainings would sometimes freeze mysteriously
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
