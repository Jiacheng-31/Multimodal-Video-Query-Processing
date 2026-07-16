# Model placeholders

No model weights are distributed with this repository. Run `scripts/download_models.sh` or place compatible checkpoints at:

- `models/Qwen3-VL-2B-Instruct`: router actor.
- `models/clip-vit-large-patch14`: context and entity encoder.
- `models/blip2-opt-2.7b`: optional offline frame caption generator.
- `models/sam2-hiera-large`: optional offline entity proposal generator.

The ERNIE scorer is accessed through the Qianfan API and does not require a local checkpoint.
