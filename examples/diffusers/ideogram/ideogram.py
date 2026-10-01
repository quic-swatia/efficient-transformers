import os

import torch

from QEfficient import QEffIdeogram4Pipeline, QEffIdeogram4PromptEnhancerHead


prompt_enhancer_head = QEffIdeogram4PromptEnhancerHead.from_pretrained(
    "diffusers/qwen3-vl-8b-instruct-lm-head",
    dtype=torch.float32,
)

pipe = QEffIdeogram4Pipeline.from_pretrained(
    "ideogram-ai/ideogram-4-nf4-diffusers",
    prompt_enhancer_head=prompt_enhancer_head,
    allow_cpu_nf4_dequant=True,
    torch_dtype=torch.float32,
    token=os.getenv("HF_TOKEN"),
).to("cpu")

prompt = """{
  "aspect_ratio": "1:1",
  "high_level_description": "A studio photograph of a ripe red apple centered on a white ceramic plate.",
  "compositional_deconstruction": {
    "background": "A seamless pale warm-gray studio backdrop with a soft shadow beneath the plate.",
    "elements": [
      {
        "type": "obj",
        "bbox": [180, 180, 820, 820],
        "desc": "A white round ceramic plate holding one ripe red apple with a short brown stem, soft diffused daylight and a natural shadow."
      }
    ]
  }
}"""


output = pipe(
    prompt,
    height=1024,
    width=1024,
    # ###########3
    # num_inference_steps=2,
    # guidance_schedule=(7.0,) * 2,
    # ################
    prompt_upsampling=False,
    generator=torch.Generator("cpu").manual_seed(12345),
)
image = output.images[0]
image.save("ideogram4.png")
print(output)
