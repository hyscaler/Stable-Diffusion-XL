import gradio as gr
from diffusers import DiffusionPipeline
import torch

# Load base and refiner models with specified parameters
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define processing steps
n_steps = 40
high_noise_frac = 0.8

def query(payload):
    # Generate and refine image based on the text prompt
    image = base(prompt=payload, num_inference_steps=n_steps, denoising_end=high_noise_frac, output_type="latent").images
    image = refiner(prompt=payload, num_inference_steps=n_steps, denoising_start=high_noise_frac, image=image).images[0]
    return image

# Setup Gradio interface for refined images
demo = gr.Interface(fn=query, inputs="text", outputs="image")

if __name__ == "__main__":
    demo.launch(show_api=False)