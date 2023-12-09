import gradio as gr
from diffusers import StableDiffusionXLPipeline
import torch

# Load the model with specified parameters
base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, 
)
base.to("cuda")

def query(payload):
    # Generate image based on the text prompt
    image = base(prompt=payload).images[0]
    return image

# Setup Gradio interface
demo = gr.Interface(fn=query, inputs="text", outputs="image")

if __name__ == "__main__":
    demo.launch(show_api=False)