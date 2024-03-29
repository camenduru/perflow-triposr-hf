import spaces
import os, logging, time, argparse, random, tempfile, rembg, shlex, subprocess
import gradio as gr
import numpy as np
import torch
from PIL import Image
from functools import partial

subprocess.run(shlex.split('pip install wheel/torchmcubes-0.1.0-cp310-cp310-linux_x86_64.whl'))

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

from src.scheduler_perflow import PeRFlowScheduler
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def fill_background(img):
    img = np.array(img).astype(np.float32) / 255.0
    img = img[:, :, :3] * img[:, :, 3:4] + (1 - img[:, :, 3:4]) * 0.5
    img = Image.fromarray((img * 255.0).astype(np.uint8))
    return img

def merge_delta_weights_into_unet(pipe, delta_weights, org_alpha = 1.0):
    unet_weights = pipe.unet.state_dict()
    for key in delta_weights.keys():
        dtype = unet_weights[key].dtype
        try:
            unet_weights[key] = org_alpha * unet_weights[key].to(dtype=delta_weights[key].dtype) + delta_weights[key].to(device=unet_weights[key].device)
        except:
            unet_weights[key] = unet_weights[key].to(dtype=delta_weights[key].dtype)
        unet_weights[key] = unet_weights[key].to(dtype)
    pipe.unet.load_state_dict(unet_weights, strict=True)
    return pipe

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

### TripoSR
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)


### PeRFlow-T2I
# pipe_t2i = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-8", torch_dtype=torch.float16, safety_checker=None)
# pipe_t2i = StableDiffusionPipeline.from_pretrained("stablediffusionapi/disney-pixar-cartoon", torch_dtype=torch.float16, safety_checker=None)
# delta_weights = UNet2DConditionModel.from_pretrained("hansyan/piecewise-rectified-flow-delta-weights", torch_dtype=torch.float16, variant="v0-1",).state_dict()
# pipe_t2i = merge_delta_weights_into_unet(pipe_t2i, delta_weights)

pipe_t2i = StableDiffusionPipeline.from_pretrained("hansyan/perflow-sd15-disney", torch_dtype=torch.float16, safety_checker=None)
pipe_t2i.scheduler = PeRFlowScheduler.from_config(pipe_t2i.scheduler.config, prediction_type="epsilon", num_time_windows=4)
pipe_t2i.to('cuda:0', torch.float16)


### gradio
rembg_session = rembg.new_session()

@spaces.GPU
def generate(text, seed):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    setup_seed(int(seed))
    prompt_prefix = "high quality, highly detailed, (best quality, masterpiece), "
    neg_prompt = "EasyNegative, drawn by bad-artist, sketch by bad-artist-anime, (bad_prompt:0.8), (artist name, signature, watermark:1.4), (ugly:1.2), (worst quality, poor details:1.4), bad-hands-5, badhandv4, blurry"
    text = prompt_prefix + text
    samples = pipe_t2i(
            prompt              = [text],
            negative_prompt     = [neg_prompt],
            height              = 512,
            width               = 512,
            # num_inference_steps = 6,
            # guidance_scale      = 7.5,
            num_inference_steps = 8,
            guidance_scale      = 7.5,
            output_type         = 'pt',
        ).images
    samples = samples.squeeze(0).permute(1, 2, 0).cpu().numpy()*255.
    samples = samples.astype(np.uint8)
    samples = Image.fromarray(samples[:, :, :3])
    return samples


@spaces.GPU
def render(image, mc_resolution=256, formats=["obj"]):
    image = Image.fromarray(image)
    image = image.resize((768, 768))
    image = remove_background(image, rembg_session)
    image = resize_foreground(image, 0.85)
    image = fill_background(image)
    
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)
    return rv[0]


# layout
css = """
h1 {
    text-align: center;
    display:block;
}
h2 {
    text-align: center;
    display:block;
}
h3 {
    text-align: center;
    display:block;
}
"""
with gr.Blocks(title="TripoSR", css=css) as interface:
    gr.Markdown(
    """
    # Instant Text-to-3D Mesh Demo

    ### [PeRFlow](https://github.com/magic-research/piecewise-rectified-flow)-T2I  +  [TripoSR](https://github.com/VAST-AI-Research/TripoSR)
    
    Two-stage synthesis: 1) generating images by PeRFlow-T2I; 2) rendering 3D assests. Here, we plug the PeRFlow-delta-weights of SD-v1.5 into the Disney-Pixar-Cartoon dreambooth.
    """
    )
    
    with gr.Column():
        with gr.Row():
                output_image = gr.Image(label='Generated Image', height=384,)

                output_model_obj = gr.Model3D(
                    label="Output 3D Model (OBJ Format)",
                    interactive=False,
                    height=384,
            )
    
    with gr.Row():
        textbox = gr.Textbox(label="Input Prompt", value="a husky dog")
        seed = gr.Textbox(label="Random Seed", value=42)


    gr.Markdown(
    """
    Images should be generated within 1 second normally, sometimes, it could a bit slow due to warm-up of the program. Here are some examples provided:
    - a policeman
    - a robot, close-up
    - a red car, side view
    - a blue mug
    - a burger
    - a tea pot
    - a wooden chair
    - a unicorn
    """
    )
    
    # activate
    textbox.submit(
        fn=generate,
        inputs=[textbox, seed],
        outputs=[output_image],
    ).success(
        fn=render,
        inputs=[output_image],
        outputs=[output_model_obj],
    )
    
    seed.submit(
        fn=generate,
        inputs=[textbox, seed],
        outputs=[output_image],
    ).success(
        fn=render,
        inputs=[output_image],
        outputs=[output_model_obj],
    )



if __name__ == '__main__':
    interface.queue(max_size=10)
    interface.launch()
