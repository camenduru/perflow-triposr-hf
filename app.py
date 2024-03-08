import os, logging, time, argparse, random, tempfile, rembg
import gradio as gr
import numpy as np
import torch
from PIL import Image
from functools import partial
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

from src.scheduler_perflow import PeRFlowScheduler
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

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
pipe_t2i = StableDiffusionPipeline.from_pretrained("stablediffusionapi/disney-pixar-cartoon", torch_dtype=torch.float16, safety_checker=None)
delta_weights = UNet2DConditionModel.from_pretrained("hansyan/piecewise-rectified-flow-delta-weights", torch_dtype=torch.float16, variant="v0-1",).state_dict()
pipe_t2i = merge_delta_weights_into_unet(pipe_t2i, delta_weights)
pipe_t2i.scheduler = PeRFlowScheduler.from_config(pipe_t2i.scheduler.config, prediction_type="epsilon", num_time_windows=4)
pipe_t2i.to('cuda:0', torch.float16)


### gradio
rembg_session = rembg.new_session()

def generate(text, seed):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    setup_seed(int(seed))
    # text = text
    samples = pipe_t2i(
            prompt              = [text],
            negative_prompt     = ["distorted, blur, low-quality, haze, out of focus"],
            height              = 512,
            width               = 512,
            # num_inference_steps = 4,
            # guidance_scale      = 4.5,
            num_inference_steps = 6,
            guidance_scale      = 7,
            output_type         = 'pt',
        ).images
    samples = torch.nn.functional.interpolate(samples, size=768, mode='bilinear')
    samples = samples.squeeze(0).permute(1, 2, 0).cpu().numpy()*255.
    samples = samples.astype(np.uint8)
    samples = Image.fromarray(samples[:, :, :3])

    image = remove_background(samples, rembg_session)
    image = resize_foreground(image, 0.85)
    image = fill_background(image)
    return image

def render(image, mc_resolution=256, formats=["obj"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)
    return rv[0]

# warm up
_ = generate("a bird", 42)

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
    
    Two-stage synthesis: 1) generating images by PeRFlow-T2I with 6-step inference; 2) rendering 3D assests.
    """
    )
    
    with gr.Column():
        with gr.Row():
                output_image = gr.Image(label='Generated Image', height=384, width=384)

                output_model_obj = gr.Model3D(
                    label="Output 3D Model (OBJ Format)",
                    interactive=False,
                    height=384, width=384,
            )
    
    with gr.Row():
        textbox = gr.Textbox(label="Input Prompt", value="a colorful bird")
        seed = gr.Textbox(label="Random Seed", value=42)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None, help='Username for authentication')
    parser.add_argument('--password', type=str, default=None, help='Password for authentication')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
    parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
    parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    args = parser.parse_args()
    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(args.username, args.password) if (args.username and args.password) else None,
        share=args.share,
        server_name="0.0.0.0" if args.listen else None, 
        server_port=args.port
    )
