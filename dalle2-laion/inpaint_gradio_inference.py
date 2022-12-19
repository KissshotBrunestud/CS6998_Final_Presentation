try:
    import gradio as gr
except ImportError:
    print("Please install gradio: `pip install gradio`")
    exit(1)
from pathlib import Path
from typing import Dict, List
from PIL import Image as PILImage
from dalle2_laion import ModelLoadConfig, DalleModelManager, utils
from dalle2_laion.scripts import BasicInference, ImageVariation, BasicInpainting

config_path = Path(__file__).parent / 'configs/gradio.example.json'
model_config = ModelLoadConfig.from_json_path(config_path)
model_manager = DalleModelManager(model_config)

output_path = Path(__file__).parent / 'output/gradio'
output_path.mkdir(parents=True, exist_ok=True)

cond_scale_sliders = [gr.Slider(minimum=0.5, maximum=5, step=0.05, label="Prior Cond Scale", value=1),]
for i in range(model_manager.model_config.decoder.final_unet_number):
    cond_scale_sliders.append(gr.Slider(minimum=0.5, maximum=5, step=0.05, label=f"Decoder {i+1} Cond Scale", value=1))


def inpaint(image: Dict[str, PILImage.Image], text: str, num_generations: int, prior_cond_scale: float, *decoder_cond_scales: List[float]):
    print("Inpainting using text:", text)
    img, mask = image['image'], image['mask']
    # Remove alpha from img
    img = img.convert('RGB')
    img = utils.center_crop_to_square(img)
    mask = utils.center_crop_to_square(mask)

    script = BasicInpainting(model_manager, verbose=True)
    mask = ~utils.get_mask_from_image(mask)
    output = script.run(images=[img], masks=[mask], text=[text], sample_count=num_generations, prior_cond_scale=prior_cond_scale, decoder_cond_scale=decoder_cond_scales)
    all_outputs = []
    for index, embedding_output in output.items():
        all_outputs.extend(embedding_output)
    return all_outputs
inpaint_interface = gr.Interface(
    inpaint,
    inputs=[
        gr.Image(value="https://www.thefarmersdog.com/digest/wp-content/uploads/2021/12/corgi-top-1400x871.jpg", source="upload", tool="sketch", interactive=True, type="pil"),
        gr.Text(),
        gr.Slider(minimum=1, maximum=6, label="Number to generate", value=2, step=1),
        *cond_scale_sliders
    ],
    outputs=[
        gr.Gallery()
    ],
    title="Dalle2 Inpainting",
    description="Fills in the details of areas you mask out.\nGeneration takes around 5 minutes so be patient.",
)
#demo = gr.TabbedInterface(interface_list=[dream_interface, variation_interface, inpaint_interface], tab_names=["Dream", "Variation", "Inpaint"])
demo = gr.TabbedInterface(interface_list=[inpaint_interface], tab_names=["Inpaint"])
#demo = gr.TabbedInterface(interface_list=[dream_interface, variation_interface], tab_names=["Dream", "Variation"])
#demo = gr.TabbedInterface(interface_list=[dream_interface, inpaint_interface], tab_names=["Dream", "inpaint_interface"])

demo.launch(share=True, enable_queue=True)