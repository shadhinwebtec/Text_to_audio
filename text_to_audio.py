import gradio as gr 
import torch
from diffusers import AudioLDMPipeline
from share_btn import community_icon_html,loading_icon_html,share_js

from transformers import AutoProcessor, ClapModel

#make space compatible with CPU  
if torch.cuda.is_available():
    device="cuda"
    torch_dtype=torch.float16
else:
    device='cpu'
    torch_dtype=torch.float16
    
# load the diffusers pipeline 
repo_id= "cvssp/audioldm-m-full"
pipe=AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype).to(device)
pipe.unet=torch.compile(pipe.unet)



# CLAP model (only required for automatic scoring)
clap_model = ClapModel.from_pretrained("sanchit-gandhi/clap-htsat-unfused-m-full").to(device)
processor = AutoProcessor.from_pretrained("sanchit-gandhi/clap-htsat-unfused-m-full")

generator=torch.Generator(device)


def text2audio(text, negative_prompt,duration,guidance_scale,random_seed, n_candidates):
    
    if text is None:
        raise gr.Error("Please provide a text input")
    
    waveforms=pipe(text,
                   audio_length_in_s=duration,
                   guidance_scale=guidance_scale,
                   num_inference_steps=100,
                   negative_prompt=negative_prompt,
                   um_waveforms_per_prompt=n_candidates if n_candidates else 1,
                   generator=generator.manual_seed(int(random_seed)),
                   
                   )['audios']   
    if waveforms.shape[0]>1:
        waveform=score_waveforms(text,waveforms)
    else:
        waveform=waveforms[0]
    return gr.make_waveform((16000,waveform), bg_image="bg.png") 


def score_waveforms(text,waveforms):
    inputs = processor(text=text, audios=list(waveforms), return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}
    with torch.no_grad():
        logits_per_text = clap_model(**inputs).logits_per_text  # this is the audio-text similarity score
        probs = logits_per_text.softmax(dim=-1)  # we can take the softmax to get the label probabilities
        most_probable = torch.argmax(probs)  # and now select the most likely audio waveform
    waveform = waveforms[most_probable]
    return waveform
         
css = """
        a {
            color: inherit; text-decoration: underline;
        } .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        } .gr-button {
            color: white; border-color: #000000; background: #000000;
        } input[type='range'] {
            accent-color: #000000;
        } .dark input[type='range'] {
            accent-color: #dfdfdf;
        } .container {
            max-width: 730px; margin: auto; padding-top: 1.5rem;
        } #gallery {
            min-height: 22rem; margin-bottom: 15px; margin-left: auto; margin-right: auto; border-bottom-right-radius:
            .5rem !important; border-bottom-left-radius: .5rem !important;
        } #gallery>div>.h-full {
            min-height: 20rem;
        } .details:hover {
            text-decoration: underline;
        } .gr-button {
            white-space: nowrap;
        } .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity)); outline: none; box-shadow:
            var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000); --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width)
            var(--tw-ring-offset-color); --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px
            var(--tw-ring-offset-width)) var(--tw-ring-color); --tw-ring-color: rgb(191 219 254 /
            var(--tw-ring-opacity)); --tw-ring-opacity: .5;
        } #advanced-btn {
            font-size: .7rem !important; line-height: 19px; margin-top: 12px; margin-bottom: 12px; padding: 2px 8px;
            border-radius: 14px !important;
        } #advanced-options {
            margin-bottom: 20px;
        } .footer {
            margin-bottom: 45px; margin-top: 35px; text-align: center; border-bottom: 1px solid #e5e5e5;
        } .footer>p {
            font-size: .8rem; display: inline-block; padding: 0 10px; transform: translateY(10px); background: white;
        } .dark .footer {
            border-color: #303030;
        } .dark .footer>p {
            background: #0b0f19;
        } .acknowledgments h4{
            margin: 1.25em 0 .25em 0; font-weight: bold; font-size: 115%;
        } #container-advanced-btns{
            display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
        } .animate-spin {
            animation: spin 1s linear infinite;
        } @keyframes spin {
            from {
                transform: rotate(0deg);
            } to {
                transform: rotate(360deg);
            }
        } #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color:
            #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
            margin-top: 10px; margin-left: auto;
        } #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif;
            margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem
            !important;right:0;
        } #share-btn * {
            all: unset;
        } #share-btn-container div:nth-child(-n+2){
            width: auto !important; min-height: 0px !important;
        } #share-btn-container .wrap {
            display: none !important;
        } .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        } #prompt-container{
            gap: 0;
        } #generated_id{
            min-height: 700px
        } #setting_id{
          margin-bottom: 12px; text-align: center; font-weight: 900;
        }
"""
iface = gr.Blocks(css=css)

with iface:
    
    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row():
                textbox = gr.Textbox(
                    value="A hammer is hitting a wooden surface",
                    max_lines=1,
                    label="Input text",
                    info="Your text is important for the audio quality. Please ensure it is descriptive by using more adjectives.",
                    elem_id="prompt-in",
            )
                negative_textbox = gr.Textbox(
                    value="low quality, average quality",
                    max_lines=1,
                    label="Negative prompt",
                    info="Enter a negative prompt not to guide the audio generation. Selecting appropriate negative prompts can improve the audio quality significantly.",
                    elem_id="prompt-in",
                )

            with gr.Accordion("Click to modify detailed configurations", open=False):
                seed = gr.Number(
                    value=45,
                    label="Seed",
                    info="Change this value (any integer number) will lead to a different generation result.",
                )
                duration = gr.Slider(2.5, 10, value=5, step=2.5, label="Duration (seconds)")
                guidance_scale = gr.Slider(
                    0,
                    5,
                    value=3.5,
                    step=0.5,
                    label="Guidance scale",
                    info="Large => better quality and relevancy to text; Small => better diversity",
                )
                n_candidates = gr.Slider(
                    1,
                    3,
                    value=3,
                    step=1,
                    label="Number waveforms to generate",
                    info="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation",
                )

            outputs = gr.Video(label="Output", elem_id="output-video")
            btn = gr.Button("Submit").style(full_width=True)

            with gr.Group(elem_id="share-btn-container", visible=False):
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn")

            btn.click(
                text2audio,
                inputs=[textbox, negative_textbox, duration, guidance_scale, seed, n_candidates],
                outputs=[outputs],
            )

            share_button.click(None, [], [], _js=share_js)
            gr.Examples(
                [
                    ["A hammer is hitting a wooden surface", "low quality, average quality", 5, 2.5, 45, 3],
                    ["Peaceful and calming ambient music with singing bowl and other instruments.", "low quality, average quality", 5, 2.5, 45, 3],
                    ["A man is speaking in a small room.", "low quality, average quality", 5, 2.5, 45, 3],
                    ["A female is speaking followed by footstep sound", "low quality, average quality", 5, 2.5, 45, 3],
                    ["Wooden table tapping sound followed by water pouring sound.", "low quality, average quality", 5, 2.5, 45, 3],
                ],
                fn=text2audio,
                inputs=[textbox, negative_textbox, duration, guidance_scale, seed, n_candidates],
                outputs=[outputs],
                cache_examples=True,
            )
            
        

iface = demo.queue(max_size=10).launch(debug=True)
