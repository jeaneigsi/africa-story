import os
from typing import List
from getpass import getpass
from PIL import Image
import streamlit as st
import torch
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download

# Set up Hugging Face API token
HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Define Story model
class Story(BaseModel):
    title: str = Field(description="A captivating title for the story.")
    characters: List[str] = Field(
        description="""Important: No JSON format. Six elements mandatory, each formatted as: 
                       "[Character Name], [comma-separated adjectives], cartoon, style Africa, painting".
                       Describe each character's appearance in detail. Be creative!"""
    )
    scenes: List[str] = Field(
        description="""Important: No JSON format. Six elements mandatory, each a string describing a character's action. Very important: use characters' description appearance, use only action verbs.
                      Each scene must follow the previous one chronologically, creating a complete narrative when combined.
                      Develop your story by detailing what each character DOES in each scene. Instead of using only names of characters to write this part, use name and this keyword 'painting benign style mushgot' as description appearance, it's very important to do like that. If it's a new character in the story, instead use his name add the keyword 'painting Benin style', it is mandatory. Use your imagination!"""
    )
    metadonne: List[str] = Field(
        description="""Important: No JSON format. Six elements mandatory, each a concise one-sentence description of the corresponding scene in the 'scenes' field.
                      Explain the action taking place in each scene. Come up with your own unique descriptions!"""
    )

# Set up Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-pro")
model = llm

# Streamlit UI
st.title("Storytelling with AI")
title = st.text_input("Enter the story you want to create")

# Prompt and model setup
story_query = system + title  # Assuming `system` is defined somewhere
parser = PydanticOutputParser(pydantic_object=Story)
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | model | parser
response = chain.invoke({"query": story_query})

# Load diffusion model
num_inference_steps = 8
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
plural = "s" if num_inference_steps > 1 else ""
ckpt_name = f"Hyper-SDXL-{num_inference_steps}step{plural}-lora.safetensors"
device = "cuda"
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
pipe.enable_sequential_cpu_offload()
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

# Generate images
negative_prompt = "bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs."
prompt = "Black background aquarelle, painting Benin style, mugshot young girl "
eta = 0.3
all_images = []

for prompt in [response.scenes[:3], response.scenes[3:]]:
    images = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.2,
        eta=eta,
        negative_prompt=negative_prompt,
        generator=torch.Generator(device).manual_seed(seed)).images
    all_images.extend(images)

# Display images and audio
st.title("Cartoon Image Viewer")
col_count = 3  # Number of columns in the grid

for i in range(0, len(all_images), col_count):
    col_imgs = all_images[i:i + col_count]
    cols = st.columns(col_count)
    for col, img in zip(cols, col_imgs):
        col.image(img, use_column_width=True)

# Buttons for sliding images and playing audio
col1, col2 = st.columns(2)
if col1.button("Previous Image"):
    st.experimental_rerun()
if col2.button("Next Image"):
    st.experimental_rerun()
if st.button("Play Audio"):
    audio_file = "audio.mp3"  # Update with your own audio filename
    st.audio(audio_file) 
