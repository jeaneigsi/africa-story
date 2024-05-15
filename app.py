import os
import mediapy as media
import random
import sys
import torch
import matplotlib.pyplot as plt
from rembg import remove
from dotenv import load_dotenv

from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from getpass import getpass
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
from PIL import Image

# set hf inference endpoint with lama for story
# get a token: https://huggingface.co/docs/api-inference/quicktour#get-your-api-token


load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class Story(BaseModel):
    title: str = Field(description="A captivating title for the story.")
    characters: list[str] = Field(
        description="""Important:no json format. Six elements mandatory, each formatted as: 
                       "[Character Name], [comma-separated adjectives], cartoon, style africa, painting".
                       Describe each character's appearance in detail. Be creative!"""
    )
    scenes: list[str] = Field(
        description="""Important:no json format.no json format. Six elements mandatory, each a string describing a character's action.very important:use charaters description apperance, use only action verbs.
                      Each scene must follow the previous one chronologically, creating a complete narrative when combined.
                      Develop your story by detailing what each character DOES in each scene.Instead to use only name of characters to write this part, use name and this key word 'painting bening style mushgot 'as description appearence, it's very import to do loke that.if it's a new characters in the story, instead use his name use his name and add the keyword 'painting benin style'it is mandatory.Use your imagination!"""
    )
    metadonne: list[str] = Field(
        description="""Important: no json format.Six elements mandatory, each a concise one-sentence description of the corresponding scene in the 'scenes' field.
                      Explain the action taking place in each scene. Come up with your own unique descriptions!"""
    )

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(model="gemini-pro")
model=llm

system="All instructions must be follow is very important, all story related to african culture and history is mandatory.You are a storyteller who specializes in creating educational tales about African culture. Your mission is to craft a narrative that teaches African children about their rich heritage. Your story is based on real events from the past, incorporating historical references, myths, and legends. story size is short length. Your narrative will be presented in six panels.Very important, For each panel, you will provide: A description of the characters, using precise and unique descriptions each time, ending with the keywords 'high quality', 'watercolor painting', 'painting Benin style', and 'mugshot', 'cartoon africa style' in the scenes or characters is mandatory.For description, using only words or groups of words separated by commas, without sentences. Each sentence in the panel's text should start with the character's name, and each sentence should be no longer than two small sentences. Each story has only three characters. Your story must always revolve around African legends and kingdoms, splitting the scenario into six parts. Be creative in each story"

st.title("Storytelling with AI")
# Create input zone
title = st.text_input("Enter une histoire que vous souhaitez creer ")








story_query=system+title
parser = PydanticOutputParser(pydantic_object=Story)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": story_query})

response =chain.invoke({"query": story_query})
response

# modele load
# Choose among 1, 2, 4 and 8:
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


negative_prompt="bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs."
prompt = "Black backgroung aquarelle, painting benin  style, mugshot young girl "


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


import random
negative_prompt="bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, photo ,realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs."
seed=6000000
eta=0.3
prompt = response.scenes[:3]
all_images=[]
all_images.clear()

for i in range(1):
  images = pipe(
    prompt=prompt,
    num_inference_steps = num_inference_steps,
    guidance_scale = 1.2,
    eta = eta,
    negative_prompt=negative_prompt,
    generator = torch.Generator(device).manual_seed(seed)).images
  all_images.extend(images)

import random
prompt = response.scenes[3:]

for i in range(1):
  images = pipe(
    prompt=prompt,
    num_inference_steps = num_inference_steps,
    guidance_scale = 1.2,
    eta = eta,
    negative_prompt=negative_prompt,
    generator = torch.Generator(device).manual_seed(seed)).images
  all_images.extend(images)

num_rows=3
num_cols=2


# Audio file
audio_file = "audio.mp3"  # Update with your own audio filename

# Function to display image grid
def display_image_grid(images):
    col_count = 3  # Number of columns in the grid
    for i in range(0, len(images), col_count):
        col_imgs = images[i:i + col_count]
        cols = st.columns(col_count)
        for col, img_path in zip(cols, col_imgs):
            col.image(Image.open(img_path), use_column_width=True)

# Main Streamlit app
st.title("Cartoon Image Viewer")

    # Display the image grid
display_image_grid(all_images)

    # Buttons for sliding images and playing audio
col1, col2 = st.columns(2)
if col1.button("Previous Image"):
    st.experimental_rerun()
if col2.button("Next Image"):
    st.experimental_rerun()
if st.button("Play Audio"):
     st.audio(audio_file) 

