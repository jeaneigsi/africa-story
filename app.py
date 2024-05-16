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

llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=api_key)
model=llm

system="All instructions must be follow is very important, all story related to african culture and history is mandatory.You are a storyteller who specializes in creating educational tales about African culture. Your mission is to craft a narrative that teaches African children about their rich heritage. Your story is based on real events from the past, incorporating historical references, myths, and legends. story size is short length. Your narrative will be presented in six panels.Very important, For each panel, you will provide: A description of the characters, using precise and unique descriptions each time, ending with the keywords 'high quality', 'watercolor painting', 'painting Benin style', and 'mugshot', 'cartoon africa style' in the scenes or characters is mandatory.For description, using only words or groups of words separated by commas, without sentences. Each sentence in the panel's text should start with the character's name, and each sentence should be no longer than two small sentences. Each story has only three characters. Your story must always revolve around African legends and kingdoms, splitting the scenario into six parts. Be creative in each story"

st.title("Storytelling with AI")
# Create input zone
title = st.text_input("Discover a new story on africa, tape a topic !")

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

import streamlit as st
import requests
import io
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}

# Fonction pour appeler l'API et générer une image pour une scène donnée
def generate_image(scene):
    payload = {
        "inputs": scene,
        "guidance_scale": 0.8,
        "num_inference_steps": 8,
        "eta": 0.5,
        "seed": 46,
        "negative_prompt": negative_prompt
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes))
    return image

# Contenu de la variable response
scenes =response.scenes

metadonne =response.metadonne
# Générer les images pour chaque scène et afficher avec les métadonnées dans une grille 2x3
st.title("Images générées avec métadonnées dans une grille 2x3")
for i in range(0, len(scenes), 2):
    col1, col2 = st.columns(2)
    col1.write(f"**Scène {i+1}:** {metadonne[i]}")
    col1.image(generate_image(scenes[i]), caption=f"Image de la scène {i+1}", width=300)
    
    # Vérifie si une deuxième scène existe pour afficher la deuxième image
    if i+1 < len(scenes):
        col2.write(f"**Scène {i+2}:** {metadonne[i+1]}")
        col2.image(generate_image(scenes[i+1]), caption=f"Image de la scène {i+2}", width=300)













negative_prompt="bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, realistic photo, extra eyes, huge eyes, 2girl, amputation, disconnected limbs."
prompt = "Black backgroung aquarelle, painting benin  style, mugshot young girl "

import streamlit as st
import requests
import io
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

image_bytes = query({
    "inputs": "Kwame, painting benin style mugshot, is training hard with his spear, determined to become a brave warrior.",
    "guidance_scale": 0.8,
    "num_inference_steps": 8,
    "eta": 0.5,
    "seed": 46,
    "negative_prompt": negative_prompt
})

# Convert image bytes to PIL Image
image = Image.open(io.BytesIO(image_bytes))

# Display the image in a 2x2 grid
st.image(image, caption='Generated Image', width=300)

# You can repeat the above lines to display the same image multiple times in the grid
# Or you can display different images in the grid as per your requirement









