import streamlit as st
import google.generativeai as genai
import torch
from diffusers import StableDiffusionPipeline
import os

GOOGLE_API_KEY = "AIzaSyDamw7zNFu-cRZk0_2_S-q5viw8zXpPwbk"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# Detect CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion Model
@st.cache_resource
def load_stable_diffusion():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        if torch.__version__ >= "2.0":
            try:
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
            except Exception as e:
                st.warning(f"UNet compilation failed: {e}")
    
    return pipe

pipe = load_stable_diffusion()

# Refine description using Gemini AI
def refine_description(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(f"Refine this logo description for AI image generation: {prompt}")
    return response.text if hasattr(response, "text") else prompt

# Set Dark Theme
st.markdown(
    """
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .instruction-box {
            background-color: #1e1e1e;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(255,255,255,0.1);
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("🎨 AI Logo Generator")

user_prompt = st.text_input("Enter your logo idea:")

# Add sliders for user customization
num_steps = st.slider("🖌 Image Quality (Steps)", min_value=10, max_value=50, value=30)
guidance_scale = st.slider("🎨 Creativity Level", min_value=3.0, max_value=15.0, value=7.5)

def generate_logo(prompt):
    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        image = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale).images[0]
    
    image_path = "logo.png"
    image.save(image_path)
    return image_path

if st.button("Generate Logo"):
    if user_prompt:
        st.write("🔄 Refining prompt...")
        refined_prompt = refine_description(user_prompt)
        st.write(f"✅ Refined Prompt: {refined_prompt}")

        st.write("🖌 Generating logo, please wait...")
        logo_path = generate_logo(refined_prompt)

        if logo_path:
            st.image(logo_path, caption="Generated Logo", use_column_width=True)

            # Download Button
            with open(logo_path, "rb") as file:
                st.download_button(
                    label="📥 Download Logo",
                    data=file,
                    file_name="generated_logo.png",
                    mime="image/png"
                )
        else:
            st.error("❌ Logo generation failed.")
    else:
        st.warning("⚠ Please enter a logo idea.")

# Instructions Panel (Appears below Generate Button)
st.markdown(
    """
    <div class='instruction-box'>
        <h2>🌟 How it Works</h2>
        <p>📝 Describe the image you want</p>
        <p>🖱️ Click 'Generate Image'</p>
        <p>🎨 View your AI-generated artwork</p>
        <p>📥 Download and share!</p>
        <h3>🎨 Tips for Great Results</h3>
        <ul>
            <li>Be specific about objects and settings</li>
            <li>Mention style, genre, or mood</li>
            <li>Describe the colors and lighting</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
