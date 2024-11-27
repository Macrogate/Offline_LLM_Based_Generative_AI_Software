import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering

st.set_page_config(page_title="BharatGPT",layout="wide" )
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_answer(image, text):
    try:
        img = Image.open(BytesIO(image)).convert("RGB")

        encoding = processor(img, text, return_tensors="pt")

        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]

        return answer

    except Exception as e:
        return str(e)


st.title("Image Summerizer:")
st.write("Upload an image and enter a question to get an answer.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    st.image(uploaded_file, use_column_width=True)

with col2:
    question = st.text_input("Question")
    if uploaded_file and question is not None:
        if st.button("Ask Question"):
            image = Image.open(uploaded_file)
            image_byte_array = BytesIO()
            image.save(image_byte_array, format='JPEG')
            image_bytes = image_byte_array.getvalue()         
            answer = get_answer(image_bytes, question)
            st.success("Answer: " + answer)
            
