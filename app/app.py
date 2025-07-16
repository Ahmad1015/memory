import streamlit as st
from PIL import Image
from detector import detect_objects
from captioner import generate_caption  # 🆕 BLIP captioning

st.set_page_config(page_title="Visual Memory Assistant", layout="wide")
st.title("🧠 Visual Memory Assistant (v1)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Original Image", use_column_width=True)

    image = Image.open(uploaded_file)

    with st.spinner("🔍 Detecting objects..."):
        annotated_img, objects = detect_objects(image)

    with st.spinner("📝 Generating caption..."):
        caption = generate_caption(image)

    # Show detection results
    st.subheader("🔍 Detected Objects")
    st.image(annotated_img, caption="Detected", use_column_width=True)

    st.write("Objects Detected:")
    for obj in objects:
        st.markdown(f"- **{obj['label']}** ({obj['confidence']:.2f})")

    # Show caption
    st.subheader("📝 Scene Caption")
    st.success(caption)
