import streamlit as st
from SemanticGuidedHumanMatting.getMatting import Matting, MattingSingleImage
from harmonizer.harmonize import get_harmonized, compose_images, placeImage
from rembg import remove
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Automatic Image Blending")
st.write(
    "Upload a foreground and background image to see it automatically blended and combined"
)
st.sidebar.write("## Upload and download :gear:")


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = remove(image)
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")

def give_transparent(upload):
    image = Image.open(upload)
    fixed = remove(image)
    return fixed

def get_transparent2(upload, mask):
    image = Image.open(upload)
    img_org  = image.convert('RGB') # or 'RGBA'
    img_mask = mask.convert('L')    # grayscale

    # the same size
    img_org  = img_org.resize((400,400))
    img_mask = img_mask.resize((400,400))

    # add alpha channel    
    img_org.putalpha(img_mask)
    return img_org

def pipeline(foreground_upload, background_upload):
    col1.write("Foreground Image")
    col1.image(foreground_upload)

    col2.write("Background Image")
    col2.image(background_upload)

    mask = MattingSingleImage(upload=foreground_upload)
    col3.write("Matting background")
    col3.image(mask)

    transparent = get_transparent2(foreground_upload, mask)
    col4.write("Foreground image seperated")
    col4.image(transparent)

    # composite = placeImage(foreground_upload, mask, background_upload)
    # composite, hard_mask, bbox = compose_images(transparent, background_upload)
    composite = Image.open("composite.jpg")
    col5.write("Composite Image")
    col5.image(composite)

    # final = get_harmonized(composite, )
    final = Image.open("final.jpg")
    col6.write("Final Harmonized Image")
    col6.image(final)






col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
my_upload_foreground = st.sidebar.file_uploader("Upload forground image", type=["png", "jpg", "jpeg"])
my_upload_background = st.sidebar.file_uploader("Upload background image", type=["png", "jpg", "jpeg"])


if my_upload_foreground is not None and my_upload_background is not None:
    pipeline(my_upload_foreground, my_upload_background)
else:
    # Image.open("foreground_img1.jpg")
    pipeline("SemanticGuidedHumanMatting/foreground_img1.jpg", "SemanticGuidedHumanMatting/background_img1.jpg")