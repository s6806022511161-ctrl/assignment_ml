from pathlib import Path
import gc

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

from src.orchid_data import CLASSES, MODEL_LABELS, ORCHID_DETAILS


ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "resnet50": ROOT / "models" / "resnet50_best.pth",
    "efficientnet_b0": ROOT / "models" / "efficientnet_b0_best.pth",
    "vit_base_patch16_224": ROOT / "models" / "vit_base_patch16_224_best.pth",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)

st.set_page_config(page_title="Thai Orchid AI", page_icon="🌸", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #fff9fd 0%, #f7fbf6 100%); }
.hero { padding: 1.3rem 1.5rem; border-radius: 22px; color: white; background: linear-gradient(120deg, #6a1b9a, #ad477f 55%, #517a4f); margin-bottom: 1rem; }
.hero h1 { margin: 0; font-size: clamp(1.8rem, 4vw, 3rem); }
.hero p { margin: .45rem 0 0; opacity: .94; }
.result-card { padding: 1.2rem 1.35rem; border: 1px solid #ead8e8; border-radius: 18px; background: rgba(255,255,255,.94); box-shadow: 0 8px 28px rgba(74,20,80,.08); }
.confidence { font-size: 1.55rem; font-weight: 750; color: #6a1b9a; }
.small-note { color: #626262; font-size: .92rem; }
div[data-testid="stMetric"] { background: white; border: 1px solid #eee; padding: .8rem; border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(model_name: str) -> nn.Module:
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    elif model_name == "vit_base_patch16_224":
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASSES))
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


@st.cache_resource(max_entries=1, show_spinner=False)
def load_model(model_name: str):
    weight_path = MODEL_FILES[model_name]
    if not weight_path.is_file():
        raise FileNotFoundError(f"ไม่พบ weights: {weight_path.name}")
    model = build_model(model_name)
    state = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    del state
    model.to(DEVICE).eval()
    gc.collect()
    return model


def read_image(file) -> Image.Image:
    image = Image.open(file)
    return ImageOps.exif_transpose(image).convert("RGB")


def predict(model: nn.Module, image: Image.Image):
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu().numpy()
    index = int(np.argmax(probabilities))
    return CLASSES[index], probabilities


st.markdown("""
<div class="hero"><h1>🌸 Thai Orchid AI</h1>
<p>จำแนกสกุลกล้วยไม้จากภาพถ่าย พร้อมข้อมูลลักษณะเด่นและแนวทางดูแลเบื้องต้น</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("ตั้งค่าการวิเคราะห์")
    selected_model = st.selectbox("เลือกโมเดล", list(MODEL_LABELS), format_func=lambda key: MODEL_LABELS[key], index=1)
    st.caption("EfficientNet-B0 เหมาะเป็นค่าเริ่มต้น เพราะมีขนาดเล็กและทำคะแนน Test สูงสุดเล็กน้อยในรอบล่าสุด")
    st.divider()
    st.markdown("**ขอบเขตของระบบ**")
    st.caption("ระบบจำแนกได้ 5 สกุล: Cattleya, Dendrobium, Oncidium, Phalaenopsis และ Vanda ผลลัพธ์ไม่ใช่การยืนยันชนิดย่อยหรือสายพันธุ์ (cultivar)")
    st.caption(f"อุปกรณ์ประมวลผล: {DEVICE.type.upper()}")

upload_tab, camera_tab = st.tabs(["📁 อัปโหลดรูปภาพ", "📷 ถ่ายภาพ"])
with upload_tab:
    uploaded = st.file_uploader("เลือกรูปกล้วยไม้", type=["jpg", "jpeg", "png", "webp"], key="orchid_upload")
with camera_tab:
    captured = st.camera_input("ถ่ายให้ดอกอยู่กลางภาพและมีแสงเพียงพอ", key="orchid_camera")

source = captured if captured is not None else uploaded
if source is None:
    st.info("อัปโหลดรูปหรือถ่ายภาพเพื่อเริ่มวิเคราะห์")
    st.markdown("**เคล็ดลับ:** ใช้ภาพดอกที่ชัด เห็นรูปทรงกลีบและกลีบปาก หลีกเลี่ยงภาพมืด ภาพไกล หรือภาพที่มีดอกหลายชนิดปะปนกัน")
    st.stop()

try:
    input_image = read_image(source)
except Exception:
    st.error("ไม่สามารถอ่านไฟล์ภาพนี้ได้ กรุณาลองไฟล์ JPG, PNG หรือ WEBP อื่น")
    st.stop()

try:
    with st.spinner(f"กำลังโหลด {MODEL_LABELS[selected_model]} และวิเคราะห์ภาพ..."):
        orchid_model = load_model(selected_model)
        predicted_class, probabilities = predict(orchid_model, input_image)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("ตรวจสอบว่าไฟล์ weights ทั้งสามอยู่ในโฟลเดอร์ models/")
    st.stop()
except Exception as exc:
    st.error("เกิดข้อผิดพลาดระหว่างโหลดโมเดลหรือทำนายภาพ")
    st.exception(exc)
    st.stop()

confidence = float(probabilities.max())
detail = ORCHID_DETAILS[predicted_class]
image_col, result_col = st.columns([1, 1.15], gap="large")
with image_col:
    st.image(input_image, caption="ภาพที่นำมาวิเคราะห์", width="stretch")
with result_col:
    st.markdown(f"""
    <div class="result-card"><div class="small-note">ผลจำแนกที่มีความน่าจะเป็นสูงสุด</div>
    <h2>{detail['icon']} {detail['thai_name']}</h2><div class="confidence">{confidence * 100:.2f}%</div>
    <div class="small-note">โมเดล: {MODEL_LABELS[selected_model]}</div></div>
    """, unsafe_allow_html=True)
    if confidence < 0.60:
        st.warning("โมเดลยังไม่มั่นใจ ควรถ่ายใหม่ให้ดอกชัดขึ้นและลองเปรียบเทียบกับโมเดลอื่น")
    elif confidence < 0.80:
        st.info("ผลอยู่ในระดับความมั่นใจปานกลาง แนะนำให้ตรวจลักษณะดอกเพิ่มเติม")

st.subheader("ความน่าจะเป็นของแต่ละสกุล")
chart_data = [
    {"สกุล": ORCHID_DETAILS[name]["short_name"], "ความน่าจะเป็น (%)": float(probabilities[i] * 100)}
    for i, name in enumerate(CLASSES)
]
st.bar_chart(chart_data, x="สกุล", y="ความน่าจะเป็น (%)", horizontal=True)

about_tab, care_tab, caution_tab = st.tabs(["🌿 ลักษณะและการแนะนำ", "💧 วิธีดูแล", "ℹ️ ข้อควรรู้"])
with about_tab:
    st.markdown(f"### {detail['thai_name']}")
    st.write(detail["description"])
    st.markdown(f"**จุดสังเกต:** {detail['identification']}")
    st.markdown(f"**เหมาะสำหรับ:** {detail['recommendation']}")
with care_tab:
    care = detail["care"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**☀️ แสง:** {care['light']}")
        st.markdown(f"**💧 น้ำ:** {care['water']}")
        st.markdown(f"**🌡️ อุณหภูมิ:** {care['temperature']}")
    with c2:
        st.markdown(f"**🌬️ อากาศและความชื้น:** {care['humidity']}")
        st.markdown(f"**🪴 วัสดุปลูก:** {care['medium']}")
        st.markdown(f"**🌱 ปุ๋ย:** {care['fertilizer']}")
with caution_tab:
    st.write("คำแนะนำเป็นแนวทางทั่วไปของสกุล การให้น้ำและแสงควรปรับตามพันธุ์ย่อย ฤดูกาล วัสดุปลูก และสภาพอากาศจริง หากใบไหม้ รากเน่า หรือมีศัตรูพืช ควรแยกต้นและปรึกษาผู้เชี่ยวชาญ")
    st.caption("ผล Test ล่าสุด: ResNet-50 73.72%, EfficientNet-B0 73.88% และ ViT-B/16 49.72% จากภาพ Test 2,500 ภาพ")

st.divider()
st.caption("โครงงานจำแนกสกุลกล้วยไม้ไทยด้วย Deep Learning · ใช้เพื่อการศึกษาและช่วยคัดกรองเบื้องต้น")
