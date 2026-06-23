import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. إعدادات الصفحة الخاصة بـ Streamlit
st.set_page_config(page_title="مستكشف القطط الذكي", page_icon="🐱")
st.title("مستكشف القطط الذكي 🧠🐱")
st.write("قم برفع أي صورة، وسيقوم النموذج بتحليل مصفوفاتها والتنبؤ بالنتيجة فوراً.")

# 2. تحميل ملف الأوزان تلقائياً من الـ Release إذا لم يكن موجوداً
MODEL_PATH = 'cat_model.pth'
if not os.path.exists(MODEL_PATH):
    url = "https://github.com/jalalalmahdi3-stack/cat-detector/releases/download/v1.0/cat_model.pth"
    with st.spinner("جاري تحميل ملف الأوزان من GitHub Releases لأول مرة... يرجى الانتظار"):
        urllib.request.urlretrieve(url, MODEL_PATH)

# 3. بناء هيكل النموذج وتحميل الأوزان (نحتفظ بهذا الكود داخل الذاكرة لكي لا يعيد التحميل مع كل صورة)
@st.cache_resource
def load_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 4. واجهة المستخدم لرفع الصورة (Streamlit Native Uploader)
uploaded_file = st.file_uploader("اختر صورة من جهازك...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة للمستخدم
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    # زر بدء التحليل والتنبؤ
    if st.button("حلل الصورة الآن 🚀"):
        with st.spinner("جاري معالجة البكسلات وتمريرها عبر الطبقات..."):
            # معالجة وتجهيز الصورة للنموذج
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            img_tensor = transform(image).unsqueeze(0)
            
            # التنبؤ بالنتيجة
            with torch.no_grad():
                outputs = model(img_tensor)
                _, preds = torch.max(outputs, 1)
            
            # عرض النتيجة النهائية بناءً على مخرج النموذج
            if preds.item() == 0:
                st.success("النتيجة: هذه قطة بالتأكيد! 🐱🎉")
            else:
                st.error("النتيجة: ليست قطة! ❌")
