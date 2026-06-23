import os
import urllib.request
import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. تحديد مسار الملف وآلية التحميل التلقائي من رابط الـ Release الخاص بك
MODEL_PATH = 'cat_model.pth'
if not os.path.exists(MODEL_PATH):
    url = "https://github.com/jalalalmahdi3-stack/cat-detector/releases/download/v1.0/cat_model.pth"
    print("جاري تحميل ملف الأوزان من GitHub Releases... يرجى الانتظار")
    urllib.request.urlretrieve(url, MODEL_PATH)

# 2. إعادة بناء بنية نموذج ResNet18 وتعديل مخرجه ليطابق تدريبنا
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# تحميل الأوزان الرقمية إلى الهيكل المستدعى على معالج الـ CPU ليتوافق مع السيرفر
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# 3. دالة معالجة الصورة والتنبؤ بها
def predict(img):
    # تحويل الصورة لتطابق الأبعاد والمعايرة التي تدرب عليها النموذج
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # تحويل الصورة إلى مصفوفة وإضافة بعد الدفعات (Batch)
    img_tensor = transform(img).unsqueeze(0)
    
    # التنبؤ بدون حساب اشتقاقات لتوفير ذاكرة السيرفر
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        
    # الترتيب الأبجدي للمجلدات جعل cat يعبر عن الرقم 0 و not_cat يعبر عن الرقم 1
    if preds.item() == 0:
        return "هذه قطة بالتأكيد! 🐱"
    else:
        return "ليست قطة! ❌"

# 4. بناء واجهة مستخدم Gradio متوافقة مع إعدادات سيرفر Streamlit
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="مستكشف القطط الذكي 🧠",
    description="قم برفع أو التقاط أي صورة، وسيقوم النموذج بتحليل مصفوفاتها والتنبؤ بالنتيجة فوراً."
)

if __name__ == "__main__":
    # تشغيل التطبيق على المنافذ البرمجية المخصصة للاستضافة السحابية لـ Streamlit
    interface.launch(server_name="0.0.0.0", server_port=8501)
