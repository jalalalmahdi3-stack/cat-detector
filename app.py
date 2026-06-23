import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. إعادة بناء هيكل النموذج (نفس الهيكل الذي تدرب في كولاب)
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 2. تحميل الأوزان الرقمية المستخرجة إلى الهيكل
model.load_state_dict(torch.load('cat_model.pth', map_location=torch.device('cpu')))
model.eval() # وضع النموذج في طور التنبؤ فقط

# 3. دالة معالجة الصورة والتنبؤ بها
def predict(img):
    # تحويل حجم الصورة ومعايرتها لتطابق مدخلات النموذج
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # تحويل الصورة إلى مصفوفة وإضافة بُعد إضافي للدفعات (Batch Dimension)
    img_tensor = transform(img).unsqueeze(0)
    
    # التنبؤ بالنتيجة بدون حساب اشتقاقات
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        
    # تحديد النص المعروض للمستخدم بناءً على المخرج الرقمي (0 أو 1)
    # ملاحظة: الترتيب يعتمد على ترتيب المجلدات أبجدياً (cat=0, not_cat=1)
    if preds.item() == 0:
        return "هذه قطة بالتأكيد! 🐱"
    else:
        return "ليست قطة! ❌"

# 4. بناء واجهة الموقع الرسومية باستخدام Gradio
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="مستكشف القطط الذكي 🧠",
    description="قم برفع أي صورة هنا، وسيقوم النموذج بالتنبؤ فوراً إذا كانت لقطة أم لا."
)

# تشغيل الموقع
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8501)

