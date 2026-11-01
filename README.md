# Thai Orchid AI

เว็บ Streamlit สำหรับจำแนกกล้วยไม้ 5 สกุลจากภาพถ่าย ได้แก่ Cattleya, Dendrobium, Oncidium, Phalaenopsis และ Vanda รองรับการอัปโหลดรูปและถ่ายภาพผ่านกล้อง พร้อมข้อมูลลักษณะเด่น คำแนะนำ และแนวทางดูแลเบื้องต้น

## โมเดล

ผู้ใช้เลือกได้ 3 โมเดล:

- ResNet-50 — Test Accuracy 73.72%
- EfficientNet-B0 — Test Accuracy 73.88%
- ViT-B/16 — Test Accuracy 49.72%

คะแนนมาจากชุด Test 2,500 ภาพของโครงงาน ระบบจำแนกในระดับสกุลเท่านั้น ไม่ได้ยืนยันชนิดย่อยหรือสายพันธุ์การค้า

## โครงสร้างโปรเจกต์

```text
assignment_ml/
├── .streamlit/
│   └── config.toml
├── models/
│   ├── resnet50_best.pth
│   ├── efficientnet_b0_best.pth
│   └── vit_base_patch16_224_best.pth
├── src/
│   ├── __init__.py
│   └── orchid_data.py
├── training/
│   ├── train_experiment.py
│   ├── run_training.py
│   ├── requirements.txt
│   └── README.md
├── data/                 # ข้อมูลในเครื่อง ไม่ส่งขึ้น Git
│   ├── dataset/
│   └── legacy_source/
├── experiments/          # weights และรายงานจากการเทรน ไม่ส่งขึ้น Git
├── app.py
├── RUN_TRAINING.bat
├── requirements.txt
├── README.md
├── .gitattributes
└── .gitignore
```

ส่วนเว็บอยู่ที่รูทเพื่อให้ Streamlit Deploy ได้ง่าย โค้ดเทรนรวมอยู่ใน `training/` ข้อมูลอยู่ใน `data/` และผลการทดลองอยู่ใน `experiments/` โฟลเดอร์ข้อมูลและผลทดลองถูกตัดออกจาก Git ด้วย `.gitignore`

## รันในเครื่อง

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

เปิด `http://localhost:8501` หากเบราว์เซอร์ไม่เปิดอัตโนมัติ

## เทรนโมเดล

ดับเบิลคลิก `RUN_TRAINING.bat` หรือรัน:

```powershell
python -m pip install -r training/requirements.txt
python training/run_training.py
```

ดูรายละเอียดการเทรนที่ `training/README.md`

## Deploy บน Streamlit Community Cloud

weights ถูกกำหนดให้จัดเก็บด้วย Git LFS เนื่องจาก ViT มีขนาดใหญ่:

```powershell
git init
git lfs install
git add .
git commit -m "Deploy Thai Orchid AI"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

จากนั้นเข้า `https://share.streamlit.io` เลือก GitHub repository, branch `main` และ entrypoint `app.py` ใน Advanced settings แนะนำ Python 3.12

Streamlit Community Cloud จะดึงไฟล์ Git LFS อัตโนมัติ แอปโหลดและ cache เฉพาะโมเดลที่เลือก โดยเก็บ cache ไม่เกินหนึ่งโมเดลเพื่อลดการใช้ RAM

## ข้อจำกัด

- ควรถ่ายภาพดอกให้ชัดและอยู่กลางภาพ
- ภาพที่อยู่นอก 5 สกุลยังถูกบังคับให้เลือกหนึ่งใน 5 สกุล จึงควรดูค่าความมั่นใจร่วมด้วย
- คำแนะนำการดูแลเป็นข้อมูลทั่วไป ต้องปรับตามพันธุ์ย่อย ฤดูกาล วัสดุปลูก และสภาพอากาศ
- ViT-B/16 มีขนาดใหญ่และอาจโหลดช้ากว่าโมเดล CNN บน Community Cloud
