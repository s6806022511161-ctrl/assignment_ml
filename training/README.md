# คู่มือเทรนและข้อมูลประกอบสไลด์

รันจากโฟลเดอร์ assignment_ml ด้วย Python ข้อมูลต้นฉบับจะไม่ถูกย้ายหรือแก้ไข

## รันไฟล์เดียวบนคอม

หลังติดตั้งไลบรารีแล้ว ดับเบิลคลิก `RUN_TRAINING.bat` ที่รูทโปรเจกต์ หรือรัน:

```powershell
python training/run_training.py
```

ไฟล์นี้จะทำงานครบลำดับเดียว: ตรวจข้อมูล -> สร้าง Validation 20% เมื่อ val ว่าง -> เทรน ResNet-50, EfficientNet-B0 และ ViT-B/16 -> เลือก checkpoint จาก Validation -> ประเมิน Test -> สร้าง CSV/PNG -> เปิด `report.html` อัตโนมัติ ผลแต่ละรอบอยู่ใน `experiments/local_วันเวลา/` เพื่อไม่เขียนทับกัน

บน Windows สามารถดับเบิลคลิก `RUN_TRAINING.bat` ได้เช่นกัน โปรแกรมจะเปิดหน้าต่างคำสั่งและคงหน้าต่างไว้เพื่อให้อ่านผลหรือข้อความผิดพลาดได้

โหมดอัตโนมัติ: ถ้าพบ CUDA จะใช้ batch size 32 และ fine-tune ทุกชั้นเหมือนแนวทางในโน้ตบุ๊ก หากเครื่องใช้ CPU จะใช้ batch size 16 และฝึก classification head ของทุกโมเดล เพื่อให้การเปรียบเทียบทั้งสามโมเดลมีโอกาสรันจบบนคอมทั่วไป รายงานจะระบุ device และ config ที่ใช้เสมอ

## ติดตั้ง

```powershell
python -m pip install -r training/requirements.txt
```

## ตรวจชุดข้อมูลก่อน

```powershell
python training/train_experiment.py --output experiments/data_check --prepare-only
```

รองรับ train/cattariya/cattleya และเปลี่ยนชื่อ label เป็น cattleya ใน manifest อัตโนมัติ หาก val ว่าง จะสุ่มแยก 20% ของแต่ละสกุลใน train ด้วย seed 42 โดยอ้างอิงภาพผ่าน manifest ไม่คัดลอกลง dataset/val หากมี val แล้วจะใช้ชุดเดิม ต้องมีครบ 5 สกุลในทุกชุด

ตรวจภาพซ้ำจากขนาดและพิกเซล RGB ลบรายการซ้ำภายในชุดจาก manifest และหยุดเมื่อพบภาพซ้ำข้ามชุดหรือ label ขัดแย้งกัน การตรวจนี้ยังไม่ตรวจจับภาพที่ crop/resize/แต่งสีมาจากภาพต้นฉบับเดียวกัน ควรแบ่งตามภาพต้นฉบับหรือแหล่งถ่ายก่อน augmentation และตรวจแหล่งที่มาของ Test ด้วย

## เทรน

คำสั่งมาตรฐานสำหรับเทรนทั้ง 3 โมเดลของโครงงาน:

```powershell
python -u training/train_experiment.py --output experiments/comparison_v1 --epochs 20 --patience 5
```

เมื่อไม่ระบุ `--models` สคริปต์จะเทรน `resnet50`, `efficientnet_b0` และ `vit_base_patch16_224` ตามลำดับ พร้อมสร้าง `benchmark.csv` และ `benchmark.png` เปรียบเทียบทั้งสามโมเดล

CPU เริ่มจาก EfficientNet-B0 และฝึกเฉพาะหัวจำแนก:

```powershell
python -u training/train_experiment.py --output experiments/efficientnet_head_v1 --models efficientnet_b0 --head-only --epochs 10 --device cpu
```

Fine-tune ทั้งโมเดล (ใช้เวลามากกว่าบน CPU):

```powershell
python -u training/train_experiment.py --output experiments/efficientnet_full_v1 --models efficientnet_b0 --epochs 20
```

เปรียบเทียบสามโมเดลด้วยข้อมูลและเงื่อนไขเดียวกัน:

```powershell
python -u training/train_experiment.py --output experiments/comparison_v1 --models resnet50 efficientnet_b0 vit_base_patch16_224 --epochs 20 --patience 5
```

ใช้ CUDA อัตโนมัติหาก PyTorch รองรับและมี GPU ไม่เช่นนั้นใช้ CPU ใช้ --batch-size 8 หากหน่วยความจำไม่พอ ใช้ --patience 0 เพื่อเทรนครบจำนวนรอบ ค่าเริ่มต้นเป็น full fine-tuning; --head-only ตรึง backbone และใช้ eval mode เพื่อไม่เปลี่ยน BatchNorm ของ backbone

แต่ละการทดลองต้องใช้ --output ใหม่เพื่อไม่เขียนทับผลเดิม สคริปต์นี้ยังไม่รองรับ resume กด Ctrl+C เพื่อหยุดได้ ผล history และ best weights ที่เขียนแล้วจะยังอยู่ การรันครั้งแรกอาจดาวน์โหลด ImageNet weights

## ไฟล์สำหรับสไลด์

| ไฟล์ | ใช้ประกอบหัวข้อ |
|---|---|
| dataset_counts.csv | จำนวนภาพรายสกุล แบ่ง Train/Val/Test |
| split_manifest.csv | เส้นทางภาพ label split และ hash เพื่อทำซ้ำ |
| config.json | seed, hyperparameters, device, เวอร์ชัน PyTorch |
| benchmark.csv / benchmark.png | เปรียบเทียบ Accuracy, Macro Precision/Recall/F1, Weighted F1, ขนาดโมเดล, ความเร็ว และเวลาเทรน |
| ชื่อโมเดล/history.csv | Train/Validation Loss และ Accuracy ทุก epoch |
| ชื่อโมเดล/learning_curves.png | กราฟการเรียนรู้และตรวจ overfitting |
| ชื่อโมเดล/best_epoch.json | รอบที่มี Validation Accuracy สูงที่สุด |
| ชื่อโมเดล/per_class.csv | Precision, Recall, F1 และ support รายสกุล |
| ชื่อโมเดล/confusion_matrix.png / .csv | คู่สกุลที่ทำนายสับสน |
| ชื่อโมเดล/predictions.csv | ผล Test รายภาพและ softmax ทุกสกุล |
| ชื่อโมเดล/ชื่อโมเดล_best.pth | weights ของโมเดลที่เลือกจาก Validation |

Accuracy ที่ลงท้าย _pct มีหน่วยเปอร์เซ็นต์ ส่วน F1/Precision/Recall และ accuracy ใน history อยู่ในช่วง 0–1 ต้องคูณ 100 เมื่อแสดงเป็นเปอร์เซ็นต์

Latency เป็น model-only batch size 1: warm-up 10 ครั้ง วัด 30 ครั้ง รายงาน median และ p95 พร้อม synchronize CUDA ไม่รวมอ่านภาพ/ปรับขนาด/ส่งข้อมูลเข้า GPU ไม่ใช่เวลาใช้งานเว็บทั้งหมด

รายงานคะแนน Test หลังเลือก checkpoint ด้วย Validation เท่านั้น อย่าใช้ Test เลือก epoch หรือปรับ hyperparameters ไม่ควรนำผล head-only ไปเทียบกับ full fine-tuning โดยไม่ระบุความต่าง ค่าความแม่นยำจากการทดลองเก่าที่ใช้ข้อมูลคนละชุดใช้แทนผลใหม่ไม่ได้

## ใช้กับเว็บเดิม

หลังตรวจผลแล้วค่อยคัดลอก weights ที่ต้องการจากโฟลเดอร์การทดลองไป models/ ด้วยชื่อ efficientnet_b0_best.pth, resnet50_best.pth หรือ vit_base_patch16_224_best.pth ตามโมเดล ลำดับคลาสในสคริปต์ตรงกับ app.py: cattleya, dendrobium, oncidium, phalaenopsis, vanda

กราฟใช้ภาษาอังกฤษเพื่อให้รันได้โดยไม่ต้องติดตั้งฟอนต์ไทยเพิ่มเติม สคริปต์นี้ไม่สร้าง Grad-CAM; เว็บเดิมมีฟังก์ชันนี้สำหรับ CNN
