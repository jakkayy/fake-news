# fake-news

Multimodal fake news detection บนชุดข้อมูล Fakeddit โดยใช้ CLIP embeddings ของหัวข้อโพสต์และรูป แล้วเทียบ classifier 3 โหมด (ข้อความ / รูป / ข้อความ+รูป)

## ติดตั้ง

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

## ใช้ข้อมูลที่เตรียมไว้แล้ว (แนะนำ)

รูปและ embeddings ถูกโหลดและดึงไว้แล้ว ไม่ต้องรันซ้ำ

1. โหลด `embeddings.zip` (56 MB) จาก Google Drive กลางของทีม แล้วแตกไว้ใน `data/`
2. ถ้าต้องดูรูป (วิเคราะห์ข้อผิดพลาด, สไลด์, เดโม) โหลด `images.zip` (348 MB) แล้วแตกไว้ใน `data/` เช่นกัน
3. `git pull` ให้ได้ `data/splits/split.csv` ล่าสุด

```python
import sys; sys.path.insert(0, "src")
from fakenews.embed_store import load_embeddings

X_text, X_image, y, ids = load_embeddings("data/embeddings", "data/splits/split.csv", "train")
```

`split` เป็นได้ `"train"`, `"val"`, `"test"` ห้ามสุ่มแบ่งใหม่เอง

```
data/
├── raw/          # .tsv ต้นฉบับจาก Kaggle (ไม่ขึ้น git)
├── images/       # <id>.jpg ย่อด้านสั้นเหลือ 256 px (จาก images.zip)
├── embeddings/   # text.npy, image.npy, ids.csv, labels.npy เรียงแถวตรงกัน (จาก embeddings.zip)
└── splits/       # split.csv และ prepare_stats.json (ขึ้น git)
```

## เดโม

ต้องมีโมเดลใน `models/` ก่อน (`.venv/bin/python scripts/train_models.py` ใช้ไม่ถึง 1 นาที เมื่อมี `data/embeddings/` แล้ว)

```bash
.venv/bin/python app/demo.py
```

ถ้าอยากให้คนอื่นลองผ่านเน็ต รัน `.venv/bin/python app/demo.py --share` จะได้ลิงก์ `https://….gradio.live` ใช้ได้ 72 ชั่วโมง ตราบที่เครื่องยังเปิดแอปอยู่

เปิด http://127.0.0.1:7860 ใส่หัวข้อโพสต์ภาษาอังกฤษกับรูป หรือกดตัวอย่าง 8 ตัวจากชุด test ใน `app/examples/` ซึ่งใช้ได้แม้ไม่มีเน็ต (ยกเว้นครั้งแรกที่ต้องโหลด CLIP)

## สร้างข้อมูลใหม่ตั้งแต่ต้น

ทำเฉพาะเมื่อจำเป็น ใช้เวลาประมาณ 50 นาทีบน CPU

1. โหลด `multimodal_train.tsv`, `multimodal_validate.tsv`, `multimodal_test_public.tsv` จากหน้า Fakeddit บน Kaggle มาวางที่ `data/raw/`
2. `.venv/bin/python scripts/prepare_data.py` สุ่ม 16,500 แถว (seed 42) โหลดรูป และแบ่ง train/val/test 70/15/15
3. `.venv/bin/python scripts/extract_embeddings.py` ดึง CLIP ViT-B/32 embeddings

## ป้ายกำกับ

ใช้ `2_way_label`: **1 = ข่าวจริง, 0 = ข่าวปลอม**

ฟีเจอร์ที่ใช้มีแค่ `clean_title` กับรูปจาก `image_url` ห้ามใช้ `subreddit`, `3_way_label`, `6_way_label` หรือเมทาดาต้าอื่น เพราะ `subreddit` กำหนดป้ายได้ 100%
