# fake-news

Multimodal fake news detection บนชุดข้อมูล Fakeddit โดยใช้ CLIP embeddings ของหัวข้อโพสต์และรูป แล้วเทียบ classifier 3 โหมด (ข้อความ / รูป / ข้อความ+รูป)

## เตรียมข้อมูล

ไฟล์ข้อมูลไม่อยู่ใน git ต้องโหลดเองจาก Kaggle

1. โหลด `multimodal_train.tsv`, `multimodal_validate.tsv`, `multimodal_test_public.tsv` จากหน้า Fakeddit บน Kaggle
2. วางไว้ที่ `data/raw/` โดยไม่แก้ไขไฟล์

```
data/
├── raw/       # .tsv ต้นฉบับจาก Kaggle (ไม่ขึ้น git)
└── splits/    # รายการ id ของ train/val/test (ขึ้น git, ห้ามแบ่งใหม่เอง)
```

รูปที่โหลดมา, embeddings และโมเดลที่เทรนแล้ว เก็บใน Google Drive กลางของทีม

## ป้ายกำกับ

ใช้ `2_way_label`: **1 = ข่าวจริง, 0 = ข่าวปลอม**

ฟีเจอร์ที่ใช้มีแค่ `clean_title` กับรูปจาก `image_url` ห้ามใช้ `subreddit`, `3_way_label`, `6_way_label` หรือเมทาดาต้าอื่น เพราะ `subreddit` กำหนดป้ายได้ 100%
