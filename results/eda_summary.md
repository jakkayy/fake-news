# Exploration summary

สร้างจาก `scripts/eda.py` ตัวเลขทั้งหมดมาจาก `data/splits/prepare_stats.json` และ `data/processed/sample.csv`

## ขั้นการกรองข้อมูล

| ขั้น | แถวที่เหลือ |
|---|---|
| rows in multimodal_train.tsv | 564,000 |
| has image_url | 562,466 |
| drop 879 URLs with conflicting labels | 555,305 |
| one post per image_url | 546,614 |
| stratified sample (seed 42) | 16,500 |
| image downloaded and readable | 15,420 |

รูปที่โหลดไม่ได้ 1,080 รูป (6.5%): http_404 711, placeholder 319, bad_image:UnidentifiedImageError 13, ConnectionError 13, ConnectTimeout 8, http_403 6, http_410 3, http_503 3, http_400 2, TooManyRedirects 1, http_530 1

## ค่าว่างในไฟล์ต้นฉบับ (multimodal_train.tsv)

จาก 564,000 แถว 16 คอลัมน์ คอลัมน์ที่ไม่ได้ระบุว่างไม่มีค่าว่าง

| คอลัมน์ | ว่าง (%) | ใช้ในโมเดล |
|---|---|---|
| linked_submission_id | 70.24 | ไม่ (ตัดทิ้ง) |
| domain | 29.76 | ไม่ (ตัดทิ้ง) |
| upvote_ratio | 29.76 | ไม่ (ตัดทิ้ง) |
| num_comments | 29.76 | ไม่ (ตัดทิ้ง) |
| author | 5.09 | ไม่ (ตัดทิ้ง) |
| image_url | 0.27 | ใช่ |

เมทาดาต้าที่ว่างเยอะ (linked_submission_id, domain, upvote_ratio, num_comments) ไม่กระทบโมเดล เพราะใช้แค่หัวข้อกับรูป ส่วน image_url ที่ว่างตัดทิ้งตอนเตรียมข้อมูล

## สัดส่วนคลาสในแต่ละ split

| split | แถว | ข่าวจริง (1) |
|---|---|---|
| train | 10,794 | 39.0% |
| val | 2,313 | 39.0% |
| test | 2,313 | 39.0% |

## ความยาวหัวข้อ (คำ)

| คลาส | เฉลี่ย | มัธยฐาน | สูงสุด |
|---|---|---|---|
| fake (0) | 6.2 | 5 | 65 |
| real (1) | 9.4 | 9 | 52 |

## subreddit

มี 22 subreddit ในตัวอย่าง แต่ละ subreddit มีป้ายเดียว จึงห้ามใช้เป็นฟีเจอร์ ใหญ่สุดคือ r/psbattle_artwork (29.7%) ดู `figures/eda_subreddits.png`
