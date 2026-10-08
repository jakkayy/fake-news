import sys
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fakenews.predict import Predictor

EXAMPLES_DIR = ROOT / "app" / "examples"
MODE_NAMES = {"text": "ข้อความ", "image": "รูป", "text+image": "ข้อความ + รูป"}
CHOICES = ["ทั้ง 3 โหมด", *MODE_NAMES.values()]

predictor = Predictor()


def run(title, image, choice):
    if not (title and title.strip()) and image is None:
        raise gr.Error("ใส่หัวข้อโพสต์ หรือรูป อย่างน้อยหนึ่งอย่าง")
    probs = predictor.predict(title, image)
    wanted = list(MODE_NAMES) if choice == CHOICES[0] else [m for m, n in MODE_NAMES.items() if n == choice]
    outputs = []
    for mode in MODE_NAMES:
        if mode in wanted and mode in probs:
            p = probs[mode]
            outputs.append(gr.Label(value={"จริง": p, "ปลอม": 1 - p}, visible=True))
        else:
            outputs.append(gr.Label(value=None, visible=mode in wanted))
    needs = {"text": "หัวข้อ", "image": "รูป", "text+image": "ทั้งหัวข้อและรูป"}
    note = "\n".join(f"โหมด{MODE_NAMES[m]}ต้องใส่{needs[m]}" for m in wanted if m not in probs)
    return (*outputs, note)


examples = pd.read_csv(EXAMPLES_DIR / "examples.csv")
example_rows = [[r.clean_title, str(EXAMPLES_DIR / r.image), r.case, f"{r.label_name} · r/{r.subreddit}"]
                for r in examples.itertuples()]

with gr.Blocks(title="Fakeddit fake news demo") as demo:
    gr.Markdown(
        "# ตรวจข่าวปลอมจากหัวข้อและรูป (Fakeddit)\n"
        "ใช้ CLIP ViT-B/32 แปลงหัวข้อและรูปเป็นเวกเตอร์ แล้วทำนายด้วย Logistic Regression แยก 3 โหมด\n\n"
        "ป้ายของชุดข้อมูลมาจาก subreddit ที่โพสต์ โมเดลจึงเรียนรู้สไตล์ของแต่ละชุมชน ไม่ได้ตรวจข้อเท็จจริง"
    )
    with gr.Row():
        with gr.Column():
            title = gr.Textbox(label="หัวข้อโพสต์ (ภาษาอังกฤษ)", lines=2)
            image = gr.Image(label="รูป", type="pil", height=300)
            choice = gr.Radio(CHOICES, value=CHOICES[0], label="โหมด")
            button = gr.Button("ทำนาย", variant="primary")
        with gr.Column():
            labels = [gr.Label(label=f"โหมด{name}", num_top_classes=2) for name in MODE_NAMES.values()]
            note = gr.Markdown()
    with gr.Row():
        case = gr.Textbox(label="กรณีของตัวอย่าง", interactive=False)
        truth = gr.Textbox(label="ป้ายจริงของตัวอย่าง", interactive=False)
    gr.Examples(example_rows, inputs=[title, image, case, truth], label="ตัวอย่างจากชุด test")
    button.click(run, inputs=[title, image, choice], outputs=[*labels, note])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share="--share" in sys.argv)
