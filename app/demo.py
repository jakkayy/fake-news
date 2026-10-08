import json
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
CLASSIFIER_NAMES = {"logreg": "Logistic Regression (ตัวหลักในรายงาน)", "mlp": "MLP"}


class DecisionLabel(gr.Label):
    is_template = True

    def postprocess(self, value):
        if isinstance(value, dict) and "decision" in value:
            data = super().postprocess(value["confidences"])
            data.label = value["decision"]
            return data
        return super().postprocess(value)


predictor = Predictor()
SIMILARITY_THRESHOLD = json.loads((ROOT / "results" / "similarity_warning.json").read_text())["threshold"]
classifier_choices = [CLASSIFIER_NAMES[k] for k in predictor.classifiers]


def run(title, image, choice, classifier_name):
    if not (title and title.strip()) and image is None:
        raise gr.Error("ใส่หัวข้อโพสต์ หรือรูป อย่างน้อยหนึ่งอย่าง")
    classifier = next(k for k, n in CLASSIFIER_NAMES.items() if n == classifier_name)
    scored = predictor.score(title, image, classifier)
    probs, similarity = scored["probs"], scored["similarity"]
    wanted = list(MODE_NAMES) if choice == CHOICES[0] else [m for m, n in MODE_NAMES.items() if n == choice]
    outputs = []
    for mode, name in MODE_NAMES.items():
        if mode in wanted and mode in probs:
            p, t = probs[mode], predictor.thresholds[classifier][mode]
            outputs.append(DecisionLabel(
                value={"decision": "จริง" if p >= t else "ปลอม", "confidences": {"จริง": p, "ปลอม": 1 - p}},
                label=f"โหมด{name} · ตอบว่าจริงเมื่อ P(จริง) ≥ {t:.2f}", visible=True))
        else:
            outputs.append(DecisionLabel(value=None, label=f"โหมด{name}", visible=mode in wanted))
    needs = {"text": "หัวข้อ", "image": "รูป", "text+image": "ทั้งหัวข้อและรูป"}
    lines = [f"โหมด{MODE_NAMES[m]}ต้องใส่{needs[m]}" for m in wanted if m not in probs]
    if similarity is not None and similarity < SIMILARITY_THRESHOLD:
        lines.append(f"⚠️ **รูปกับหัวข้อดูไม่เกี่ยวกัน** (ความสอดคล้อง {similarity:.2f} ต่ำกว่าเกณฑ์ {SIMILARITY_THRESHOLD:.2f} "
                     "ซึ่งเป็น 5% ล่างสุดของชุดเทรน) อาจเป็นการใช้รูปผิดบริบท โมเดลไม่ได้ใช้ค่านี้ในการทำนาย")
    elif similarity is not None:
        lines.append(f"ความสอดคล้องของรูปกับหัวข้อ {similarity:.2f} (เกณฑ์เตือน < {SIMILARITY_THRESHOLD:.2f})")
    return (*outputs, "\n\n".join(lines))


examples = pd.read_csv(EXAMPLES_DIR / "examples.csv")
example_rows = [[r.clean_title, str(EXAMPLES_DIR / r.image), r.case, f"{r.label_name} · r/{r.subreddit}"]
                for r in examples.itertuples()]

with gr.Blocks(title="Fakeddit fake news demo") as demo:
    gr.Markdown(
        "# ตรวจข่าวปลอมจากหัวข้อและรูป (Fakeddit)\n"
        "ใช้ CLIP ViT-B/32 แปลงหัวข้อและรูปเป็นเวกเตอร์ แล้วทำนายด้วย Logistic Regression หรือ MLP แยก 3 โหมด\n\n"
        "ป้ายของชุดข้อมูลมาจาก subreddit ที่โพสต์ โมเดลจึงเรียนรู้สไตล์ของแต่ละชุมชน ไม่ได้ตรวจข้อเท็จจริง"
    )
    with gr.Row():
        with gr.Column():
            title = gr.Textbox(label="หัวข้อโพสต์ (ภาษาอังกฤษ)", lines=2)
            image = gr.Image(label="รูป", type="pil", height=300)
            choice = gr.Radio(CHOICES, value=CHOICES[0], label="โหมด")
            classifier = gr.Radio(classifier_choices, value=classifier_choices[0], label="Classifier")
            button = gr.Button("ทำนาย", variant="primary")
        with gr.Column():
            labels = [DecisionLabel(label=f"โหมด{name}", num_top_classes=2) for name in MODE_NAMES.values()]
            note = gr.Markdown()
    with gr.Row():
        case = gr.Textbox(label="กรณีของตัวอย่าง", interactive=False)
        truth = gr.Textbox(label="ป้ายจริงของตัวอย่าง", interactive=False)
    gr.Examples(example_rows, inputs=[title, image, case, truth], label="ตัวอย่างจากชุด test")
    button.click(run, inputs=[title, image, choice, classifier], outputs=[*labels, note])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share="--share" in sys.argv)
