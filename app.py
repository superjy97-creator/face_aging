import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from pathlib import Path
from io import BytesIO
import html

# =========================================================
# Streamlit 湲곕낯 ?ㅼ젙
# =========================================================

st.set_page_config(
    page_title="?쇨뎬 ?대?吏 湲곕컲 ?섏씠? 遺꾩꽍 ?뱀빋",
    page_icon="?벜",
    layout="wide"
)

# =========================================================
# 寃쎈줈 ?ㅼ젙
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "outputs" / "models" / "efficientnet_b0_age_past_5class_best.pth"
MODEL_INFO_CSV_PATH = BASE_DIR / "outputs" / "csv" / "final_selected_model_efficientnet_b0.csv"

CSV_DIR = BASE_DIR / "outputs" / "csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"
MODEL_DIR = BASE_DIR / "outputs" / "models"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# ?대옒???뺣낫
# =========================================================

CLASS_NAME_MAP = {
    0: "10? ?댄븯",
    1: "20?",
    2: "30?",
    3: "40?",
    4: "50? ?댁긽"
}

NUM_CLASSES = 5

# =========================================================
# 怨듯넻 ?⑥닔
# =========================================================

def safe_read_csv(csv_path):
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def show_csv_if_exists(title, csv_path):
    st.markdown(f"### {title}")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.caption(str(csv_path))
        st.dataframe(df, use_container_width=True)
    else:
        st.warning(f"CSV ?뚯씪??李얠쓣 ???놁뒿?덈떎: {csv_path}")


def show_image_if_exists(title, image_path):
    st.markdown(f"### {title}")

    if image_path.exists():
        st.caption(str(image_path))
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning(f"?대?吏 ?뚯씪??李얠쓣 ???놁뒿?덈떎: {image_path}")


# =========================================================
# 紐⑤뜽 ?앹꽦 ?⑥닔
# =========================================================

def create_efficientnet_b0_model(num_classes=5):
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# =========================================================
# 紐⑤뜽 濡쒕뱶 ?⑥닔
# =========================================================

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"紐⑤뜽 ?뚯씪??李얠쓣 ???놁뒿?덈떎: {MODEL_PATH}")

    model = create_efficientnet_b0_model(NUM_CLASSES)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()

    return model, device, checkpoint


# =========================================================
# ?대?吏 ?꾩쿂由??⑥닔
# =========================================================

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


# =========================================================
# ?덉륫 ?⑥닔
# =========================================================

def predict_age_group(model, device, image):
    image_tensor = preprocess_image(image)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probs = torch.softmax(outputs, dim=1)

        confidence, pred_idx = torch.max(probs, dim=1)

    pred_idx = int(pred_idx.item())

    confidence = float(confidence.item())

    probs = probs.squeeze(0).cpu().tolist()

    return pred_idx, confidence, probs


# =========================================================
# ?ъ씠?쒕컮
# =========================================================

st.sidebar.title("?뱦 硫붾돱")

page = st.sidebar.radio(
    "?섏씠吏 ?좏깮",
    [
        "?벜 ?섏씠? 遺꾩꽍",
        "?뱤 ?쒓컖??,
        "?뱲 ?⑹뼱 ?뺣━"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 理쒖쥌 ?좏깮 紐⑤뜽")
st.sidebar.success("EfficientNet-B0")

st.sidebar.markdown("### ?섏씠? ?대옒??)
st.sidebar.write("0 = 10? ?댄븯")
st.sidebar.write("1 = 20?")
st.sidebar.write("2 = 30?")
st.sidebar.write("3 = 40?")
st.sidebar.write("4 = 50? ?댁긽")


# =========================================================
# 1?섏씠吏. ?섏씠? 遺꾩꽍
# =========================================================

if page == "?벜 ?섏씠? 遺꾩꽍":

    st.title("?벜 ?쇨뎬 ?대?吏 湲곕컲 ?섏씠? 遺꾩꽍 ?뱀빋")

    st.write(
        """
        ?뱀틺?쇰줈 ?쇨뎬 ?대?吏瑜?珥ъ쁺????  
        理쒖쥌 ?좏깮 紐⑤뜽??**EfficientNet-B0 best 紐⑤뜽**???댁슜???섏씠?瑜??덉륫?⑸땲??
        """
    )

    with st.expander("?뱦 紐⑤뜽 ?뚯씪 諛?CSV 寃쎈줈 ?뺤씤"):
        st.write("?꾨줈?앺듃 湲곗? ?대뜑:", str(BASE_DIR))
        st.write("紐⑤뜽 ?뚯씪 寃쎈줈:", str(MODEL_PATH))
        st.write("紐⑤뜽 ?뚯씪 議댁옱 ?щ?:", MODEL_PATH.exists())
        st.write("紐⑤뜽 ?뺣낫 CSV 寃쎈줈:", str(MODEL_INFO_CSV_PATH))
        st.write("紐⑤뜽 ?뺣낫 CSV 議댁옱 ?щ?:", MODEL_INFO_CSV_PATH.exists())

    if not MODEL_PATH.exists():
        st.error(f"紐⑤뜽 ?뚯씪??李얠쓣 ???놁뒿?덈떎:\n\n{MODEL_PATH}")
        st.stop()

    if MODEL_INFO_CSV_PATH.exists():
        model_info_df = pd.read_csv(MODEL_INFO_CSV_PATH)

        with st.expander("?뱞 理쒖쥌 ?좏깮 紐⑤뜽 CSV ?뺣낫"):
            st.dataframe(model_info_df, use_container_width=True)
    else:
        st.warning("理쒖쥌 ?좏깮 紐⑤뜽 CSV ?뚯씪??李얠? 紐삵뻽?듬땲?? 紐⑤뜽 ?뚯씪???덉쑝硫?遺꾩꽍? 媛?ν빀?덈떎.")

    try:
        model, device, checkpoint = load_model()
        st.success("??EfficientNet-B0 best 紐⑤뜽 濡쒕뱶 ?꾨즺")
    except Exception as e:
        st.error(f"紐⑤뜽 濡쒕뱶 以??ㅻ쪟 諛쒖깮:\n\n{e}")
        st.stop()

    with st.expander("?쭬 Best 紐⑤뜽 checkpoint ?뺣낫"):
        st.write("?ъ슜 ?μ튂:", str(device))

        if isinstance(checkpoint, dict):
            st.write("Best Epoch:", checkpoint.get("epoch"))
            st.write("Best Val Loss:", checkpoint.get("best_val_loss"))
            st.write("Best Val Acc:", checkpoint.get("best_val_acc"))
            st.write("Label Source:", checkpoint.get("label_source"))
        else:
            st.write("checkpoint ?뺤떇: state_dict only")

    st.divider()

    st.subheader("1. ?뱀틺 珥ъ쁺")

    st.info(
        "釉뚮씪?곗? 蹂댁븞??移대찓??沅뚰븳 ?덉슜???꾩슂?⑸땲?? "
        "移대찓??沅뚰븳???덉슜????李곗뭇 踰꾪듉???뚮윭 ?ъ쭊??珥ъ쁺?섏꽭??"
    )

    camera_col, button_col = st.columns([3, 1])

    with camera_col:
        camera_image = st.camera_input("?벝 李곗뭇 踰꾪듉?쇰줈 ?쇨뎬 ?대?吏 珥ъ쁺")

    with button_col:
        st.write("")
        st.write("")
        st.write("")
        analyze_button = st.button("?뵇 遺꾩꽍?섍린", use_container_width=True)

    st.divider()

    st.subheader("2. 遺꾩꽍 寃곌낵")

    if analyze_button:

        if camera_image is None:
            st.warning("癒쇱? ?뱀틺?쇰줈 ?ъ쭊??珥ъ쁺?섏꽭??")

        else:
            image = Image.open(BytesIO(camera_image.getvalue()))

            st.image(image, caption="珥ъ쁺???대?吏", use_container_width=True)

            pred_idx, confidence, probs = predict_age_group(model, device, image)

            pred_name = CLASS_NAME_MAP[pred_idx]

            st.markdown("### ???덉륫 寃곌낵")

            st.success(f"?덉륫 ?섏씠?: **{pred_name}**")

            st.metric(
                label="?덉륫 ?좊ː??,
                value=f"{confidence * 100:.2f}%"
            )

            prob_df = pd.DataFrame({
                "?섏씠?": [CLASS_NAME_MAP[i] for i in range(NUM_CLASSES)],
                "?뺣쪧(%)": [round(p * 100, 2) for p in probs]
            })

            st.markdown("### ?대옒?ㅻ퀎 ?덉륫 ?뺣쪧")
            st.dataframe(prob_df, use_container_width=True)

            st.markdown("### ?대옒?ㅻ퀎 ?뺣쪧 洹몃옒??)
            st.bar_chart(prob_df.set_index("?섏씠?"))

            st.info(
                "??寃곌낵??AI 紐⑤뜽???덉륫媛믪엯?덈떎. "
                "?ㅼ젣 ?섏씠瑜??뺥솗???먮떒?섎뒗 怨듭떇 湲곗??쇰줈 ?ъ슜?섎㈃ ???⑸땲??"
            )

    else:
        st.info("?ъ쭊??珥ъ쁺???? ?ㅻⅨ履쎌쓽 遺꾩꽍?섍린 踰꾪듉???뚮윭二쇱꽭??")


# =========================================================
# 2?섏씠吏. ?쒓컖??
# =========================================================

elif page == "?뱤 ?쒓컖??:

    st.title("?뱤 ?숈뒿 寃곌낵 ?쒓컖???섏씠吏")

    st.write(
        """
        ???섏씠吏?먯꽌??EfficientNet-B0? MobileNetV2???숈뒿 寃곌낵,  
        紐⑤뜽 鍮꾧탳 CSV, 理쒖쥌 ?좏깮 紐⑤뜽 ?뺣낫, ??λ맂 洹몃옒???뚯씪???뺤씤?????덉뒿?덈떎.
        """
    )

    st.divider()

    st.subheader("1. 紐⑤뜽 鍮꾧탳 ?붿빟")

    model_compare_summary_path = CSV_DIR / "model_compare_summary.csv"
    model_compare_history_path = CSV_DIR / "model_compare_history_merged.csv"
    final_selected_model_path = CSV_DIR / "final_selected_model_efficientnet_b0.csv"

    summary_df = safe_read_csv(model_compare_summary_path)

    if summary_df is not None:
        st.dataframe(summary_df, use_container_width=True)

        if "model" in summary_df.columns and "best_val_acc" in summary_df.columns:
            st.markdown("### 紐⑤뜽蹂?Best Validation Accuracy")
            acc_chart_df = summary_df[["model", "best_val_acc"]].set_index("model")
            st.bar_chart(acc_chart_df)

        if "model" in summary_df.columns and "best_val_loss" in summary_df.columns:
            st.markdown("### 紐⑤뜽蹂?Best Validation Loss")
            loss_chart_df = summary_df[["model", "best_val_loss"]].set_index("model")
            st.bar_chart(loss_chart_df)

        if "model" in summary_df.columns and "total_train_time_min" in summary_df.columns:
            st.markdown("### 紐⑤뜽蹂?珥??숈뒿 ?쒓컙")
            time_chart_df = summary_df[["model", "total_train_time_min"]].set_index("model")
            st.bar_chart(time_chart_df)

    else:
        st.warning(f"紐⑤뜽 鍮꾧탳 ?붿빟 CSV瑜?李얠쓣 ???놁뒿?덈떎: {model_compare_summary_path}")

    st.divider()

    st.subheader("2. 理쒖쥌 ?좏깮 紐⑤뜽 ?뺣낫")

    final_df = safe_read_csv(final_selected_model_path)

    if final_df is not None:
        st.dataframe(final_df, use_container_width=True)
        st.success("理쒖쥌 ?좏깮 紐⑤뜽: EfficientNet-B0")
    else:
        st.warning(f"理쒖쥌 ?좏깮 紐⑤뜽 CSV瑜?李얠쓣 ???놁뒿?덈떎: {final_selected_model_path}")

    st.divider()

    st.subheader("3. ?숈뒿 湲곕줉 CSV")

    tab1, tab2, tab3 = st.tabs(
        [
            "EfficientNet-B0 History",
            "MobileNetV2 History",
            "Merged History"
        ]
    )

    with tab1:
        show_csv_if_exists(
            "EfficientNet-B0 ?숈뒿 湲곕줉",
            CSV_DIR / "efficientnet_b0_age_past_5class_history.csv"
        )

    with tab2:
        show_csv_if_exists(
            "MobileNetV2 ?숈뒿 湲곕줉",
            CSV_DIR / "mobilenet_v2_age_past_5class_history.csv"
        )

    with tab3:
        show_csv_if_exists(
            "紐⑤뜽 鍮꾧탳 ?듯빀 ?숈뒿 湲곕줉",
            model_compare_history_path
        )

    st.divider()

    st.subheader("4. ??λ맂 洹몃옒???먮룞 ?쒖떆")

    image_files = []

    if FIG_DIR.exists():
        image_files.extend(list(FIG_DIR.rglob("*.png")))
        image_files.extend(list(FIG_DIR.rglob("*.jpg")))
        image_files.extend(list(FIG_DIR.rglob("*.jpeg")))

    if MODEL_DIR.exists():
        image_files.extend(list(MODEL_DIR.rglob("*.png")))
        image_files.extend(list(MODEL_DIR.rglob("*.jpg")))
        image_files.extend(list(MODEL_DIR.rglob("*.jpeg")))

    image_files = sorted(list(set(image_files)))

    if len(image_files) == 0:
        st.warning("?쒖떆??洹몃옒???대?吏 ?뚯씪???놁뒿?덈떎.")
    else:
        st.write(f"李얠? ?대?吏 ?뚯씪 ?? {len(image_files)}媛?)

        image_info_df = pd.DataFrame([
            {
                "file_name": p.name,
                "folder": str(p.parent),
                "full_path": str(p),
                "size_kb": round(p.stat().st_size / 1024, 2)
            }
            for p in image_files
        ])

        with st.expander("?대?吏 ?뚯씪 紐⑸줉 蹂닿린"):
            st.dataframe(image_info_df, use_container_width=True)

        for img_path in image_files:
            st.markdown(f"### {img_path.name}")
            st.caption(str(img_path))
            st.image(str(img_path), use_container_width=True)

    st.divider()

    st.subheader("5. outputs ?대뜑 ??CSV ?뚯씪 紐⑸줉")

    csv_files = []

    if CSV_DIR.exists():
        csv_files.extend(list(CSV_DIR.rglob("*.csv")))

    csv_files = sorted(list(set(csv_files)))

    if len(csv_files) == 0:
        st.warning("CSV ?뚯씪??李얠? 紐삵뻽?듬땲??")
    else:
        csv_info_df = pd.DataFrame([
            {
                "file_name": p.name,
                "folder": str(p.parent),
                "full_path": str(p),
                "size_kb": round(p.stat().st_size / 1024, 2)
            }
            for p in csv_files
        ])

        st.dataframe(csv_info_df, use_container_width=True)


# =========================================================
# 3?섏씠吏. ?⑹뼱 ?뺣━
# =========================================================

elif page == "?뱲 ?⑹뼱 ?뺣━":

    st.title("?뱲 ?꾨줈?앺듃 ?⑹뼱 ?뺣━ ?섏씠吏")

    st.write(
        """
        ???섏씠吏???쇨뎬 ?대?吏 湲곕컲 ?섏씠? 遺꾨쪟 ?꾨줈?앺듃?먯꽌 ?ъ슜???⑹뼱瑜?移대뱶 ?뺤떇?쇰줈 ?뺣━???섏씠吏?낅땲??  
        ?곗씠??以鍮? ?꾩쿂由? ?숈뒿, ?됯?, 洹몃옒?? ?뱀빋 ?쒖옉 怨쇱젙?먯꽌 ?ъ슜???⑹뼱瑜??뺤씤?????덉뒿?덈떎.
        """
    )

    st.divider()

    terms_data = [
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "ZIP ?뚯씪", "?섎?": "?щ윭 ?뚯씪???섎굹濡?臾띠뼱 ?뺤텞???뚯씪", "?꾨줈?앺듃?먯꽌????븷": "AI Hub ?먮낯 ?대?吏? ?쇰꺼留??곗씠?곕? ?뺤텞 ?댁젣?섍린 ???뺥깭"},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "?뺤텞 ?댁젣", "?섎?": "ZIP ?뚯씪 ?덉쓽 ?ㅼ젣 ?뚯씪??爰쇰궡??怨쇱젙", "?꾨줈?앺듃?먯꽌????븷": "?대?吏 ?뚯씪怨?JSON ?쇰꺼 ?뚯씪???ㅼ젣 ?대뜑濡?????숈뒿 以鍮?},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "?대?吏 ?뚯씪", "?섎?": "紐⑤뜽???낅젰?쇰줈 諛쏅뒗 ?쇨뎬 ?ъ쭊 ?곗씠??, "?꾨줈?앺듃?먯꽌????븷": "EfficientNet-B0 紐⑤뜽???섏씠?瑜??덉륫?섍린 ?꾪빐 ?숈뒿???낅젰 ?곗씠??},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "JSON ?쇰꺼 ?뚯씪", "?섎?": "?대?吏??????뺣떟 ?뺣낫媛 ?ㅼ뼱?덈뒗 ?뚯씪", "?꾨줈?앺듃?먯꽌????븷": "age_now, age_past, gender ?깆쓽 ?뺣낫瑜?異붿텧?섎뒗 ???ъ슜"},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "CSV ?뚯씪", "?섎?": "???뺥깭???곗씠?곕? ??ν븯???뚯씪", "?꾨줈?앺듃?먯꽌????븷": "?대?吏 寃쎈줈, ?쇰꺼, ?숈뒿 湲곕줉, 紐⑤뜽 鍮꾧탳 寃곌낵 ???},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "file_stem", "?섎?": "?뺤옣?먮? ?쒖쇅???뚯씪 ?대쫫", "?꾨줈?앺듃?먯꽌????븷": "?대?吏 ?뚯씪怨?JSON ?쇰꺼 ?뚯씪??媛숈? ?대쫫 湲곗??쇰줈 留ㅼ묶"},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "image_path", "?섎?": "?대?吏 ?뚯씪????λ맂 ?꾩껜 寃쎈줈", "?꾨줈?앺듃?먯꽌????븷": "Dataset???ㅼ젣 ?대?吏瑜?遺덈윭?????ъ슜"},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "json_path", "?섎?": "JSON ?쇰꺼 ?뚯씪????λ맂 ?꾩껜 寃쎈줈", "?꾨줈?앺듃?먯꽌????븷": "?쇰꺼 ?뺣낫瑜?異붿텧??JSON ?뚯씪 ?꾩튂"},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "?대?吏-JSON 留ㅽ븨", "?섎?": "?대?吏 ?뚯씪怨??대떦 JSON ?쇰꺼 ?뚯씪???곌껐?섎뒗 怨쇱젙", "?꾨줈?앺듃?먯꽌????븷": "?대?吏? ?뺣떟 ?쇰꺼??紐⑤몢 ?덈뒗 ?곗씠?곕쭔 ?숈뒿???ъ슜?섍린 ?꾪빐 ?섑뻾"},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "MATCHED", "?섎?": "?대?吏? JSON??紐⑤몢 議댁옱?섎뒗 ?뺤긽 ?곗씠??, "?꾨줈?앺듃?먯꽌????븷": "理쒖쥌 ?숈뒿???ъ슜???곗씠??},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "MISSING_IMAGE", "?섎?": "JSON? ?덉?留??대?吏媛 ?녿뒗 ?곹깭", "?꾨줈?앺듃?먯꽌????븷": "?대?吏媛 ?놁쑝誘濡??숈뒿?먯꽌 ?쒖쇅"},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "MISSING_JSON", "?섎?": "?대?吏???덉?留?JSON ?쇰꺼???녿뒗 ?곹깭", "?꾨줈?앺듃?먯꽌????븷": "?뺣떟 ?쇰꺼???놁쑝誘濡??숈뒿?먯꽌 ?쒖쇅"},
        {"遺꾨쪟": "?곗씠??以鍮?, "?⑹뼱": "寃곗륫移?, "?섎?": "?꾩슂??媛믪씠???뚯씪??鍮좎졇 ?덈뒗 ?곗씠??, "?꾨줈?앺듃?먯꽌????븷": "MISSING_IMAGE, MISSING_JSON ?곗씠?곕? 寃곗륫 ?곗씠?곕줈 蹂닿퀬 ?쒖쇅"},

        {"遺꾨쪟": "?쇰꺼留?, "?⑹뼱": "age_now", "?섎?": "?щ엺???꾩옱 ?섏씠", "?꾨줈?앺듃?먯꽌????븷": "?ъ쭊 ???섏씠媛 ?꾨땲誘濡?理쒖쥌 ?숈뒿 ?쇰꺼濡??ъ슜?섏? ?딆쓬"},
        {"遺꾨쪟": "?쇰꺼留?, "?⑹뼱": "age_past", "?섎?": "?ъ쭊??珥ъ쁺???뱀떆???섏씠", "?꾨줈?앺듃?먯꽌????븷": "?쇨뎬 ?대?吏???섑????ㅼ젣 ?섏씠 湲곗??쇰줈 ?ъ슜??理쒖쥌 ?쇰꺼"},
        {"遺꾨쪟": "?쇰꺼留?, "?⑹뼱": "label_source", "?섎?": "?대뼡 媛믪쓣 湲곗??쇰줈 ?쇰꺼??留뚮뱾?덈뒗吏 ?쒖떆?섎뒗 而щ읆", "?꾨줈?앺듃?먯꽌????븷": "age_past 湲곗? ?쇰꺼?꾩쓣 湲곕줉"},
        {"遺꾨쪟": "?쇰꺼留?, "?⑹뼱": "?대옒??, "?섎?": "紐⑤뜽???덉륫?댁빞 ?섎뒗 ?뺣떟 踰붿＜", "?꾨줈?앺듃?먯꽌????븷": "?섏씠瑜?5媛??섏씠? 洹몃９?쇰줈 遺꾨쪟"},
        {"遺꾨쪟": "?쇰꺼留?, "?⑹뼱": "age_group", "?섎?": "?섏씠?瑜??レ옄 ?대옒?ㅻ줈 蹂?섑븳 媛?, "?꾨줈?앺듃?먯꽌????븷": "0~4 ?쇰꺼濡?紐⑤뜽 ?숈뒿???ъ슜"},
        {"遺꾨쪟": "?쇰꺼留?, "?⑹뼱": "age_group_name", "?섎?": "?섏씠? ?대옒???대쫫", "?꾨줈?앺듃?먯꽌????븷": "10? ?댄븯, 20?, 30?, 40?, 50? ?댁긽?쇰줈 ?쒖떆"},
        {"遺꾨쪟": "?쇰꺼留?, "?⑹뼱": "5媛??대옒??, "?섎?": "?섏씠?瑜?5媛?洹몃９?쇰줈 ?섎늿 遺꾨쪟 湲곗?", "?꾨줈?앺듃?먯꽌????븷": "0=10? ?댄븯, 1=20?, 2=30?, 3=40?, 4=50? ?댁긽"},
        {"遺꾨쪟": "?쇰꺼留?, "?⑹뼱": "?대옒??遺꾪룷", "?섎?": "媛??대옒?ㅼ뿉 ?곗씠?곌? 紐?媛쒖뵫 ?덈뒗吏 ?섑???媛?, "?꾨줈?앺듃?먯꽌????븷": "10? ?댄븯 ?곗씠?곌? 留롪퀬 40?, 50? ?댁긽 ?곗씠?곌? ?곸? 寃껋쓣 ?뺤씤"},
        {"遺꾨쪟": "?쇰꺼留?, "?⑹뼱": "?대옒??遺덇퇏??, "?섎?": "?뱀젙 ?대옒???곗씠?곌? ?ㅻⅨ ?대옒?ㅻ낫??留롪굅???곸? ?곹깭", "?꾨줈?앺듃?먯꽌????븷": "class_weights瑜??곸슜???댁쑀"},
        {"遺꾨쪟": "?쇰꺼留?, "?⑹뼱": "class_weights", "?섎?": "?대옒??遺덇퇏?뺤쓣 蹂댁젙?섍린 ?꾪븳 媛以묒튂", "?꾨줈?앺듃?먯꽌????븷": "?곗씠?곌? ?곸? ?대옒?ㅼ뿉 ?????먯떎 媛以묒튂瑜?遺??},

        {"遺꾨쪟": "?곗씠??遺꾨━", "?⑹뼱": "Training ?곗씠??, "?섎?": "紐⑤뜽???ㅼ젣濡??숈뒿?섎뒗 ?곗씠??, "?꾨줈?앺듃?먯꽌????븷": "紐⑤뜽 媛以묒튂瑜??낅뜲?댄듃?섎뒗 ???ъ슜"},
        {"遺꾨쪟": "?곗씠??遺꾨━", "?⑹뼱": "Validation ?곗씠??, "?섎?": "?숈뒿 以?紐⑤뜽 ?깅뒫???뺤씤?섎뒗 ?곗씠??, "?꾨줈?앺듃?먯꽌????븷": "紐⑤뜽???덈줈???곗씠?곗뿉 ?쇰쭏????留욌뒗吏 ?뺤씤"},
        {"遺꾨쪟": "?곗씠??遺꾨━", "?⑹뼱": "Train / Validation 遺꾨━", "?섎?": "?꾩껜 ?곗씠?곕? ?숈뒿?⑷낵 寃利앹슜?쇰줈 ?섎늻??怨쇱젙", "?꾨줈?앺듃?먯꽌????븷": "Training ?대뜑? Validation ?대뜑 湲곗??쇰줈 遺꾨━"},
        {"遺꾨쪟": "?곗씠??遺꾨━", "?⑹뼱": "phase", "?섎?": "?곗씠?곌? Training?몄? Validation?몄? ?쒖떆?섎뒗 而щ읆", "?꾨줈?앺듃?먯꽌????븷": "image_path?먯꽌 Training/Validation???먮퀎???앹꽦"},

        {"遺꾨쪟": "?대?吏 ?꾩쿂由?, "?⑹뼱": "?꾩쿂由?, "?섎?": "紐⑤뜽???ｊ린 ?꾩뿉 ?곗씠?곕? ?숈뒿 媛?ν븳 ?뺥깭濡?諛붽씀??怨쇱젙", "?꾨줈?앺듃?먯꽌????븷": "?대?吏瑜?224x224, Tensor, Normalize ?뺥깭濡?蹂??},
        {"遺꾨쪟": "?대?吏 ?꾩쿂由?, "?⑹뼱": "PIL Image", "?섎?": "Python?먯꽌 ?대?吏瑜??닿퀬 泥섎━?섎뒗 ?대?吏 媛앹껜", "?꾨줈?앺듃?먯꽌????븷": "Image.open()?쇰줈 ?대?吏 ?뚯씪???쎈뒗 ???ъ슜"},
        {"遺꾨쪟": "?대?吏 ?꾩쿂由?, "?⑹뼱": "RGB 蹂??, "?섎?": "?대?吏瑜?Red, Green, Blue 3梨꾨꼸濡?留욎텛??怨쇱젙", "?꾨줈?앺듃?먯꽌????븷": "紐⑤뜽 ?낅젰 梨꾨꼸??3梨꾨꼸濡??듭씪"},
        {"遺꾨쪟": "?대?吏 ?꾩쿂由?, "?⑹뼱": "Resize", "?섎?": "?대?吏 ?ш린瑜??쇱젙???ш린濡?諛붽씀??怨쇱젙", "?꾨줈?앺듃?먯꽌????븷": "紐⑤뱺 ?대?吏瑜?224x224濡?蹂??},
        {"遺꾨쪟": "?대?吏 ?꾩쿂由?, "?⑹뼱": "Tensor", "?섎?": "PyTorch 紐⑤뜽??怨꾩궛?????덈뒗 ?レ옄 諛곗뿴", "?꾨줈?앺듃?먯꽌????븷": "?대?吏瑜?紐⑤뜽 ?낅젰 ?뺥깭濡?蹂??},
        {"遺꾨쪟": "?대?吏 ?꾩쿂由?, "?⑹뼱": "Normalize", "?섎?": "?쎌? 媛믪쓣 ?됯퇏怨??쒖??몄감 湲곗??쇰줈 ?뺢퇋?뷀븯??怨쇱젙", "?꾨줈?앺듃?먯꽌????븷": "ImageNet pretrained 紐⑤뜽 湲곗? mean/std ?곸슜"},
        {"遺꾨쪟": "?대?吏 ?꾩쿂由?, "?⑹뼱": "Data Augmentation", "?섎?": "?대?吏??蹂?뺤쓣 二쇱뼱 ?곗씠???ㅼ뼇?깆쓣 ?섎━??諛⑸쾿", "?꾨줈?앺듃?먯꽌????븷": "醫뚯슦 諛섏쟾, ?뚯쟾???ъ슜???쇰컲???깅뒫 ?μ긽 ?쒕룄"},
        {"遺꾨쪟": "?대?吏 ?꾩쿂由?, "?⑹뼱": "RandomHorizontalFlip", "?섎?": "?대?吏瑜??쇱젙 ?뺣쪧濡?醫뚯슦 諛섏쟾?섎뒗 ?꾩쿂由?, "?꾨줈?앺듃?먯꽌????븷": "?쇨뎬 諛⑺뼢 蹂?붿뿉 ??묓븯?꾨줉 ?숈뒿"},
        {"遺꾨쪟": "?대?吏 ?꾩쿂由?, "?⑹뼱": "RandomRotation", "?섎?": "?대?吏瑜??쇱젙 媛곷룄 踰붿쐞?먯꽌 ?뚯쟾?섎뒗 ?꾩쿂由?, "?꾨줈?앺듃?먯꽌????븷": "珥ъ쁺 媛곷룄 蹂?붿뿉 ??묓븯?꾨줉 ?숈뒿"},

        {"遺꾨쪟": "?숈뒿 援ъ“", "?⑹뼱": "Dataset", "?섎?": "?대?吏? ?쇰꺼???섎굹??諛섑솚?섎뒗 PyTorch ?곗씠??援ъ“", "?꾨줈?앺듃?먯꽌????븷": "CSV?먯꽌 image_path? age_group???쎌뼱 ?대?吏? ?쇰꺼 諛섑솚"},
        {"遺꾨쪟": "?숈뒿 援ъ“", "?⑹뼱": "DataLoader", "?섎?": "Dataset??batch ?⑥쐞濡?臾띠뼱二쇰뒗 ?꾧뎄", "?꾨줈?앺듃?먯꽌????븷": "?숈뒿 ?곗씠?곕? batch ?⑥쐞濡?紐⑤뜽???꾨떖"},
        {"遺꾨쪟": "?숈뒿 援ъ“", "?⑹뼱": "Batch Size", "?섎?": "??踰덉뿉 紐⑤뜽???ｋ뒗 ?대?吏 媛쒖닔", "?꾨줈?앺듃?먯꽌????븷": "OOM 臾몄젣濡?理쒖쥌 64 ?ъ슜"},
        {"遺꾨쪟": "?숈뒿 援ъ“", "?⑹뼱": "Epoch", "?섎?": "?꾩껜 ?숈뒿 ?곗씠?곕? ??踰?紐⑤몢 ?숈뒿???잛닔", "?꾨줈?앺듃?먯꽌????븷": "理쒕? 15 epoch ?ㅼ젙"},
        {"遺꾨쪟": "?숈뒿 援ъ“", "?⑹뼱": "Forward", "?섎?": "?낅젰 ?대?吏瑜?紐⑤뜽???ｌ뼱 ?덉륫媛믪쓣 怨꾩궛?섎뒗 怨쇱젙", "?꾨줈?앺듃?먯꽌????븷": "outputs = model(images)"},
        {"遺꾨쪟": "?숈뒿 援ъ“", "?⑹뼱": "Backward", "?섎?": "?먯떎??湲곗??쇰줈 媛以묒튂 ?섏젙 諛⑺뼢??怨꾩궛?섎뒗 怨쇱젙", "?꾨줈?앺듃?먯꽌????븷": "loss.backward()"},
        {"遺꾨쪟": "?숈뒿 援ъ“", "?⑹뼱": "Loss", "?섎?": "?덉륫媛믨낵 ?뺣떟??李⑥씠瑜??섑??대뒗 媛?, "?꾨줈?앺듃?먯꽌????븷": "紐⑤뜽 ?숈뒿??湲곗?媛?},
        {"遺꾨쪟": "?숈뒿 援ъ“", "?⑹뼱": "Optimizer", "?섎?": "紐⑤뜽 媛以묒튂瑜??낅뜲?댄듃?섎뒗 ?뚭퀬由ъ쬁", "?꾨줈?앺듃?먯꽌????븷": "Adam Optimizer ?ъ슜"},
        {"遺꾨쪟": "?숈뒿 援ъ“", "?⑹뼱": "Adam", "?섎?": "?λ윭?앹뿉???먯＜ ?곕뒗 理쒖쟻???뚭퀬由ъ쬁", "?꾨줈?앺듃?먯꽌????븷": "lr=0.0001濡?紐⑤뜽 媛以묒튂 ?낅뜲?댄듃"},
        {"遺꾨쪟": "?숈뒿 援ъ“", "?⑹뼱": "Learning Rate", "?섎?": "媛以묒튂瑜??쇰쭏???ш쾶 ?섏젙?좎? ?뺥븯??媛?, "?꾨줈?앺듃?먯꽌????븷": "0.0001 ?ъ슜"},
        {"遺꾨쪟": "?숈뒿 援ъ“", "?⑹뼱": "CrossEntropyLoss", "?섎?": "?ㅼ쨷 ?대옒??遺꾨쪟?먯꽌 ?ъ슜?섎뒗 ?먯떎 ?⑥닔", "?꾨줈?앺듃?먯꽌????븷": "5媛??섏씠? 遺꾨쪟 ?먯떎 怨꾩궛"},

        {"遺꾨쪟": "紐⑤뜽", "?⑹뼱": "CNN", "?섎?": "?대?吏 ?뱀쭠 異붿텧??媛뺥븳 ?λ윭??援ъ“", "?꾨줈?앺듃?먯꽌????븷": "EfficientNet-B0, MobileNetV2??湲곕컲 援ъ“"},
        {"遺꾨쪟": "紐⑤뜽", "?⑹뼱": "EfficientNet-B0", "?섎?": "?깅뒫怨?怨꾩궛?됱쓽 洹좏삎??醫뗭? ?대?吏 遺꾨쪟 紐⑤뜽", "?꾨줈?앺듃?먯꽌????븷": "理쒖쥌 ?좏깮 紐⑤뜽"},
        {"遺꾨쪟": "紐⑤뜽", "?⑹뼱": "MobileNetV2", "?섎?": "媛蹂띻퀬 鍮좊Ⅸ ?대?吏 遺꾨쪟 紐⑤뜽", "?꾨줈?앺듃?먯꽌????븷": "2李?鍮꾧탳 紐⑤뜽"},
        {"遺꾨쪟": "紐⑤뜽", "?⑹뼱": "Pretrained Model", "?섎?": "?洹쒕え ?대?吏 ?곗씠?곕줈 誘몃━ ?숈뒿??紐⑤뜽", "?꾨줈?앺듃?먯꽌????븷": "泥섏쓬遺???숈뒿?섏? ?딄퀬 ?대?吏 ?뱀쭠???쒖슜"},
        {"遺꾨쪟": "紐⑤뜽", "?⑹뼱": "Classifier", "?섎?": "紐⑤뜽??留덉?留?遺꾨쪟湲?遺遺?, "?꾨줈?앺듃?먯꽌????븷": "1000媛?異쒕젰?먯꽌 5媛?異쒕젰?쇰줈 援먯껜"},
        {"遺꾨쪟": "紐⑤뜽", "?⑹뼱": "Linear Layer", "?섎?": "?낅젰 ?뱀쭠??理쒖쥌 ?대옒???먯닔濡?諛붽씀??痢?, "?꾨줈?앺듃?먯꽌????븷": "EfficientNet-B0 留덉?留??덉씠?대? 5?대옒?ㅻ줈 蹂寃?},
        {"遺꾨쪟": "紐⑤뜽", "?⑹뼱": "Checkpoint", "?섎?": "?숈뒿 以???ν븳 紐⑤뜽 ?곹깭 ?뚯씪", "?꾨줈?앺듃?먯꽌????븷": "best 紐⑤뜽, last 紐⑤뜽 ???},
        {"遺꾨쪟": "紐⑤뜽", "?⑹뼱": "state_dict", "?섎?": "PyTorch 紐⑤뜽??媛以묒튂 ?뺣낫", "?꾨줈?앺듃?먯꽌????븷": "?뱀빋?먯꽌 best 紐⑤뜽 媛以묒튂 濡쒕뱶"},
        {"遺꾨쪟": "紐⑤뜽", "?⑹뼱": "Best Model", "?섎?": "寃利??깅뒫??媛??醫뗭븯???쒖젏??紐⑤뜽", "?꾨줈?앺듃?먯꽌????븷": "?뱀빋 理쒖쥌 ?ъ슜 紐⑤뜽"},
        {"遺꾨쪟": "紐⑤뜽", "?⑹뼱": "Last Model", "?섎?": "?숈뒿 留덉?留??쒖젏??紐⑤뜽", "?꾨줈?앺듃?먯꽌????븷": "best 紐⑤뜽怨?鍮꾧탳?⑹쑝濡????},

        {"遺꾨쪟": "?깅뒫 ?됯?", "?⑹뼱": "Train Loss", "?섎?": "?숈뒿 ?곗씠?곗뿉??怨꾩궛???먯떎媛?, "?꾨줈?앺듃?먯꽌????븷": "?숈뒿 ?곗씠?곗뿉 ?쇰쭏????留욌뒗吏 ?뺤씤"},
        {"遺꾨쪟": "?깅뒫 ?됯?", "?⑹뼱": "Validation Loss", "?섎?": "寃利??곗씠?곗뿉??怨꾩궛???먯떎媛?, "?꾨줈?앺듃?먯꽌????븷": "Best Model ?좏깮 湲곗?"},
        {"遺꾨쪟": "?깅뒫 ?됯?", "?⑹뼱": "Train Accuracy", "?섎?": "?숈뒿 ?곗씠???뺣떟瑜?, "?꾨줈?앺듃?먯꽌????븷": "?숈뒿 吏꾪뻾 ?뺣룄 ?뺤씤"},
        {"遺꾨쪟": "?깅뒫 ?됯?", "?⑹뼱": "Validation Accuracy", "?섎?": "寃利??곗씠???뺣떟瑜?, "?꾨줈?앺듃?먯꽌????븷": "紐⑤뜽 ?깅뒫 鍮꾧탳 湲곗?"},
        {"遺꾨쪟": "?깅뒫 ?됯?", "?⑹뼱": "Best Val Loss", "?섎?": "媛????븯??Validation Loss", "?꾨줈?앺듃?먯꽌????븷": "EfficientNet-B0 best 紐⑤뜽 ???湲곗?"},
        {"遺꾨쪟": "?깅뒫 ?됯?", "?⑹뼱": "Best Val Acc", "?섎?": "媛???믪븯??Validation Accuracy", "?꾨줈?앺듃?먯꽌????븷": "紐⑤뜽 ?깅뒫 鍮꾧탳 吏??},
        {"遺꾨쪟": "?깅뒫 ?됯?", "?⑹뼱": "Confusion Matrix", "?섎?": "?ㅼ젣 ?쇰꺼怨??덉륫 ?쇰꺼??鍮꾧탳?섎뒗 ??, "?꾨줈?앺듃?먯꽌????븷": "?대뼡 ?섏씠?瑜??룰컝由щ뒗吏 ?뺤씤"},
        {"遺꾨쪟": "?깅뒫 ?됯?", "?⑹뼱": "True Label", "?섎?": "?ㅼ젣 ?뺣떟 ?쇰꺼", "?꾨줈?앺듃?먯꽌????븷": "Confusion Matrix???ㅼ젣 ?대옒??},
        {"遺꾨쪟": "?깅뒫 ?됯?", "?⑹뼱": "Predicted Label", "?섎?": "紐⑤뜽???덉륫???쇰꺼", "?꾨줈?앺듃?먯꽌????븷": "Confusion Matrix???덉륫 ?대옒??},
        {"遺꾨쪟": "?깅뒫 ?됯?", "?⑹뼱": "Overfitting", "?섎?": "?숈뒿 ?곗씠?곗뿉????留욎?留?寃利??곗씠???깅뒫???⑥뼱吏???꾩긽", "?꾨줈?앺듃?먯꽌????븷": "EfficientNet-B0 ?꾨컲遺?먯꽌 Val Loss 利앷?濡??뺤씤"},
        {"遺꾨쪟": "?깅뒫 ?됯?", "?⑹뼱": "EarlyStopping", "?섎?": "寃利??깅뒫 媛쒖꽑???놁쑝硫??숈뒿???먮룞 以묐떒?섎뒗 諛⑸쾿", "?꾨줈?앺듃?먯꽌????븷": "Val Loss媛 5踰??곗냽 媛쒖꽑?섏? ?딆쑝硫?以묐떒"},

        {"遺꾨쪟": "洹몃옒??, "?⑹뼱": "History CSV", "?섎?": "epoch蹂??숈뒿 寃곌낵瑜???ν븳 CSV", "?꾨줈?앺듃?먯꽌????븷": "?숈뒿 洹몃옒???앹꽦???ъ슜"},
        {"遺꾨쪟": "洹몃옒??, "?⑹뼱": "Line Graph", "?섎?": "媛믪쓽 蹂?붾? ?좎쑝濡?蹂댁뿬二쇰뒗 洹몃옒??, "?꾨줈?앺듃?먯꽌????븷": "epoch蹂?Loss, Accuracy 蹂???쒓컖??},
        {"遺꾨쪟": "洹몃옒??, "?⑹뼱": "Bar Graph", "?섎?": "媛믪쓣 留됰?濡?鍮꾧탳?섎뒗 洹몃옒??, "?꾨줈?앺듃?먯꽌????븷": "紐⑤뜽蹂?Best Val Acc, Best Val Loss, ?숈뒿 ?쒓컙 鍮꾧탳"},
        {"遺꾨쪟": "洹몃옒??, "?⑹뼱": "Validation Accuracy 洹몃옒??, "?섎?": "寃利??뺥솗?꾩쓽 蹂?붾? 蹂댁뿬二쇰뒗 洹몃옒??, "?꾨줈?앺듃?먯꽌????븷": "紐⑤뜽 ?깅뒫 蹂???뺤씤"},
        {"遺꾨쪟": "洹몃옒??, "?⑹뼱": "Validation Loss 洹몃옒??, "?섎?": "寃利??먯떎??蹂?붾? 蹂댁뿬二쇰뒗 洹몃옒??, "?꾨줈?앺듃?먯꽌????븷": "怨쇱쟻???щ? ?뺤씤"},
        {"遺꾨쪟": "洹몃옒??, "?⑹뼱": "Train vs Validation 洹몃옒??, "?섎?": "?숈뒿 ?깅뒫怨?寃利??깅뒫???④퍡 鍮꾧탳?섎뒗 洹몃옒??, "?꾨줈?앺듃?먯꽌????븷": "?쇰컲???깅뒫怨?怨쇱쟻???뺤씤"},

        {"遺꾨쪟": "?섍꼍/?ㅻ쪟", "?⑹뼱": "CUDA", "?섎?": "NVIDIA GPU瑜??댁슜???λ윭???곗궛??鍮좊Ⅴ寃?泥섎━?섎뒗 湲곗닠", "?꾨줈?앺듃?먯꽌????븷": "RTX 4060 GPU ?숈뒿???ъ슜"},
        {"遺꾨쪟": "?섍꼍/?ㅻ쪟", "?⑹뼱": "GPU", "?섎?": "?λ윭???됰젹 ?곗궛??鍮좊Ⅴ寃?泥섎━?섎뒗 ?μ튂", "?꾨줈?앺듃?먯꽌????븷": "紐⑤뜽 ?숈뒿 媛??},
        {"遺꾨쪟": "?섍꼍/?ㅻ쪟", "?⑹뼱": "CPU", "?섎?": "?쇰컲 ?곗궛怨??곗씠??濡쒕뵫???대떦?섎뒗 ?μ튂", "?꾨줈?앺듃?먯꽌????븷": "?대?吏 ?뚯씪 濡쒕뵫怨??꾩쿂由??쇰? ?대떦"},
        {"遺꾨쪟": "?섍꼍/?ㅻ쪟", "?⑹뼱": "OutOfMemoryError", "?섎?": "GPU 硫붾え由ш? 遺議깊븷 ??諛쒖깮?섎뒗 ?ㅻ쪟", "?꾨줈?앺듃?먯꽌????븷": "batch size 160?먯꽌 諛쒖깮??64濡?以꾩엫"},
        {"遺꾨쪟": "?섍꼍/?ㅻ쪟", "?⑹뼱": "DecompressionBombWarning", "?섎?": "PIL??留ㅼ슦 ???대?吏瑜??????꾩슦??寃쎄퀬", "?꾨줈?앺듃?먯꽌????븷": "?쇰? ???AI Hub ?대?吏?먯꽌 諛쒖깮"},
        {"遺꾨쪟": "?섍꼍/?ㅻ쪟", "?⑹뼱": "requirements.txt", "?섎?": "?ㅽ뻾???꾩슂??Python ?⑦궎吏 紐⑸줉 ?뚯씪", "?꾨줈?앺듃?먯꽌????븷": "Streamlit ?뱀빋 ?ㅽ뻾 諛?諛고룷 以鍮?},

        {"遺꾨쪟": "?뱀빋", "?⑹뼱": "Streamlit", "?섎?": "Python?쇰줈 ?뱀빋??留뚮뱶???꾧뎄", "?꾨줈?앺듃?먯꽌????븷": "?섏씠? 遺꾩꽍 ?뱀빋 ?쒖옉"},
        {"遺꾨쪟": "?뱀빋", "?⑹뼱": "app.py", "?섎?": "Streamlit ?뱀빋 ?ㅽ뻾 ?뚯씪", "?꾨줈?앺듃?먯꽌????븷": "?뱀틺 珥ъ쁺, 紐⑤뜽 濡쒕뱶, ?덉륫 寃곌낵 異쒕젰"},
        {"遺꾨쪟": "?뱀빋", "?⑹뼱": "st.camera_input", "?섎?": "Streamlit???뱀틺 珥ъ쁺 湲곕뒫", "?꾨줈?앺듃?먯꽌????븷": "?뱀틺?쇰줈 ?쇨뎬 ?대?吏 珥ъ쁺"},
        {"遺꾨쪟": "?뱀빋", "?⑹뼱": "Inference", "?섎?": "?숈뒿??紐⑤뜽濡??덈줈???곗씠?곕? ?덉륫?섎뒗 怨쇱젙", "?꾨줈?앺듃?먯꽌????븷": "珥ъ쁺 ?대?吏瑜??섏씠?濡?遺꾨쪟"},
        {"遺꾨쪟": "?뱀빋", "?⑹뼱": "Softmax", "?섎?": "紐⑤뜽 異쒕젰媛믪쓣 ?대옒?ㅻ퀎 ?뺣쪧濡?蹂?섑븯???⑥닔", "?꾨줈?앺듃?먯꽌????븷": "5媛??섏씠?蹂??덉륫 ?뺣쪧 怨꾩궛"},
        {"遺꾨쪟": "?뱀빋", "?⑹뼱": "Confidence", "?섎?": "紐⑤뜽??理쒖쥌 ?덉륫??????쇰쭏???뺤떊?섎뒗吏 ?섑??대뒗 媛?, "?꾨줈?앺듃?먯꽌????븷": "?뱀빋?먯꽌 ?덉륫 ?좊ː?꾨줈 ?쒖떆"}
    ]

    terms_df = pd.DataFrame(terms_data)

    glossary_csv_path = CSV_DIR / "project_terms_glossary.csv"

    terms_df.to_csv(glossary_csv_path, index=False, encoding="utf-8-sig")

    st.markdown(
        """
        <style>
        .term-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 16px;
            padding: 18px 18px 16px 18px;
            margin-bottom: 16px;
            background: rgba(250, 250, 250, 0.04);
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            min-height: 220px;
        }
        .term-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            background: rgba(80, 120, 255, 0.14);
            margin-bottom: 10px;
        }
        .term-title {
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .term-label {
            font-weight: 800;
            margin-top: 8px;
            margin-bottom: 2px;
        }
        .term-text {
            font-size: 15px;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("?꾩껜 ?⑹뼱 ??, len(terms_df))

    with col2:
        st.metric("遺꾨쪟 ??, terms_df["遺꾨쪟"].nunique())

    with col3:
        st.metric("CSV ???, "?꾨즺")

    st.caption(f"?⑹뼱 ?뺣━ CSV ???寃쎈줈: {glossary_csv_path}")

    st.divider()

    st.subheader("?⑹뼱 寃??諛??꾪꽣")

    category_options = sorted(terms_df["遺꾨쪟"].unique())

    selected_categories = st.multiselect(
        "遺꾨쪟 ?좏깮",
        options=category_options,
        default=category_options
    )

    keyword = st.text_input(
        "寃?됱뼱 ?낅젰",
        placeholder="?? EarlyStopping, age_past, Validation Loss"
    )

    filtered_df = terms_df[terms_df["遺꾨쪟"].isin(selected_categories)].copy()

    if keyword.strip():
        keyword_lower = keyword.lower()

        filtered_df = filtered_df[
            filtered_df["遺꾨쪟"].str.lower().str.contains(keyword_lower)
            | filtered_df["?⑹뼱"].str.lower().str.contains(keyword_lower)
            | filtered_df["?섎?"].str.lower().str.contains(keyword_lower)
            | filtered_df["?꾨줈?앺듃?먯꽌????븷"].str.lower().str.contains(keyword_lower)
        ]

    st.divider()

    st.subheader("?⑹뼱 移대뱶 紐⑸줉")

    if len(filtered_df) == 0:
        st.warning("寃??議곌굔??留욌뒗 ?⑹뼱媛 ?놁뒿?덈떎.")
    else:
        for start_idx in range(0, len(filtered_df), 2):
            card_cols = st.columns(2)

            for col_idx in range(2):
                row_idx = start_idx + col_idx

                if row_idx >= len(filtered_df):
                    break

                row = filtered_df.iloc[row_idx]

                category = html.escape(str(row["遺꾨쪟"]))
                term = html.escape(str(row["?⑹뼱"]))
                meaning = html.escape(str(row["?섎?"]))
                role = html.escape(str(row["?꾨줈?앺듃?먯꽌????븷"]))

                card_html = f"""
                <div class="term-card">
                    <div class="term-badge">{category}</div>
                    <div class="term-title">{term}</div>
                    <div class="term-label">?섎?</div>
                    <div class="term-text">{meaning}</div>
                    <div class="term-label">?꾨줈?앺듃?먯꽌????븷</div>
                    <div class="term-text">{role}</div>
                </div>
                """

                with card_cols[col_idx]:
                    st.markdown(card_html, unsafe_allow_html=True)

    csv_data = terms_df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="?뱿 ?⑹뼱 ?뺣━ CSV ?ㅼ슫濡쒕뱶",
        data=csv_data,
        file_name="project_terms_glossary.csv",
        mime="text/csv"
    )

    st.success("移대뱶 ?뺤떇 ?⑹뼱 ?뺣━ ?섏씠吏 ?앹꽦 ?꾨즺")
