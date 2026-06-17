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
# Streamlit 기본 설정
# =========================================================

st.set_page_config(
    page_title="얼굴 이미지 기반 나이대 분석 웹앱",
    page_icon="📷",
    layout="wide"
)

# =========================================================
# 경로 설정
# =========================================================

BASE_DIR = Path("C:/Users/USER/Desktop/new_face")

MODEL_PATH = BASE_DIR / "outputs" / "models" / "efficientnet_b0_age_past_5class_best.pth"
MODEL_INFO_CSV_PATH = BASE_DIR / "outputs" / "csv" / "final_selected_model_efficientnet_b0.csv"

CSV_DIR = BASE_DIR / "outputs" / "csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"
MODEL_DIR = BASE_DIR / "outputs" / "models"

CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 클래스 정보
# =========================================================

CLASS_NAME_MAP = {
    0: "10대 이하",
    1: "20대",
    2: "30대",
    3: "40대",
    4: "50대 이상"
}

NUM_CLASSES = 5

# =========================================================
# 공통 함수
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
        st.warning(f"CSV 파일을 찾을 수 없습니다: {csv_path}")


def show_image_if_exists(title, image_path):
    st.markdown(f"### {title}")

    if image_path.exists():
        st.caption(str(image_path))
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning(f"이미지 파일을 찾을 수 없습니다: {image_path}")


# =========================================================
# 모델 생성 함수
# =========================================================

def create_efficientnet_b0_model(num_classes=5):
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# =========================================================
# 모델 로드 함수
# =========================================================

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

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
# 이미지 전처리 함수
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
# 예측 함수
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
# 사이드바
# =========================================================

st.sidebar.title("📌 메뉴")

page = st.sidebar.radio(
    "페이지 선택",
    [
        "📷 나이대 분석",
        "📊 시각화",
        "📘 용어 정리"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 최종 선택 모델")
st.sidebar.success("EfficientNet-B0")

st.sidebar.markdown("### 나이대 클래스")
st.sidebar.write("0 = 10대 이하")
st.sidebar.write("1 = 20대")
st.sidebar.write("2 = 30대")
st.sidebar.write("3 = 40대")
st.sidebar.write("4 = 50대 이상")


# =========================================================
# 1페이지. 나이대 분석
# =========================================================

if page == "📷 나이대 분석":

    st.title("📷 얼굴 이미지 기반 나이대 분석 웹앱")

    st.write(
        """
        웹캠으로 얼굴 이미지를 촬영한 뒤,  
        최종 선택 모델인 **EfficientNet-B0 best 모델**을 이용해 나이대를 예측합니다.
        """
    )

    with st.expander("📌 모델 파일 및 CSV 경로 확인"):
        st.write("프로젝트 기준 폴더:", str(BASE_DIR))
        st.write("모델 파일 경로:", str(MODEL_PATH))
        st.write("모델 파일 존재 여부:", MODEL_PATH.exists())
        st.write("모델 정보 CSV 경로:", str(MODEL_INFO_CSV_PATH))
        st.write("모델 정보 CSV 존재 여부:", MODEL_INFO_CSV_PATH.exists())

    if not MODEL_PATH.exists():
        st.error(f"모델 파일을 찾을 수 없습니다:\n\n{MODEL_PATH}")
        st.stop()

    if MODEL_INFO_CSV_PATH.exists():
        model_info_df = pd.read_csv(MODEL_INFO_CSV_PATH)

        with st.expander("📄 최종 선택 모델 CSV 정보"):
            st.dataframe(model_info_df, use_container_width=True)
    else:
        st.warning("최종 선택 모델 CSV 파일을 찾지 못했습니다. 모델 파일이 있으면 분석은 가능합니다.")

    try:
        model, device, checkpoint = load_model()
        st.success("✅ EfficientNet-B0 best 모델 로드 완료")
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생:\n\n{e}")
        st.stop()

    with st.expander("🧠 Best 모델 checkpoint 정보"):
        st.write("사용 장치:", str(device))

        if isinstance(checkpoint, dict):
            st.write("Best Epoch:", checkpoint.get("epoch"))
            st.write("Best Val Loss:", checkpoint.get("best_val_loss"))
            st.write("Best Val Acc:", checkpoint.get("best_val_acc"))
            st.write("Label Source:", checkpoint.get("label_source"))
        else:
            st.write("checkpoint 형식: state_dict only")

    st.divider()

    st.subheader("1. 웹캠 촬영")

    st.info(
        "브라우저 보안상 카메라 권한 허용이 필요합니다. "
        "카메라 권한을 허용한 뒤 찰칵 버튼을 눌러 사진을 촬영하세요."
    )

    camera_col, button_col = st.columns([3, 1])

    with camera_col:
        camera_image = st.camera_input("📸 찰칵 버튼으로 얼굴 이미지 촬영")

    with button_col:
        st.write("")
        st.write("")
        st.write("")
        analyze_button = st.button("🔍 분석하기", use_container_width=True)

    st.divider()

    st.subheader("2. 분석 결과")

    if analyze_button:

        if camera_image is None:
            st.warning("먼저 웹캠으로 사진을 촬영하세요.")

        else:
            image = Image.open(BytesIO(camera_image.getvalue()))

            st.image(image, caption="촬영된 이미지", use_container_width=True)

            pred_idx, confidence, probs = predict_age_group(model, device, image)

            pred_name = CLASS_NAME_MAP[pred_idx]

            st.markdown("### ✅ 예측 결과")

            st.success(f"예측 나이대: **{pred_name}**")

            st.metric(
                label="예측 신뢰도",
                value=f"{confidence * 100:.2f}%"
            )

            prob_df = pd.DataFrame({
                "나이대": [CLASS_NAME_MAP[i] for i in range(NUM_CLASSES)],
                "확률(%)": [round(p * 100, 2) for p in probs]
            })

            st.markdown("### 클래스별 예측 확률")
            st.dataframe(prob_df, use_container_width=True)

            st.markdown("### 클래스별 확률 그래프")
            st.bar_chart(prob_df.set_index("나이대"))

            st.info(
                "이 결과는 AI 모델의 예측값입니다. "
                "실제 나이를 정확히 판단하는 공식 기준으로 사용하면 안 됩니다."
            )

    else:
        st.info("사진을 촬영한 뒤, 오른쪽의 분석하기 버튼을 눌러주세요.")


# =========================================================
# 2페이지. 시각화
# =========================================================

elif page == "📊 시각화":

    st.title("📊 학습 결과 시각화 페이지")

    st.write(
        """
        이 페이지에서는 EfficientNet-B0와 MobileNetV2의 학습 결과,  
        모델 비교 CSV, 최종 선택 모델 정보, 저장된 그래프 파일을 확인할 수 있습니다.
        """
    )

    st.divider()

    st.subheader("1. 모델 비교 요약")

    model_compare_summary_path = CSV_DIR / "model_compare_summary.csv"
    model_compare_history_path = CSV_DIR / "model_compare_history_merged.csv"
    final_selected_model_path = CSV_DIR / "final_selected_model_efficientnet_b0.csv"

    summary_df = safe_read_csv(model_compare_summary_path)

    if summary_df is not None:
        st.dataframe(summary_df, use_container_width=True)

        if "model" in summary_df.columns and "best_val_acc" in summary_df.columns:
            st.markdown("### 모델별 Best Validation Accuracy")
            acc_chart_df = summary_df[["model", "best_val_acc"]].set_index("model")
            st.bar_chart(acc_chart_df)

        if "model" in summary_df.columns and "best_val_loss" in summary_df.columns:
            st.markdown("### 모델별 Best Validation Loss")
            loss_chart_df = summary_df[["model", "best_val_loss"]].set_index("model")
            st.bar_chart(loss_chart_df)

        if "model" in summary_df.columns and "total_train_time_min" in summary_df.columns:
            st.markdown("### 모델별 총 학습 시간")
            time_chart_df = summary_df[["model", "total_train_time_min"]].set_index("model")
            st.bar_chart(time_chart_df)

    else:
        st.warning(f"모델 비교 요약 CSV를 찾을 수 없습니다: {model_compare_summary_path}")

    st.divider()

    st.subheader("2. 최종 선택 모델 정보")

    final_df = safe_read_csv(final_selected_model_path)

    if final_df is not None:
        st.dataframe(final_df, use_container_width=True)
        st.success("최종 선택 모델: EfficientNet-B0")
    else:
        st.warning(f"최종 선택 모델 CSV를 찾을 수 없습니다: {final_selected_model_path}")

    st.divider()

    st.subheader("3. 학습 기록 CSV")

    tab1, tab2, tab3 = st.tabs(
        [
            "EfficientNet-B0 History",
            "MobileNetV2 History",
            "Merged History"
        ]
    )

    with tab1:
        show_csv_if_exists(
            "EfficientNet-B0 학습 기록",
            CSV_DIR / "efficientnet_b0_age_past_5class_history.csv"
        )

    with tab2:
        show_csv_if_exists(
            "MobileNetV2 학습 기록",
            CSV_DIR / "mobilenet_v2_age_past_5class_history.csv"
        )

    with tab3:
        show_csv_if_exists(
            "모델 비교 통합 학습 기록",
            model_compare_history_path
        )

    st.divider()

    st.subheader("4. 저장된 그래프 자동 표시")

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
        st.warning("표시할 그래프 이미지 파일이 없습니다.")
    else:
        st.write(f"찾은 이미지 파일 수: {len(image_files)}개")

        image_info_df = pd.DataFrame([
            {
                "file_name": p.name,
                "folder": str(p.parent),
                "full_path": str(p),
                "size_kb": round(p.stat().st_size / 1024, 2)
            }
            for p in image_files
        ])

        with st.expander("이미지 파일 목록 보기"):
            st.dataframe(image_info_df, use_container_width=True)

        for img_path in image_files:
            st.markdown(f"### {img_path.name}")
            st.caption(str(img_path))
            st.image(str(img_path), use_container_width=True)

    st.divider()

    st.subheader("5. outputs 폴더 내 CSV 파일 목록")

    csv_files = []

    if CSV_DIR.exists():
        csv_files.extend(list(CSV_DIR.rglob("*.csv")))

    csv_files = sorted(list(set(csv_files)))

    if len(csv_files) == 0:
        st.warning("CSV 파일을 찾지 못했습니다.")
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
# 3페이지. 용어 정리
# =========================================================

elif page == "📘 용어 정리":

    st.title("📘 프로젝트 용어 정리 페이지")

    st.write(
        """
        이 페이지는 얼굴 이미지 기반 나이대 분류 프로젝트에서 사용된 용어를 카드 형식으로 정리한 페이지입니다.  
        데이터 준비, 전처리, 학습, 평가, 그래프, 웹앱 제작 과정에서 사용된 용어를 확인할 수 있습니다.
        """
    )

    st.divider()

    terms_data = [
        {"분류": "데이터 준비", "용어": "ZIP 파일", "의미": "여러 파일을 하나로 묶어 압축한 파일", "프로젝트에서의 역할": "AI Hub 원본 이미지와 라벨링 데이터를 압축 해제하기 전 형태"},
        {"분류": "데이터 준비", "용어": "압축 해제", "의미": "ZIP 파일 안의 실제 파일을 꺼내는 과정", "프로젝트에서의 역할": "이미지 파일과 JSON 라벨 파일을 실제 폴더로 풀어 학습 준비"},
        {"분류": "데이터 준비", "용어": "이미지 파일", "의미": "모델이 입력으로 받는 얼굴 사진 데이터", "프로젝트에서의 역할": "EfficientNet-B0 모델이 나이대를 예측하기 위해 학습한 입력 데이터"},
        {"분류": "데이터 준비", "용어": "JSON 라벨 파일", "의미": "이미지에 대한 정답 정보가 들어있는 파일", "프로젝트에서의 역할": "age_now, age_past, gender 등의 정보를 추출하는 데 사용"},
        {"분류": "데이터 준비", "용어": "CSV 파일", "의미": "표 형태의 데이터를 저장하는 파일", "프로젝트에서의 역할": "이미지 경로, 라벨, 학습 기록, 모델 비교 결과 저장"},
        {"분류": "데이터 준비", "용어": "file_stem", "의미": "확장자를 제외한 파일 이름", "프로젝트에서의 역할": "이미지 파일과 JSON 라벨 파일을 같은 이름 기준으로 매칭"},
        {"분류": "데이터 준비", "용어": "image_path", "의미": "이미지 파일이 저장된 전체 경로", "프로젝트에서의 역할": "Dataset이 실제 이미지를 불러올 때 사용"},
        {"분류": "데이터 준비", "용어": "json_path", "의미": "JSON 라벨 파일이 저장된 전체 경로", "프로젝트에서의 역할": "라벨 정보를 추출할 JSON 파일 위치"},
        {"분류": "데이터 준비", "용어": "이미지-JSON 매핑", "의미": "이미지 파일과 해당 JSON 라벨 파일을 연결하는 과정", "프로젝트에서의 역할": "이미지와 정답 라벨이 모두 있는 데이터만 학습에 사용하기 위해 수행"},
        {"분류": "데이터 준비", "용어": "MATCHED", "의미": "이미지와 JSON이 모두 존재하는 정상 데이터", "프로젝트에서의 역할": "최종 학습에 사용한 데이터"},
        {"분류": "데이터 준비", "용어": "MISSING_IMAGE", "의미": "JSON은 있지만 이미지가 없는 상태", "프로젝트에서의 역할": "이미지가 없으므로 학습에서 제외"},
        {"분류": "데이터 준비", "용어": "MISSING_JSON", "의미": "이미지는 있지만 JSON 라벨이 없는 상태", "프로젝트에서의 역할": "정답 라벨이 없으므로 학습에서 제외"},
        {"분류": "데이터 준비", "용어": "결측치", "의미": "필요한 값이나 파일이 빠져 있는 데이터", "프로젝트에서의 역할": "MISSING_IMAGE, MISSING_JSON 데이터를 결측 데이터로 보고 제외"},

        {"분류": "라벨링", "용어": "age_now", "의미": "사람의 현재 나이", "프로젝트에서의 역할": "사진 속 나이가 아니므로 최종 학습 라벨로 사용하지 않음"},
        {"분류": "라벨링", "용어": "age_past", "의미": "사진이 촬영된 당시의 나이", "프로젝트에서의 역할": "얼굴 이미지에 나타난 실제 나이 기준으로 사용한 최종 라벨"},
        {"분류": "라벨링", "용어": "label_source", "의미": "어떤 값을 기준으로 라벨을 만들었는지 표시하는 컬럼", "프로젝트에서의 역할": "age_past 기준 라벨임을 기록"},
        {"분류": "라벨링", "용어": "클래스", "의미": "모델이 예측해야 하는 정답 범주", "프로젝트에서의 역할": "나이를 5개 나이대 그룹으로 분류"},
        {"분류": "라벨링", "용어": "age_group", "의미": "나이대를 숫자 클래스로 변환한 값", "프로젝트에서의 역할": "0~4 라벨로 모델 학습에 사용"},
        {"분류": "라벨링", "용어": "age_group_name", "의미": "나이대 클래스 이름", "프로젝트에서의 역할": "10대 이하, 20대, 30대, 40대, 50대 이상으로 표시"},
        {"분류": "라벨링", "용어": "5개 클래스", "의미": "나이대를 5개 그룹으로 나눈 분류 기준", "프로젝트에서의 역할": "0=10대 이하, 1=20대, 2=30대, 3=40대, 4=50대 이상"},
        {"분류": "라벨링", "용어": "클래스 분포", "의미": "각 클래스에 데이터가 몇 개씩 있는지 나타낸 값", "프로젝트에서의 역할": "10대 이하 데이터가 많고 40대, 50대 이상 데이터가 적은 것을 확인"},
        {"분류": "라벨링", "용어": "클래스 불균형", "의미": "특정 클래스 데이터가 다른 클래스보다 많거나 적은 상태", "프로젝트에서의 역할": "class_weights를 적용한 이유"},
        {"분류": "라벨링", "용어": "class_weights", "의미": "클래스 불균형을 보정하기 위한 가중치", "프로젝트에서의 역할": "데이터가 적은 클래스에 더 큰 손실 가중치를 부여"},

        {"분류": "데이터 분리", "용어": "Training 데이터", "의미": "모델이 실제로 학습하는 데이터", "프로젝트에서의 역할": "모델 가중치를 업데이트하는 데 사용"},
        {"분류": "데이터 분리", "용어": "Validation 데이터", "의미": "학습 중 모델 성능을 확인하는 데이터", "프로젝트에서의 역할": "모델이 새로운 데이터에 얼마나 잘 맞는지 확인"},
        {"분류": "데이터 분리", "용어": "Train / Validation 분리", "의미": "전체 데이터를 학습용과 검증용으로 나누는 과정", "프로젝트에서의 역할": "Training 폴더와 Validation 폴더 기준으로 분리"},
        {"분류": "데이터 분리", "용어": "phase", "의미": "데이터가 Training인지 Validation인지 표시하는 컬럼", "프로젝트에서의 역할": "image_path에서 Training/Validation을 판별해 생성"},

        {"분류": "이미지 전처리", "용어": "전처리", "의미": "모델에 넣기 전에 데이터를 학습 가능한 형태로 바꾸는 과정", "프로젝트에서의 역할": "이미지를 224x224, Tensor, Normalize 형태로 변환"},
        {"분류": "이미지 전처리", "용어": "PIL Image", "의미": "Python에서 이미지를 열고 처리하는 이미지 객체", "프로젝트에서의 역할": "Image.open()으로 이미지 파일을 읽는 데 사용"},
        {"분류": "이미지 전처리", "용어": "RGB 변환", "의미": "이미지를 Red, Green, Blue 3채널로 맞추는 과정", "프로젝트에서의 역할": "모델 입력 채널을 3채널로 통일"},
        {"분류": "이미지 전처리", "용어": "Resize", "의미": "이미지 크기를 일정한 크기로 바꾸는 과정", "프로젝트에서의 역할": "모든 이미지를 224x224로 변환"},
        {"분류": "이미지 전처리", "용어": "Tensor", "의미": "PyTorch 모델이 계산할 수 있는 숫자 배열", "프로젝트에서의 역할": "이미지를 모델 입력 형태로 변환"},
        {"분류": "이미지 전처리", "용어": "Normalize", "의미": "픽셀 값을 평균과 표준편차 기준으로 정규화하는 과정", "프로젝트에서의 역할": "ImageNet pretrained 모델 기준 mean/std 적용"},
        {"분류": "이미지 전처리", "용어": "Data Augmentation", "의미": "이미지에 변형을 주어 데이터 다양성을 늘리는 방법", "프로젝트에서의 역할": "좌우 반전, 회전을 사용해 일반화 성능 향상 시도"},
        {"분류": "이미지 전처리", "용어": "RandomHorizontalFlip", "의미": "이미지를 일정 확률로 좌우 반전하는 전처리", "프로젝트에서의 역할": "얼굴 방향 변화에 대응하도록 학습"},
        {"분류": "이미지 전처리", "용어": "RandomRotation", "의미": "이미지를 일정 각도 범위에서 회전하는 전처리", "프로젝트에서의 역할": "촬영 각도 변화에 대응하도록 학습"},

        {"분류": "학습 구조", "용어": "Dataset", "의미": "이미지와 라벨을 하나씩 반환하는 PyTorch 데이터 구조", "프로젝트에서의 역할": "CSV에서 image_path와 age_group을 읽어 이미지와 라벨 반환"},
        {"분류": "학습 구조", "용어": "DataLoader", "의미": "Dataset을 batch 단위로 묶어주는 도구", "프로젝트에서의 역할": "학습 데이터를 batch 단위로 모델에 전달"},
        {"분류": "학습 구조", "용어": "Batch Size", "의미": "한 번에 모델에 넣는 이미지 개수", "프로젝트에서의 역할": "OOM 문제로 최종 64 사용"},
        {"분류": "학습 구조", "용어": "Epoch", "의미": "전체 학습 데이터를 한 번 모두 학습한 횟수", "프로젝트에서의 역할": "최대 15 epoch 설정"},
        {"분류": "학습 구조", "용어": "Forward", "의미": "입력 이미지를 모델에 넣어 예측값을 계산하는 과정", "프로젝트에서의 역할": "outputs = model(images)"},
        {"분류": "학습 구조", "용어": "Backward", "의미": "손실을 기준으로 가중치 수정 방향을 계산하는 과정", "프로젝트에서의 역할": "loss.backward()"},
        {"분류": "학습 구조", "용어": "Loss", "의미": "예측값과 정답의 차이를 나타내는 값", "프로젝트에서의 역할": "모델 학습의 기준값"},
        {"분류": "학습 구조", "용어": "Optimizer", "의미": "모델 가중치를 업데이트하는 알고리즘", "프로젝트에서의 역할": "Adam Optimizer 사용"},
        {"분류": "학습 구조", "용어": "Adam", "의미": "딥러닝에서 자주 쓰는 최적화 알고리즘", "프로젝트에서의 역할": "lr=0.0001로 모델 가중치 업데이트"},
        {"분류": "학습 구조", "용어": "Learning Rate", "의미": "가중치를 얼마나 크게 수정할지 정하는 값", "프로젝트에서의 역할": "0.0001 사용"},
        {"분류": "학습 구조", "용어": "CrossEntropyLoss", "의미": "다중 클래스 분류에서 사용하는 손실 함수", "프로젝트에서의 역할": "5개 나이대 분류 손실 계산"},

        {"분류": "모델", "용어": "CNN", "의미": "이미지 특징 추출에 강한 딥러닝 구조", "프로젝트에서의 역할": "EfficientNet-B0, MobileNetV2의 기반 구조"},
        {"분류": "모델", "용어": "EfficientNet-B0", "의미": "성능과 계산량의 균형이 좋은 이미지 분류 모델", "프로젝트에서의 역할": "최종 선택 모델"},
        {"분류": "모델", "용어": "MobileNetV2", "의미": "가볍고 빠른 이미지 분류 모델", "프로젝트에서의 역할": "2차 비교 모델"},
        {"분류": "모델", "용어": "Pretrained Model", "의미": "대규모 이미지 데이터로 미리 학습된 모델", "프로젝트에서의 역할": "처음부터 학습하지 않고 이미지 특징을 활용"},
        {"분류": "모델", "용어": "Classifier", "의미": "모델의 마지막 분류기 부분", "프로젝트에서의 역할": "1000개 출력에서 5개 출력으로 교체"},
        {"분류": "모델", "용어": "Linear Layer", "의미": "입력 특징을 최종 클래스 점수로 바꾸는 층", "프로젝트에서의 역할": "EfficientNet-B0 마지막 레이어를 5클래스로 변경"},
        {"분류": "모델", "용어": "Checkpoint", "의미": "학습 중 저장한 모델 상태 파일", "프로젝트에서의 역할": "best 모델, last 모델 저장"},
        {"분류": "모델", "용어": "state_dict", "의미": "PyTorch 모델의 가중치 정보", "프로젝트에서의 역할": "웹앱에서 best 모델 가중치 로드"},
        {"분류": "모델", "용어": "Best Model", "의미": "검증 성능이 가장 좋았던 시점의 모델", "프로젝트에서의 역할": "웹앱 최종 사용 모델"},
        {"분류": "모델", "용어": "Last Model", "의미": "학습 마지막 시점의 모델", "프로젝트에서의 역할": "best 모델과 비교용으로 저장"},

        {"분류": "성능 평가", "용어": "Train Loss", "의미": "학습 데이터에서 계산한 손실값", "프로젝트에서의 역할": "학습 데이터에 얼마나 잘 맞는지 확인"},
        {"분류": "성능 평가", "용어": "Validation Loss", "의미": "검증 데이터에서 계산한 손실값", "프로젝트에서의 역할": "Best Model 선택 기준"},
        {"분류": "성능 평가", "용어": "Train Accuracy", "의미": "학습 데이터 정답률", "프로젝트에서의 역할": "학습 진행 정도 확인"},
        {"분류": "성능 평가", "용어": "Validation Accuracy", "의미": "검증 데이터 정답률", "프로젝트에서의 역할": "모델 성능 비교 기준"},
        {"분류": "성능 평가", "용어": "Best Val Loss", "의미": "가장 낮았던 Validation Loss", "프로젝트에서의 역할": "EfficientNet-B0 best 모델 저장 기준"},
        {"분류": "성능 평가", "용어": "Best Val Acc", "의미": "가장 높았던 Validation Accuracy", "프로젝트에서의 역할": "모델 성능 비교 지표"},
        {"분류": "성능 평가", "용어": "Confusion Matrix", "의미": "실제 라벨과 예측 라벨을 비교하는 표", "프로젝트에서의 역할": "어떤 나이대를 헷갈리는지 확인"},
        {"분류": "성능 평가", "용어": "True Label", "의미": "실제 정답 라벨", "프로젝트에서의 역할": "Confusion Matrix의 실제 클래스"},
        {"분류": "성능 평가", "용어": "Predicted Label", "의미": "모델이 예측한 라벨", "프로젝트에서의 역할": "Confusion Matrix의 예측 클래스"},
        {"분류": "성능 평가", "용어": "Overfitting", "의미": "학습 데이터에는 잘 맞지만 검증 데이터 성능이 떨어지는 현상", "프로젝트에서의 역할": "EfficientNet-B0 후반부에서 Val Loss 증가로 확인"},
        {"분류": "성능 평가", "용어": "EarlyStopping", "의미": "검증 성능 개선이 없으면 학습을 자동 중단하는 방법", "프로젝트에서의 역할": "Val Loss가 5번 연속 개선되지 않으면 중단"},

        {"분류": "그래프", "용어": "History CSV", "의미": "epoch별 학습 결과를 저장한 CSV", "프로젝트에서의 역할": "학습 그래프 생성에 사용"},
        {"분류": "그래프", "용어": "Line Graph", "의미": "값의 변화를 선으로 보여주는 그래프", "프로젝트에서의 역할": "epoch별 Loss, Accuracy 변화 시각화"},
        {"분류": "그래프", "용어": "Bar Graph", "의미": "값을 막대로 비교하는 그래프", "프로젝트에서의 역할": "모델별 Best Val Acc, Best Val Loss, 학습 시간 비교"},
        {"분류": "그래프", "용어": "Validation Accuracy 그래프", "의미": "검증 정확도의 변화를 보여주는 그래프", "프로젝트에서의 역할": "모델 성능 변화 확인"},
        {"분류": "그래프", "용어": "Validation Loss 그래프", "의미": "검증 손실의 변화를 보여주는 그래프", "프로젝트에서의 역할": "과적합 여부 확인"},
        {"분류": "그래프", "용어": "Train vs Validation 그래프", "의미": "학습 성능과 검증 성능을 함께 비교하는 그래프", "프로젝트에서의 역할": "일반화 성능과 과적합 확인"},

        {"분류": "환경/오류", "용어": "CUDA", "의미": "NVIDIA GPU를 이용해 딥러닝 연산을 빠르게 처리하는 기술", "프로젝트에서의 역할": "RTX 4060 GPU 학습에 사용"},
        {"분류": "환경/오류", "용어": "GPU", "의미": "딥러닝 행렬 연산을 빠르게 처리하는 장치", "프로젝트에서의 역할": "모델 학습 가속"},
        {"분류": "환경/오류", "용어": "CPU", "의미": "일반 연산과 데이터 로딩을 담당하는 장치", "프로젝트에서의 역할": "이미지 파일 로딩과 전처리 일부 담당"},
        {"분류": "환경/오류", "용어": "OutOfMemoryError", "의미": "GPU 메모리가 부족할 때 발생하는 오류", "프로젝트에서의 역할": "batch size 160에서 발생해 64로 줄임"},
        {"분류": "환경/오류", "용어": "DecompressionBombWarning", "의미": "PIL이 매우 큰 이미지를 열 때 띄우는 경고", "프로젝트에서의 역할": "일부 대형 AI Hub 이미지에서 발생"},
        {"분류": "환경/오류", "용어": "requirements.txt", "의미": "실행에 필요한 Python 패키지 목록 파일", "프로젝트에서의 역할": "Streamlit 웹앱 실행 및 배포 준비"},

        {"분류": "웹앱", "용어": "Streamlit", "의미": "Python으로 웹앱을 만드는 도구", "프로젝트에서의 역할": "나이대 분석 웹앱 제작"},
        {"분류": "웹앱", "용어": "app.py", "의미": "Streamlit 웹앱 실행 파일", "프로젝트에서의 역할": "웹캠 촬영, 모델 로드, 예측 결과 출력"},
        {"분류": "웹앱", "용어": "st.camera_input", "의미": "Streamlit의 웹캠 촬영 기능", "프로젝트에서의 역할": "웹캠으로 얼굴 이미지 촬영"},
        {"분류": "웹앱", "용어": "Inference", "의미": "학습된 모델로 새로운 데이터를 예측하는 과정", "프로젝트에서의 역할": "촬영 이미지를 나이대로 분류"},
        {"분류": "웹앱", "용어": "Softmax", "의미": "모델 출력값을 클래스별 확률로 변환하는 함수", "프로젝트에서의 역할": "5개 나이대별 예측 확률 계산"},
        {"분류": "웹앱", "용어": "Confidence", "의미": "모델이 최종 예측에 대해 얼마나 확신하는지 나타내는 값", "프로젝트에서의 역할": "웹앱에서 예측 신뢰도로 표시"}
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
        st.metric("전체 용어 수", len(terms_df))

    with col2:
        st.metric("분류 수", terms_df["분류"].nunique())

    with col3:
        st.metric("CSV 저장", "완료")

    st.caption(f"용어 정리 CSV 저장 경로: {glossary_csv_path}")

    st.divider()

    st.subheader("용어 검색 및 필터")

    category_options = sorted(terms_df["분류"].unique())

    selected_categories = st.multiselect(
        "분류 선택",
        options=category_options,
        default=category_options
    )

    keyword = st.text_input(
        "검색어 입력",
        placeholder="예: EarlyStopping, age_past, Validation Loss"
    )

    filtered_df = terms_df[terms_df["분류"].isin(selected_categories)].copy()

    if keyword.strip():
        keyword_lower = keyword.lower()

        filtered_df = filtered_df[
            filtered_df["분류"].str.lower().str.contains(keyword_lower)
            | filtered_df["용어"].str.lower().str.contains(keyword_lower)
            | filtered_df["의미"].str.lower().str.contains(keyword_lower)
            | filtered_df["프로젝트에서의 역할"].str.lower().str.contains(keyword_lower)
        ]

    st.divider()

    st.subheader("용어 카드 목록")

    if len(filtered_df) == 0:
        st.warning("검색 조건에 맞는 용어가 없습니다.")
    else:
        for start_idx in range(0, len(filtered_df), 2):
            card_cols = st.columns(2)

            for col_idx in range(2):
                row_idx = start_idx + col_idx

                if row_idx >= len(filtered_df):
                    break

                row = filtered_df.iloc[row_idx]

                category = html.escape(str(row["분류"]))
                term = html.escape(str(row["용어"]))
                meaning = html.escape(str(row["의미"]))
                role = html.escape(str(row["프로젝트에서의 역할"]))

                card_html = f"""
                <div class="term-card">
                    <div class="term-badge">{category}</div>
                    <div class="term-title">{term}</div>
                    <div class="term-label">의미</div>
                    <div class="term-text">{meaning}</div>
                    <div class="term-label">프로젝트에서의 역할</div>
                    <div class="term-text">{role}</div>
                </div>
                """

                with card_cols[col_idx]:
                    st.markdown(card_html, unsafe_allow_html=True)

    csv_data = terms_df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="📥 용어 정리 CSV 다운로드",
        data=csv_data,
        file_name="project_terms_glossary.csv",
        mime="text/csv"
    )

    st.success("카드 형식 용어 정리 페이지 생성 완료")