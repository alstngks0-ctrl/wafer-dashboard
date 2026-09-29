import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 페이지 설정 (와이드 모드)
st.set_page_config(
    page_title="반도체 웨이퍼 불량 분석 및 수율 대시보드",
    page_icon="🔬",
    layout="wide"
)

# 1. 샘플 데이터 생성 함수 (기존 코드 응용)
def generate_sample_wafer_data(defect_mode: str, num_dies: int = 1000, wafer_radius: float = 150.0) -> pd.DataFrame:
    np.random.seed(42)
    r = np.random.uniform(0, wafer_radius, num_dies)
    theta = np.random.uniform(0, 360, num_dies)
    status = np.array(["PASS"] * num_dies, dtype=object)
    
    if defect_mode == "ring":
        indices = np.random.choice(num_dies, size=int(num_dies*0.1), replace=False)
        r[indices] = np.random.normal(wafer_radius * 0.85, 5, len(indices))
        status[indices] = "FAIL"
    elif defect_mode == "center":
        indices = np.random.choice(num_dies, size=int(num_dies*0.1), replace=False)
        r[indices] = np.random.uniform(0, wafer_radius * 0.2, len(indices))
        status[indices] = "FAIL"
    elif defect_mode == "scratch":
        indices = np.random.choice(num_dies, size=int(num_dies*0.1), replace=False)
        theta[indices] = 45.0 + np.random.normal(0, 1, len(indices))
        status[indices] = "FAIL"
    elif defect_mode == "random":
        indices = np.random.choice(num_dies, size=int(num_dies*0.1), replace=False)
        status[indices] = "FAIL"
        
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    
    return pd.DataFrame({
        "status": status,
        "radius": r,
        "angle_deg": theta,
        "die_x": x,
        "die_y": y
    })

# 2. 불량 패턴 분류 함수 (기존 5번 코드)
def classify_wafer_pattern(df: pd.DataFrame, wafer_radius: float = 150.0):
    fail_df = df[df["status"] == "FAIL"]
    total = len(df)
    fail_count = len(fail_df)
    yield_pct = (total - fail_count) / total * 100

    if fail_count == 0 or yield_pct > 98.0:
        return "CLEAN", 1.0, yield_pct, fail_count, "이상 없음. 정상 모니터링 유지."

    fail_radius_mean = fail_df["radius"].mean()
    fail_radius_std  = fail_df["radius"].std()

    angles_rad = np.radians(fail_df["angle_deg"].values)
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    angle_concentration = np.sqrt(sin_mean**2 + cos_mean**2)

    if fail_radius_mean > wafer_radius * 0.75 and fail_radius_std < 20:
        return "EDGE_RING", min(1.0, (fail_radius_mean / wafer_radius)), yield_pct, fail_count, "엣지 링 불량 감지. 증착 균일도(Uniformity) 및 Edge exclusion 재검토 권장."
    elif fail_radius_mean < wafer_radius * 0.25:
        return "CENTER", 1.0 - (fail_radius_mean / (wafer_radius * 0.25)), yield_pct, fail_count, "중심부 불량 감지. 척(Chuck) 온도 프로파일 및 중심 가스 플로우 점검 권장."
    elif angle_concentration > 0.6:
        return "SCRATCH", angle_concentration, yield_pct, fail_count, "선형 스크래치 패턴 감지. CMP 패드 상태 및 슬러리 파티클 오염 여부 확인 필요."
    else:
        return "RANDOM", 1.0 - angle_concentration, yield_pct, fail_count, "랜덤 불량 패턴. 파티클 카운트 및 노광 오차 통계 확인 필요."

# --- 대시보드 UI 레이아웃 구성 ---
st.title("🔬 반도체 웨이퍼 공정 모니터링 & AI 분류 대시보드")
st.markdown("학습동아리 기초 프로젝트를 확장한 **실시간 웹 기반 웨이퍼 분석 시스템**입니다.")

# 사이드바 설정
st.sidebar.header("⚙️ 시뮬레이션 설정")
selected_mode = st.sidebar.selectbox(
    "불량 모드 선택 (Defect Mode)",
    options=["ring", "center", "scratch", "random"],
    format_func=lambda x: {"ring": "EDGE RING", "center": "CENTER", "scratch": "SCRATCH", "random": "RANDOM"}[x]
)
num_dies = st.sidebar.slider("총 다이(Die) 개수", min_value=500, max_value=2000, value=1000, step=100)

# 데이터 생성 및 분석 수행
df = generate_sample_wafer_data(defect_mode=selected_mode, num_dies=num_dies)
pattern, confidence, yield_pct, fail_count, action_guide = classify_wafer_pattern(df)

# 메인 화면 상단 지표 카드 (Metrics)
col1, col2, col3, col4 = st.columns(4)
col1.metric("종합 수율 (Yield)", f"{yield_pct:.1f}%", delta=f"{-fail_count}개 불량" if fail_count > 0 else "정상")
col2.metric("AI 패턴 분류", pattern)
col3.metric("분류 신뢰도", f"{confidence * 100:.1f}%")
col4.metric("총 불량 다이", f"{fail_count} 개")

st.markdown("---")

# 레이아웃 분할 (좌측: 웨이퍼 맵 시각화 / 우측: 엔지니어링 조치 가이드)
left_col, right_col = st.columns([1.5, 1])

with left_col:
    st.subheader("🗺️ 실시간 웨이퍼 맵 시각화")
    
    # Matplotlib 웨이퍼 맵 그리기 (4번 코드 응용)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 웨이퍼 외곽선 (150mm 기준)
    wafer_circle = plt.Circle((0, 0), 150, color='#e0e0e0', fill=True, zorder=1)
    edge_exclusion = plt.Circle((0, 0), 147, color='blue', fill=False, linestyle='--', linewidth=1.5, zorder=2)
    ax.add_patch(wafer_circle)
    ax.add_patch(edge_exclusion)
    
    # PASS / FAIL 다이 산점도
    pass_df = df[df["status"] == "PASS"]
    fail_df = df[df["status"] == "FAIL"]
    
    ax.scatter(pass_df["die_x"], pass_df["die_y"], c='#2ecc71', s=12, label=f'PASS ({len(pass_df)})', zorder=3)
    ax.scatter(fail_df["die_x"], fail_df["die_y"], c='#e74c3c', s=25, marker='X', label=f'FAIL ({len(fail_df)})', zorder=4)
    
    ax.set_xlim(-165, 165)
    ax.set_ylim(-165, 165)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right')
    ax.set_title(f"Wafer Map [{selected_mode.upper()}]", fontsize=12, fontweight='bold')
    
    st.pyplot(fig)

with right_col:
    st.subheader("💡 팹(Fab) 엔지니어링 대응 가이드")
    
    # 패턴별 상태에 따른 색상 카드 박스
    if pattern == "CLEAN":
        st.success(f"**상태:** 정상 수율 범위\n\n{action_guide}")
    else:
        st.error(f"**경고:** 불량 패턴 감지 (`{pattern}`)\n\n{action_guide}")
        
    st.markdown("### 📊 세부 분석 통계")
    st.write(f"- **전체 다이 수:** {num_dies} 개")
    st.write(f"- **정상 다이 수:** {num_dies - fail_count} 개")
    st.write(f"- **불량률:** {(fail_count/num_dies)*100:.1f}%")
    
    with st.expander("🔍 원본 데이터 상위 5개 미리보기"):
        st.dataframe(df.head())

st.markdown("---")
st.caption("Semiconductor Yield & Defect Analysis Monitoring System powered by Streamlit.")