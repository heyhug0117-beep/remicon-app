import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="친환경 레미콘 추천 시스템",
    page_icon="🏗️",
    layout="wide"
)

@st.cache_data
def load_data():
    FILE_PATH = Path(__file__).parent / '레미콘_확정데이터_저탄소추가.xlsx'
    df = pd.read_excel(FILE_PATH)
    df = df[df['지역'].isin(['수도권', '경상도', '충청도', '제주도', '강원도', '전라도'])]
    return df.reset_index(drop=True)

CRITERIA_COLS = [
    '탄소발자국_총량', '자원발자국_총량', '산성화_총량', '부영양화_총량',
    '오존층영향_총량', '광화학스모그_총량', '물발자국_총량', '공급가격'
]
CRITERIA_LABELS = {
    '탄소발자국_총량': '탄소발자국',
    '자원발자국_총량': '자원발자국',
    '산성화_총량':    '산성화',
    '부영양화_총량':  '부영양화',
    '오존층영향_총량': '오존층영향',
    '광화학스모그_총량': '광화학스모그',
    '물발자국_총량':  '물발자국',
    '공급가격':      '공급가격(원/m³)'
}

# CRITIC 가중치 (이상값 수정 후 전체 데이터 기준 고정값)
CRITIC_WEIGHTS = np.array([
    0.118597,  # 탄소발자국  (11.86%)
    0.114689,  # 자원발자국  (11.47%)
    0.144640,  # 산성화      (14.46%)
    0.131165,  # 부영양화    (13.12%)
    0.227922,  # 오존층영향  (22.79%)
    0.090546,  # 광화학스모그  (9.05%)
    0.033750,  # 물발자국     (3.38%)
    0.138692,  # 공급가격    (13.87%)
])

def filter_data(df, region, min_strength, slump, strength_range=2, slump_range=0):
    strength_levels = sorted(df['설계기준강도'].unique())
    slump_levels    = sorted(df['슬럼프'].unique())
    i               = strength_levels.index(min_strength)
    upper_str       = strength_levels[min(i + strength_range, len(strength_levels) - 1)]
    j               = slump_levels.index(slump)
    slump_lower     = slump
    slump_upper     = slump_levels[min(j + slump_range, len(slump_levels) - 1)]
    filtered = df[
        (df['지역'] == region) &
        (df['설계기준강도'] >= min_strength) &
        (df['설계기준강도'] <= upper_str) &
        (df['슬럼프'] >= slump_lower) &
        (df['슬럼프'] <= slump_upper)
    ].copy().reset_index(drop=True)
    return filtered, upper_str, slump_lower, slump_upper

def topsis(data, cols, weights):
    X    = data[cols].values.astype(float)
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1
    Z    = X / norm
    R    = Z * weights
    V_pos = R.min(axis=0)
    V_neg = R.max(axis=0)
    S_pos = np.sqrt(((R - V_pos) ** 2).sum(axis=1))
    S_neg = np.sqrt(((R - V_neg) ** 2).sum(axis=1))
    denom = S_pos + S_neg
    denom[denom == 0] = 1e-10
    C = S_neg / denom
    result        = data.copy()
    result['근접도(Ci)'] = C
    result['순위']       = C.argsort()[::-1].argsort() + 1
    return result.sort_values('순위').reset_index(drop=True)

def main():
    st.title("🏗️ 친환경 레미콘 추천 시스템")
    st.markdown("**EPD(환경성적표지) 기반 CRITIC-TOPSIS 의사결정 지원 모델**")
    st.caption("※ 모든 환경영향 수치 및 공급가격은 레미콘 **1m³** 기준입니다.")
    st.divider()

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ 데이터 파일을 찾을 수 없습니다. FILE_PATH를 확인해주세요.")
        st.stop()

    strength_levels = sorted(df['설계기준강도'].unique().tolist())
    slump_levels    = sorted(df['슬럼프'].unique().tolist())

    with st.sidebar:
        st.header("🔍 검색 조건")
        st.info("🔨굵은골재 최대치수: **25mm** (전 제품 동일)")

        region = st.selectbox("지역",
            options=['수도권', '경상도', '충청도', '제주도', '강원도', '전라도'])

        min_strength = st.selectbox("최소 설계기준강도 (MPa)",
            options=strength_levels, index=2,
            help="입력값 이상 ~ 최대 2단계 높은 강도까지 포함하여 검색합니다.")

        slump = st.selectbox("최소 슬럼프 (mm)",
            options=slump_levels, index=3,
            help="입력값 이상 ~ +1단계까지 포함합니다.")

        top_n = st.slider("상위 몇 개 표시", min_value=5, max_value=30, value=10)

        st.divider()
        st.subheader("⚖️ 환경/공급가격 비율")
        st.caption("환경영향과 공급가격의 반영 비율을 조절합니다.")

        env_ratio = st.slider(
            "환경/공급가격 비율 (%)",
            min_value=10, max_value=90, value=50, step=1,
            help="환경영향 10~90% 범위에서 자유롭게 조절합니다. (환경 최소 10%, 가격 최소 10%)"
        )
        st.caption(f"환경 **{env_ratio}%** / 가격 **{100-env_ratio}%**")
        price_ratio = 100 - env_ratio

        env_weight_sum   = sum(CRITIC_WEIGHTS[:-1])
        adjusted_weights = np.zeros(len(CRITIC_WEIGHTS))
        for k in range(len(CRITIC_WEIGHTS) - 1):
            adjusted_weights[k] = CRITIC_WEIGHTS[k] / env_weight_sum * (env_ratio / 100)
        adjusted_weights[-1] = price_ratio / 100

        st.divider()
        st.subheader("💰 예산 상한 설정")
        st.caption("입력한 금액 이하 제품에 ✅ 표시됩니다. 0이면 전체 표시.")
        price_limit = st.number_input(
            "공급가격 상한 (원/m³)",
            min_value=0, value=130000, step=1000,
            help="레미콘 1m³당 공급가격 상한선을 설정합니다."
        )

        search_btn = st.button("🔎 추천 자재 검색", use_container_width=True, type="primary")



    if search_btn:
        filtered, upper_str, slump_lower, slump_upper = filter_data(
            df, region, min_strength, slump, slump_range=0)
        slump_expanded = False

        if len(filtered) < 5:
            filtered_exp, upper_str, slump_lower, slump_upper = filter_data(
                df, region, min_strength, slump, slump_range=1)
            if len(filtered_exp) > len(filtered):
                filtered       = filtered_exp
                slump_expanded = True

        info_msg = (
            f"📌 검색 범위: **{region}** / "
            f"강도 **{min_strength}~{upper_str}MPa** / "
            f"슬럼프 **{slump_lower}~{slump_upper}mm** / "
            f"검색된 자재 수: **{len(filtered)}개**"
        )
        if slump_expanded:
            info_msg += f"  \n⚠️ 슬럼프 {slump}mm 결과가 부족하여 {slump_upper}mm까지 자동 확장되었습니다."
        st.info(info_msg)

        if len(filtered) == 0:
            st.error("❌ 해당 조건의 자재가 없습니다. 조건을 변경해주세요.")

        elif len(filtered) < 5:
            st.warning(f"⚠️ 자재 수가 {len(filtered)}개로 적어 단순 비교표를 제공합니다.")
            display_df = filtered[['인증업체명', '저탄소인증', '설계기준강도', '슬럼프'] + CRITERIA_COLS].copy()
            # 환경영향 수치 소수 둘째자리
            for col in CRITERIA_COLS[:-1]:  # 공급가격 제외
                display_df[col] = display_df[col].round(2)
            display_df['공급가격'] = display_df['공급가격'].apply(lambda x: f"{int(x):,}")
            display_df['저탄소인증'] = display_df['저탄소인증'].map({True: '✅', False: ''})
            display_df = display_df.rename(columns={**CRITERIA_LABELS, '저탄소인증': '저탄소'})
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        else:
            result     = topsis(filtered, CRITERIA_COLS, adjusted_weights)
            top_result = result.head(top_n).copy()

            tab1, tab2, tab3 = st.tabs(["🏆 추천 순위", "📈 환경영향 비교", "📋 전체 데이터"])

            with tab1:
                st.subheader(f"상위 {min(top_n, len(result))}개 추천 자재")

                # 컬럼 순서: 순위 > 업체명 > 저탄소 > 가격 > 예산적합 > 강도 > 슬럼프 > 근접도
                show_df = top_result[['순위', '인증업체명', '저탄소인증', '공급가격',
                                       '설계기준강도', '슬럼프', '근접도(Ci)']].copy()
                show_df['근접도(Ci)'] = show_df['근접도(Ci)'].round(4)
                show_df['저탄소인증'] = show_df['저탄소인증'].map({True: '✅', False: ''})

                if price_limit > 0:
                    show_df['예산적합'] = show_df['공급가격'].apply(
                        lambda x: '✅' if x <= price_limit else '')
                    show_df = show_df[['순위', '인증업체명', '저탄소인증', '예산적합',
                                       '공급가격', '설계기준강도', '슬럼프', '근접도(Ci)']]

                show_df['공급가격'] = show_df['공급가격'].apply(lambda x: f"{int(x):,}")
                show_df = show_df.rename(columns={
                    '인증업체명': '업체명', '저탄소인증': '저탄소',
                    '공급가격': '공급가격(원/m³)', '설계기준강도': '강도(MPa)',
                    '슬럼프': '슬럼프(mm)', '근접도(Ci)': '근접도(Ci)'
                })

                st.dataframe(show_df, hide_index=True, use_container_width=True,
                    column_config={
                        '순위': st.column_config.NumberColumn('순위', width='small'),
                        '근접도(Ci)': st.column_config.ProgressColumn(
                            '근접도(Ci)', min_value=0, max_value=1, format="%.4f")
                    })

                best      = top_result.iloc[0]
                cert_label = '🌱저탄소인증제품' if best['저탄소인증'] else 'EPD'
                st.success(
                    f"🥇 **1위 추천 자재: {best['인증업체명']}** [{cert_label}] | "
                    f"강도 {int(best['설계기준강도'])}MPa | "
                    f"슬럼프 {int(best['슬럼프'])}mm | "
                    f"탄소발자국 {best['탄소발자국_총량']:.2f} | "
                    f"가격 {int(best['공급가격']):,}원"
                )

            with tab2:
                st.subheader("상위 10개 자재 환경영향 비교")
                chart_data = top_result.head(10)[
                    ['인증업체명', '탄소발자국_총량', '산성화_총량', '부영양화_총량', '오존층영향_총량']
                ].copy()
                chart_data['라벨'] = (chart_data['인증업체명'] + '(' +
                                      top_result.head(10)['순위'].astype(str) + '위)')
                chart_data = chart_data.set_index('라벨')
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**탄소발자국 (kg CO₂-eq/m³)**")
                    st.bar_chart(chart_data['탄소발자국_총량'])
                with col2:
                    st.markdown("**공급가격 (원/m³)**")
                    st.bar_chart(top_result.head(10).set_index('인증업체명')['공급가격'])

            with tab3:
                st.subheader(f"전체 {len(result)}개 자재 순위")
                full_display = result[['순위', '인증업체명', '설계기준강도', '슬럼프'] +
                                       CRITERIA_COLS + ['저탄소인증', '근접도(Ci)']].copy()
                full_display['근접도(Ci)']  = full_display['근접도(Ci)'].round(4)
                for col in CRITERIA_COLS[:-1]:  # 공급가격 제외
                    full_display[col] = full_display[col].round(2)
                full_display['저탄소인증'] = full_display['저탄소인증'].map({True: '저탄소인증✅', False: ''})
                full_display = full_display.rename(columns={**CRITERIA_LABELS, '저탄소인증': '저탄소'})
                st.dataframe(full_display, hide_index=True, use_container_width=True)

                csv = full_display.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 결과 다운로드 (CSV)",
                    data=csv,
                    file_name=f'추천결과_{region}_{min_strength}MPa_{slump}mm.csv',
                    mime='text/csv'
                )

    else:
        st.markdown("### 👈 왼쪽에서 조건을 선택하고 검색 버튼을 눌러주세요.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("전체 자재 수", f"{len(df):,}개")
        with col2:
            st.metric("인증업체 수", f"{df['인증업체명'].nunique()}개")
        with col3:
            st.metric("저탄소 제품 수", f"{df['저탄소인증'].sum():,}개")
        with col4:
            st.metric("지역 수", "6개")
        st.divider()
        st.markdown("""
        #### 📌 시스템 안내
        - **EPD(환경성적표지)** 인증 데이터 기반 친환경 레미콘 추천 시스템입니다.
        - **CRITIC** 방법으로 산출한 객관적 가중치와 **TOPSIS** 방법으로 순위를 계산합니다.
        - 환경영향(탄소발자국, 산성화 등 7개 항목)과 공급가격을 동시에 고려합니다.
        - 입력한 최소 강도보다 **최대 2단계 높은 강도**까지 포함하여 검색합니다.
        - 입력한 최소 슬럼프 **이상** ~ **+1단계**까지 포함하여 검색합니다.
        - 저탄소 인증 여부 및 예산 적합 여부를 결과에 함께 표시합니다.
        """)

if __name__ == '__main__':
    main()
