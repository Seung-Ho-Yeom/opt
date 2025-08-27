# streamlit_app_fixed.py
# ---------------------------------------------------------------
# Streamlit UI for your Bayesian Optimization script (fixed params)
# - 사용자 조정 섹션(2,3,4) 제거
# - 사이드바: 고정 경로 표시 + Run 버튼
# - 모델 JSON은 지정 경로에서 자동 로드
# ---------------------------------------------------------------
import json
import time
from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor


# ------------------ 원본 스크립트의 고정 값 ------------------
run_counter = 0
initial_design_run = 20
total_design_run = 100
history_data = []  # 결과 저장
base_dp, base_sd = None, None
w1, w2 = 1, 5  # w1 -> dp, w2 -> N2_sd

# 탐색 공간 (고정)
search_space = [
    Real(25.0, 65.0, name='FinAngle'),
    Real(0.2, 0.75, name='FinDepth'),
    Real(0.5, 1.5, name='FinLength'),
    Real(0.01, 0.2, name='FinWidth'),
    Integer(3, 6, name='NumFins')
]

# baseline 값 (고정)
baseline_params = [45, 0.5, 1.0, 0.02, 4]


# ------------------ GPR 파이프라인 복원 헬퍼 ------------------
def _to_numpy(x):
    return np.asarray(x, dtype=float) if x is not None else None

def _build_kernel_from_deep_params(deep: dict):
    cv = float(deep.get("k1__constant_value", 1.0))
    ls = deep.get("k2__length_scale", 1.0)
    if isinstance(ls, list):
        ls = np.asarray(ls, dtype=float)
    else:
        ls = float(ls)
    nl = float(deep.get("k3__noise_level", 1e-6))
    return (C(cv, constant_value_bounds="fixed")
            * RBF(length_scale=ls, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=nl, noise_level_bounds="fixed"))

def _rebuild_gpr_pipeline_from_json_dict(j: dict) -> Pipeline:
    kernel = _build_kernel_from_deep_params(j.get("kernel_params_deep", {}))
    est = j.get("estimator_params", {}) or {}
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=est.get("alpha", 1e-10),
        normalize_y=est.get("normalize_y", False),
        n_restarts_optimizer=est.get("n_restarts_optimizer", 0),
        random_state=est.get("random_state", None),
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # 저장된 스케일러 상태 복원(있으면)
    if "x_scaler" in j:
        xs = j["x_scaler"]
        x_scaler.mean_ = _to_numpy(xs.get("mean"))
        x_scaler.scale_ = _to_numpy(xs.get("scale"))
        x_scaler.var_ = (x_scaler.scale_ ** 2) if x_scaler.scale_ is not None else None
        x_scaler.n_features_in_ = x_scaler.mean_.shape[0] if x_scaler.mean_ is not None else None

    if "y_scaler" in j:
        ys = j["y_scaler"]
        y_scaler.mean_ = _to_numpy(ys.get("mean")) if ys.get("mean") is not None else None
        y_scaler.scale_ = _to_numpy(ys.get("scale")) if ys.get("scale") is not None else None
        y_scaler.var_ = (y_scaler.scale_ ** 2) if y_scaler.scale_ is not None else None

    return Pipeline([
        ("x_scaler", x_scaler),
        ("gpr", TransformedTargetRegressor(regressor=gpr, transformer=y_scaler)),
    ])

def _load_pipeline_from_json_bytes(json_bytes: bytes) -> Pipeline:
    j = json.loads(json_bytes.decode("utf-8"))
    if "train_data" not in j:
        raise RuntimeError("JSON에 'train_data'가 없습니다. 저장 시 X,y를 포함해야 합니다.")
    pipe = _rebuild_gpr_pipeline_from_json_dict(j)
    tr = j["train_data"]
    X_tr = _to_numpy(tr.get("X"))
    y_tr = _to_numpy(tr.get("y"))
    if X_tr is None or y_tr is None:
        raise RuntimeError("'train_data' 형식이 올바르지 않습니다. 'X'와 'y'가 필요합니다.")
    pipe.fit(X_tr, y_tr)  # 재학습이 아닌 posterior 재현
    return pipe

# JSON 직렬화 안전 변환
def _to_native(o):
    import numpy as _np
    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    return o


# ------------------ 모델 경로 (자동 로드) ------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
SD_JSON_PATH = os.path.join(MODEL_DIR, "gpr_rbf_Outlet_N2_SD.json")
DP_JSON_PATH = os.path.join(MODEL_DIR, "gpr_rbf_Outlet_dp.json")


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Bayesian Optimization (Fixed)", layout="wide")
st.title("Bayesian Optimization (GPR from JSON) — Fixed Parameters")

with st.sidebar:
    st.header("Models (auto-load)")
    st.caption(f"SD: {SD_JSON_PATH}")
    st.caption(f"DP: {DP_JSON_PATH}")
    run_button = st.button("▶ Run Optimization")

# 상태 관리
if "history_data" not in st.session_state:
    st.session_state.history_data = []
if "run_counter" not in st.session_state:
    st.session_state.run_counter = 0

# 자리표시자
progress_ph = st.empty()
col_left, col_right = st.columns([2, 1])
obj_plot_ph = col_left.empty()
pareto_plot_ph = col_left.empty()
table_ph = col_right.empty()
best_ph = col_right.empty()


# ---- 업로드 대신 고정 경로에서 자동 로드 ----
def build_models():
    if not os.path.exists(SD_JSON_PATH):
        st.error(f"SD 모델 JSON 경로가 없습니다.\n{SD_JSON_PATH}")
        st.stop()
    if not os.path.exists(DP_JSON_PATH):
        st.error(f"DP 모델 JSON 경로가 없습니다.\n{DP_JSON_PATH}")
        st.stop()

    try:
        with open(SD_JSON_PATH, "rb") as f:
            sd_bytes = f.read()
        gpr_sd = _load_pipeline_from_json_bytes(sd_bytes)
    except Exception as e:
        st.error(f"SD 모델 JSON 로딩 실패: {e}")
        st.stop()

    try:
        with open(DP_JSON_PATH, "rb") as f:
            dp_bytes = f.read()
        gpr_dp = _load_pipeline_from_json_bytes(dp_bytes)
    except Exception as e:
        st.error(f"DP 모델 JSON 로딩 실패: {e}")
        st.stop()

    return gpr_sd, gpr_dp


class LivePlotCallback:
    """Streamlit에서 매 반복마다 Matplotlib 그림을 업데이트"""
    def __init__(self, obj_ph, pareto_ph, table_ph):
        self.obj_ph = obj_ph
        self.pareto_ph = pareto_ph
        self.table_ph = table_ph

    def __call__(self, res):
        # Objective progress plot
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        y_vals = res.func_vals
        x_vals = np.arange(1, len(y_vals) + 1)
        ax1.plot(x_vals, y_vals, 'o-')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Objective Value")
        ax1.set_title("Optimization Progress")
        self.obj_ph.pyplot(fig1)
        plt.close(fig1)

        # Pareto scatter (colored by objective)
        hist = st.session_state.history_data
        if hist and len(res.func_vals) == len(hist):
            n2_sd_vals = [row["Outlet_N2_SD"] for row in hist]
            dp_vals = [row["Outlet_dp"] for row in hist]
            objective_vals = list(res.func_vals)

            # 마지막 row에 objective 기록
            hist[-1]["Objective Value"] = float(objective_vals[-1])

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sc = ax2.scatter(n2_sd_vals, dp_vals, c=objective_vals, cmap='hot', s=60, edgecolor='k')
            best_idx = int(np.argmin(objective_vals))
            ax2.plot(n2_sd_vals[best_idx], dp_vals[best_idx], '*', markersize=14, label="Best")
            ax2.plot(n2_sd_vals[0], dp_vals[0], 'X', markersize=12, label="Baseline")
            ax2.set_xlabel("Outlet_N2_SD")
            ax2.set_ylabel("Outlet_dp")
            ax2.set_title("Pareto (colored by Objective)")
            ax2.grid(True)
            ax2.legend()
            cbar = fig2.colorbar(sc, ax=ax2)
            cbar.set_label("Objective Value")
            self.pareto_ph.pyplot(fig2)
            plt.close(fig2)

        # 표 갱신 (최근 10개)
        if st.session_state.history_data:
            df_hist = pd.DataFrame(st.session_state.history_data)
            self.table_ph.dataframe(df_hist.tail(10), use_container_width=True)


def run_optimization():
    global run_counter, history_data, base_dp, base_sd

    gpr_sd, gpr_dp = build_models()

    # reset
    run_counter = 0
    history_data = []
    base_dp, base_sd = None, None
    st.session_state.history_data = []
    st.session_state.run_counter = 0

    # 목적함수 (원본 로직 유지)
    @use_named_args(search_space)
    def objective(**params):
        global run_counter, history_data, base_dp, base_sd, w1, w2

        run_counter += 1
        st.session_state.run_counter = run_counter
        t0 = time.time()

        # 파라미터를 기본 타입으로 정리해서 기록
        clean_params = {
            "FinAngle": float(params["FinAngle"]),
            "FinDepth": float(params["FinDepth"]),
            "FinLength": float(params["FinLength"]),
            "FinWidth": float(params["FinWidth"]),
            "NumFins": int(params["NumFins"]),
        }
        param_value = np.array(list(clean_params.values()), dtype=float).reshape(1, -1)

        outlet_n2_sd = float(gpr_sd.predict(param_value)[0])
        outlet_dp = float(gpr_dp.predict(param_value)[0])

        result_row = OrderedDict()
        result_row["Case"] = int(run_counter)
        result_row.update(clean_params)
        result_row["Outlet_N2_SD"] = outlet_n2_sd
        result_row["Outlet_dp"] = outlet_dp
        history_data.append(result_row)
        st.session_state.history_data = history_data

        if run_counter == 1:
            base_dp = outlet_dp
            base_sd = outlet_n2_sd
            obj_val = w1 * (outlet_dp / base_dp) + w2 * (outlet_n2_sd / base_sd)
        else:
            dp_norm = outlet_dp / base_dp
            sd_norm = outlet_n2_sd / base_sd
            obj_val = w1 * dp_norm + w2 * sd_norm

        dt = time.time() - t0
        progress_ph.info(f"#{run_counter}/{total_design_run} done in {dt:.2f}s  |  Objective={obj_val:.6f}")
        return float(obj_val)

    # 콜백
    live_cb = LivePlotCallback(obj_plot_ph, pareto_plot_ph, table_ph)

    # 최적화 실행 (원본 세팅 유지)
    res = gp_minimize(
        func=objective,
        dimensions=search_space,
        acq_func="EI",
        n_calls=total_design_run,
        n_initial_points=initial_design_run,
        x0=[baseline_params],
        callback=[live_cb],
        initial_point_generator='lhs',
        random_state=42
    )

    # 결과 요약
    df_hist = pd.DataFrame(history_data)
    best_idx = int(np.argmin(res.func_vals))
    # best_params: NumFins만 int, 나머지 float로 변환
    best_params = [float(v) if i != 4 else int(v) for i, v in enumerate(res.x)]
    best_value = float(res.fun)

    best_ph.success(f"Best Objective: {best_value:.6f} at iteration #{best_idx+1}")
    st.subheader("Best design (params and outputs)")
    st.dataframe(df_hist.iloc[[best_idx]], use_container_width=True)

    # 다운로드
    st.download_button(
        "Download optimization_history.csv",
        data=df_hist.to_csv(index=False).encode("utf-8"),
        file_name="optimization_history.csv",
        mime="text/csv",
    )

    out_json = {
        "best_params": best_params,
        "best_value": best_value,
        "history": history_data,
    }
    safe_out = _to_native(out_json)  # NumPy → 파이썬 기본 타입으로 변환
    st.download_button(
        "Download results.json",
        data=json.dumps(safe_out, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="results.json",
        mime="application/json",
    )

# 실행 트리거
if run_button:
    run_optimization()



