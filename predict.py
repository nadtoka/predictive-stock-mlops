import os
import warnings
import joblib
import pandas as pd
import yfinance as yf
from huggingface_hub import hf_hub_download

from features import (
    FEATURE_COLS,
    build_features,
    calculate_ticker_metrics,
    apply_bias_correction,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_evaluation_history(filename="evaluation_history.csv"):
    """Завантажує історію оцінок для Continuous Learning без обов'язкових токенів."""
    hf_repo = os.getenv("HF_REPO", "nadtoka/predictive-stock-dataset")
    eval_url = f"https://huggingface.co/datasets/{hf_repo}/raw/main/{filename}"
    try:
        df_eval = pd.read_csv(eval_url)
        if not df_eval.empty and "actual_price" in df_eval.columns:
            df_eval = df_eval.dropna(subset=["actual_price", "mae_pct"])
            df_eval = df_eval[df_eval["actual_price"] > 0]
        return df_eval
    except Exception as e:
        print(f"ℹ️ Не вдалося завантажити {filename} (працюємо без компенсації зсуву): {e}")
        return pd.DataFrame()


def prepare_features_for_prediction(ticker):
    """Збір та підготовка свіжих ринкових даних для інференсу."""
    print(f"📡 Збір свіжих даних з Yahoo Finance для {ticker}...")
    stock = yf.Ticker(ticker)
    df_raw = stock.history(period="2y")

    if df_raw.empty:
        raise ValueError(f"Не вдалося отримати дані для {ticker}")

    sp500_df = yf.Ticker("^GSPC").history(period="2y")
    vix_df = yf.Ticker("^VIX").history(period="2y")

    try:
        vxn_df = yf.Ticker("^VXN").history(period="2y")
    except Exception:
        vxn_df = None

    try:
        tnx_df = yf.Ticker("^TNX").history(period="2y")
    except Exception:
        tnx_df = None

    try:
        soxx_df = yf.Ticker("SOXX").history(period="2y")
    except Exception:
        soxx_df = None

    try:
        earnings_df = stock.get_earnings_dates(limit=20)
    except Exception:
        earnings_df = None

    df = build_features(
        df_raw,
        sp500_df=sp500_df,
        vix_df=vix_df,
        vxn_df=vxn_df,
        tnx_df=tnx_df,
        soxx_df=soxx_df,
        earnings_df=earnings_df,
    )
    df_latest = df.dropna(subset=FEATURE_COLS)

    if df_latest.empty or len(df_latest) < 25:
        raise ValueError(f"❌ Недостатньо даних для розрахунку індикаторів для {ticker}.")

    current_price = float(df_latest["Close"].iloc[-1])
    vix_current = float(df_latest["VIX_Close"].iloc[-1]) if "VIX_Close" in df_latest else 15.0
    return df_latest[FEATURE_COLS].tail(1), current_price, vix_current


def _predict_with_model(model, latest_features, current_price, vix_current, ticker_metrics):
    """Допоміжна функція інференсу з автоматичним fallback, якщо стара модель на HF очікує старий набір фіч."""
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_names_in_"):
        expected_cols = list(model.estimators_[0].feature_names_in_)
    else:
        expected_cols = FEATURE_COLS

    X_input = latest_features.copy()
    for col in expected_cols:
        if col not in X_input.columns:
            X_input[col] = current_price if col in ("Close", "MA_5", "MA_20", "PE_Ratio", "PS_Ratio") else 0.0
    X_input = X_input[expected_cols]

    preds = model.predict(X_input)[0]
    raw_1d = current_price * (1.0 + preds[0])
    raw_5d = current_price * (1.0 + preds[1])
    raw_20d = current_price * (1.0 + preds[2])

    pred_1d = apply_bias_correction(raw_1d, current_price, ticker_metrics, "1d", vix_current)
    pred_5d = apply_bias_correction(raw_5d, current_price, ticker_metrics, "5d", vix_current)
    pred_20d = apply_bias_correction(raw_20d, current_price, ticker_metrics, "20d", vix_current)
    return pred_1d, pred_5d, pred_20d


def run_inference(ticker, df_eval_rf=None, df_eval_xgb=None):
    try:
        latest_features, current_price, vix_current = prepare_features_for_prediction(ticker)
        metrics_rf = calculate_ticker_metrics(df_eval_rf, ticker)
        metrics_xgb = calculate_ticker_metrics(df_eval_xgb, ticker)

        rf_preds = None
        xgb_preds = None

        try:
            rf_path = hf_hub_download(
                repo_id=os.getenv("HF_MODEL_REPO", "nadtoka/predictive-stock-models"),
                filename=f"{ticker}_model.joblib",
                repo_type="model",
            )
            rf_model = joblib.load(rf_path)
            rf_preds = _predict_with_model(rf_model, latest_features, current_price, vix_current, metrics_rf)
        except Exception as e:
            print(f"ℹ️ Random Forest модель для {ticker} недоступна: {e}")

        try:
            xgb_path = hf_hub_download(
                repo_id=os.getenv("HF_MODEL_REPO_XGB", "nadtoka/predictive-stock-models-xgb"),
                filename=f"{ticker}_model.joblib",
                repo_type="model",
            )
            xgb_model = joblib.load(xgb_path)
            xgb_preds = _predict_with_model(xgb_model, latest_features, current_price, vix_current, metrics_xgb)
        except Exception as e:
            print(f"ℹ️ XGBoost модель для {ticker} недоступна: {e}")

        if not rf_preds and not xgb_preds:
            raise RuntimeError(f"Жодна з моделей не знайдена для {ticker}")

        wr_rf = metrics_rf["1d"]["win_rate"]
        wr_xgb = metrics_xgb["1d"]["win_rate"]
        cnt_rf = metrics_rf["1d"]["count"]
        cnt_xgb = metrics_xgb["1d"]["count"]

        if rf_preds and xgb_preds:
            # Ваговий ансамбль пропорційно точності (або 50/50)
            w_rf = max(0.3, wr_rf / 100.0) if cnt_rf >= 5 else 0.5
            w_xgb = max(0.3, wr_xgb / 100.0) if cnt_xgb >= 5 else 0.5
            total_w = w_rf + w_xgb
            ens_1d = (rf_preds[0] * w_rf + xgb_preds[0] * w_xgb) / total_w
            ens_5d = (rf_preds[1] * w_rf + xgb_preds[1] * w_xgb) / total_w
            ens_20d = (rf_preds[2] * w_rf + xgb_preds[2] * w_xgb) / total_w
            leader = "⚡ XGB" if wr_xgb >= wr_rf else "🌲 RF"
        elif xgb_preds:
            ens_1d, ens_5d, ens_20d = xgb_preds
            leader = "⚡ XGB"
        else:
            ens_1d, ens_5d, ens_20d = rf_preds
            leader = "🌲 RF"

        best_wr = max(wr_rf if cnt_rf >= 5 else 50.0, wr_xgb if cnt_xgb >= 5 else 50.0)
        if max(cnt_rf, cnt_xgb) < 5:
            badge = "🟡"
        elif best_wr >= 60.0:
            badge = "🟢"
        elif best_wr >= 45.0:
            badge = "🟡"
        else:
            badge = "🔴"

        last_date = latest_features.index[-1]
        ua_days = {0: "Понеділок", 1: "Вівторок", 2: "Середа", 3: "Четвер", 4: "П'ятниця", 5: "Субота", 6: "Неділя"}
        date_1d = last_date + pd.offsets.BusinessDay(1)
        date_5d = last_date + pd.offsets.BusinessDay(5)
        date_20d = last_date + pd.offsets.BusinessDay(20)

        day_1d_text = f"{ua_days[date_1d.weekday()]} ({date_1d.strftime('%d.%m')})"
        day_5d_text = f"{ua_days[date_5d.weekday()]} ({date_5d.strftime('%d.%m')})"
        day_20d_text = f"{ua_days[date_20d.weekday()]} ({date_20d.strftime('%d.%m')})"

        emoji_1d = "📈" if ens_1d > current_price else "📉"
        emoji_5d = "🚀" if ens_5d > current_price else "📉"
        emoji_20d = "🚀" if ens_20d > current_price else "📉"

        print(f"\n🔮 === АНСАМБЛЬ ПРОГНОЗ ШІ ({badge} {ticker} | Лідер: {leader}) ===")
        print(f"   📈 Поточна ціна: ${current_price:.2f}")
        print(f"   🚀 {day_1d_text} (1d):  ${ens_1d:.2f} ({(ens_1d - current_price)/current_price*100:+.2f}%) {emoji_1d}")
        if rf_preds and xgb_preds:
            print(f"      ├─ 🌲 RF: ${rf_preds[0]:.2f} (WR: {wr_rf:.0f}%) | ⚡ XGB: ${xgb_preds[0]:.2f} (WR: {wr_xgb:.0f}%)")
        print(f"   📅 {day_5d_text} (5d):  ${ens_5d:.2f} ({(ens_5d - current_price)/current_price*100:+.2f}%) {emoji_5d}")
        print(f"   📆 {day_20d_text} (20d): ${ens_20d:.2f} ({(ens_20d - current_price)/current_price*100:+.2f}%) {emoji_20d}")

    except Exception as e:
        print(f"❌ Сталася помилка під час інференсу для {ticker}: {e}")


if __name__ == "__main__":
    target_tickers = os.getenv("STOCK_TICKER", "AAPL")
    tickers = list(dict.fromkeys(t.strip() for t in target_tickers.split(",") if t.strip()))

    print(f"🚀 Запуск інференсу для списку тікерів: {tickers}")
    df_eval_rf = load_evaluation_history("evaluation_history.csv")
    df_eval_xgb = load_evaluation_history("evaluation_history_xgb.csv")

    for ticker in tickers:
        run_inference(ticker, df_eval_rf=df_eval_rf, df_eval_xgb=df_eval_xgb)
