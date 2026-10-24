import os
import warnings
import joblib
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from huggingface_hub import HfApi

from features import (
    FEATURE_COLS,
    build_features,
    calculate_ticker_metrics,
    apply_bias_correction,
    send_telegram_report,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def train_and_upload():
    target_tickers = os.getenv("STOCK_TICKER", "AAPL")
    tickers = list(dict.fromkeys(t.strip() for t in target_tickers.split(",") if t.strip()))

    hf_token = os.getenv("HF_TOKEN")
    hf_model_repo = os.getenv("HF_MODEL_REPO_XGB", "nadtoka/predictive-stock-models-xgb")
    hf_repo_inside = os.getenv("HF_REPO", "nadtoka/predictive-stock-dataset")
    quiet_tg = os.getenv("QUIET_TELEGRAM", "0") == "1"

    os.makedirs("models_xgb", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Завантажуємо окрему історію аудиту XGBoost
    df_eval = pd.DataFrame()
    if hf_repo_inside:
        try:
            eval_url = f"https://huggingface.co/datasets/{hf_repo_inside}/raw/main/evaluation_history_xgb.csv"
            df_eval = pd.read_csv(eval_url)
            if not df_eval.empty and "actual_price" in df_eval.columns:
                df_eval = df_eval.dropna(subset=["actual_price", "mae_pct"])
                df_eval = df_eval[df_eval["actual_price"] > 0]
            print(f"📋 Завантажено базу оцінювання evaluation_history_xgb.csv ({len(df_eval)} рядків).")
        except Exception as e:
            print(f"ℹ️ Не вдалося завантажити evaluation_history_xgb.csv (працюємо без компенсації): {e}")

    sp500_path = "data/SP500_history.csv"
    sp500_df = pd.read_csv(sp500_path, index_col=0, parse_dates=True) if os.path.exists(sp500_path) else None

    vix_path = "data/VIX_history.csv"
    vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True) if os.path.exists(vix_path) else None

    tg_report = "⚡ *XGBOOST (v3.3 Stationary Features)*\n"
    tg_report += "━━━━━━━━━━━━━━━━\n\n"

    daily_predictions = []
    successful_models = 0

    for ticker in tickers:
        try:
            data_path = f"data/{ticker}_history.csv"
            if not os.path.exists(data_path):
                print(f"⚠️ Файл даних для {ticker} не знайдено. Пропускаємо.")
                continue

            df_raw = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if len(df_raw) < 35:
                print(f"⚠️ Для {ticker} недостатньо історії ({len(df_raw)} рядків, потрібно >= 35). Пропускаємо.")
                continue

            print(f"⚡ Обробка та XGBoost тренування для {ticker}...")
            df = build_features(df_raw, sp500_df=sp500_df, vix_df=vix_df)
            df = df.dropna(subset=FEATURE_COLS)
            if df.empty or len(df) < 25:
                print(f"❌ Недостатньо валідних рядків після створення ознак для {ticker}.")
                continue

            latest_features = df[FEATURE_COLS].tail(1).copy()
            current_price = float(df["Close"].iloc[-1])
            vix_current = float(df["VIX_Close"].iloc[-1]) if "VIX_Close" in df.columns else 15.0

            df["Target_1d"] = df["Close"].pct_change(fill_method=None).shift(-1)
            df["Target_5d"] = df["Close"].pct_change(periods=5, fill_method=None).shift(-5)
            df["Target_20d"] = df["Close"].pct_change(periods=20, fill_method=None).shift(-20)
            df = df.dropna(subset=["Target_1d", "Target_5d", "Target_20d"])

            if df.empty or len(df) < 20:
                print(f"❌ Недостатньо тарґетів для {ticker}.")
                continue

            test_size = max(3, min(20, int(len(df) * 0.15)))
            # Embargo gap (буфер 20 сесій між train та test), щоб уникнути витоку майбутнього
            embargo = 20 if len(df) > (test_size + 45) else 0
            train_end = len(df) - test_size - embargo
            train_df = df.iloc[:train_end]
            test_df = df.iloc[-test_size:]

            X_train, y_train = train_df[FEATURE_COLS], train_df[["Target_1d", "Target_5d", "Target_20d"]]
            X_test, y_test = test_df[FEATURE_COLS], test_df[["Target_1d", "Target_5d", "Target_20d"]]

            if len(X_train) < 5 or len(X_test) == 0:
                print(f"❌ Недостатньо тренувальних зразків для {ticker} (train: {len(X_train)}). Пропускаємо.")
                continue

            xgb_params = dict(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            )

            # Чесна Out-of-Sample валідація з буфером embargo
            eval_model = MultiOutputRegressor(XGBRegressor(**xgb_params))
            eval_model.fit(X_train, y_train)
            predictions = eval_model.predict(X_test)
            mae_1d, mae_5d, mae_20d = mean_absolute_error(y_test, predictions, multioutput="raw_values")
            mae_1d_pct = mae_1d * 100
            mae_5d_pct = mae_5d * 100
            mae_20d_pct = mae_20d * 100

            # Фінальне бойове навчання на всіх доступних даних
            model = MultiOutputRegressor(XGBRegressor(**xgb_params))
            model.fit(df[FEATURE_COLS], df[["Target_1d", "Target_5d", "Target_20d"]])

            preds = model.predict(latest_features)[0]
            raw_1d = current_price * (1.0 + preds[0])
            raw_5d = current_price * (1.0 + preds[1])
            raw_20d = current_price * (1.0 + preds[2])

            # Уніфікований Feedback loop (відсоткова корекція bias_pct)
            ticker_metrics = calculate_ticker_metrics(df_eval, ticker)
            tomorrow_pred = apply_bias_correction(raw_1d, current_price, ticker_metrics, "1d", vix_current)
            week_pred = apply_bias_correction(raw_5d, current_price, ticker_metrics, "5d", vix_current)
            month_pred = apply_bias_correction(raw_20d, current_price, ticker_metrics, "20d", vix_current)

            count_1d = ticker_metrics["1d"]["count"]
            win_rate_1d = ticker_metrics["1d"]["win_rate"]

            if count_1d < 5:
                badge = "🟡"
            elif win_rate_1d >= 60.0:
                badge = "🟢"
            elif win_rate_1d >= 45.0:
                badge = "🟡"
            else:
                badge = "🔴"

            model_path = f"models_xgb/{ticker}_model.joblib"
            joblib.dump(model, model_path)
            successful_models += 1

            last_date = latest_features.index[-1]
            daily_predictions.append(
                {
                    "date": last_date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "current_price": float(current_price),
                    "pred_1d": float(tomorrow_pred),
                    "pred_5d": float(week_pred),
                    "pred_20d": float(month_pred),
                }
            )

            pct_1d = (tomorrow_pred - current_price) / current_price * 100
            pct_5d = (week_pred - current_price) / current_price * 100
            pct_20d = (month_pred - current_price) / current_price * 100
            wr_str = f" WR:{win_rate_1d:.0f}%" if count_1d >= 5 else ""

            tg_report += (
                f"{badge} *{ticker}* `${current_price:.2f}`{wr_str} | "
                f"1d: `{pct_1d:+.1f}%` (${tomorrow_pred:.2f}) • "
                f"5d: `{pct_5d:+.1f}%` • "
                f"20d: `{pct_20d:+.1f}%` _(MAE {mae_1d_pct:.1f}%/{mae_5d_pct:.1f}%/{mae_20d_pct:.1f}%)_\n"
            )

        except Exception as e:
            print(f"❌ Помилка тренування XGBoost для {ticker}: {e}")

    # Зберігаємо локальну копію сьогоднішніх прогнозів для Скрипта-Арбітра
    if daily_predictions:
        pd.DataFrame(daily_predictions).to_csv("data/latest_predictions_xgb.csv", index=False)

    # Пуш моделей в окремий репозиторій XGBoost
    if hf_token and hf_model_repo and successful_models > 0:
        try:
            print(f"📦 Пуш оновлених XGBoost моделей на Hugging Face Hub ({hf_model_repo})...")
            api = HfApi()
            api.upload_folder(
                folder_path="models_xgb",
                repo_id=hf_model_repo,
                repo_type="model",
                token=hf_token,
            )
            tg_report += "\n☁️ *HF:* XGBoost моделі синхронізовано."
        except Exception as e:
            tg_report += f"\n⚠️ *HF:* Помилка завантаження ваг: {e}"

    if not quiet_tg:
        send_telegram_report(tg_report)
    else:
        print("ℹ️ QUIET_TELEGRAM=1: індивідуальний звіт XGBoost приховано (буде надіслано об'єднаний дайджест Арбітра).")

    # Збереження історії в окремий predictions_history_xgb.csv
    if hf_token and hf_repo_inside and daily_predictions:
        try:
            print("💾 Синхронізація predictions_history_xgb.csv з Hugging Face...")
            api = HfApi()
            file_path = "predictions_history_xgb.csv"
            remote_url = f"https://huggingface.co/datasets/{hf_repo_inside}/raw/main/{file_path}"

            try:
                df_history = pd.read_csv(remote_url)
            except Exception:
                df_history = pd.DataFrame(
                    columns=["date", "ticker", "current_price", "pred_1d", "pred_5d", "pred_20d"]
                )

            df_new = pd.DataFrame(daily_predictions)
            df_combined = pd.concat([df_history, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["date", "ticker"], keep="last")

            csv_data = df_combined.to_csv(index=False)
            api.upload_file(
                path_or_fileobj=csv_data.encode("utf-8"),
                path_in_repo=file_path,
                repo_id=hf_repo_inside,
                repo_type="dataset",
                token=hf_token,
            )
            print("✅ predictions_history_xgb.csv засинкронено в Hugging Face!")
        except Exception as e:
            print(f"⚠️ Помилка синхронізації predictions_history_xgb.csv: {e}")


if __name__ == "__main__":
    train_and_upload()
