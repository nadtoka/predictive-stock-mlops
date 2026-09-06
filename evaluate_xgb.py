import os
import warnings
import pandas as pd
import yfinance as yf
from huggingface_hub import HfApi
from curl_cffi import requests

warnings.simplefilter(action='ignore', category=FutureWarning)


def send_telegram_report(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    MAX_LEN = 4000
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) + 1 > MAX_LEN:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chunk in chunks:
        if not chunk:
            continue
        try:
            requests.post(url, json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"}, timeout=10)
        except Exception as e:
            print(f"❌ Telegram API Error: {e}")


def evaluate_predictions():
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")
    if not hf_token or not hf_repo:
        return

    file_path = "predictions_history_xgb.csv"
    remote_url = f"https://huggingface.co/datasets/{hf_repo}/raw/main/{file_path}"

    try:
        df_history = pd.read_csv(remote_url)
    except Exception as e:
        print(f"⚠️ Не вдалося завантажити {file_path}: {e}")
        return

    if df_history.empty:
        return

    eval_file_path = "evaluation_history_xgb.csv"
    remote_eval_url = f"https://huggingface.co/datasets/{hf_repo}/raw/main/{eval_file_path}"
    try:
        df_eval_existing = pd.read_csv(remote_eval_url)
        if not df_eval_existing.empty and "actual_price" in df_eval_existing.columns:
            df_eval_existing = df_eval_existing.dropna(subset=["actual_price", "predicted_price", "mae_pct"])
            df_eval_existing = df_eval_existing[(df_eval_existing["actual_price"] > 0) & (df_eval_existing["predicted_price"] > 0)]
        processed_keys = set(df_eval_existing["eval_id"].tolist())
    except Exception:
        df_eval_existing = pd.DataFrame()
        processed_keys = set()

    tickers = df_history["ticker"].unique()
    actual_data = {}

    for ticker in tickers:
        try:
            stock_df = yf.Ticker(ticker).history(period="3mo")
            if not stock_df.empty and "Close" in stock_df:
                stock_df = stock_df.dropna(subset=["Close"])
                stock_df = stock_df[stock_df["Close"] > 0]
                stock_df.index = pd.to_datetime(stock_df.index, utc=True).normalize()
                actual_data[ticker] = stock_df["Close"]
        except Exception as e:
            print(f"⚠️ Помилка отримання котирувань для {ticker}: {e}")

    new_evaluations = []

    for _, row in df_history.iterrows():
        pred_date_str = str(row["date"])
        ticker = row["ticker"]
        current_price = float(row["current_price"])

        eval_id_1d = f"{pred_date_str}_{ticker}_1d_xgb"
        eval_id_5d = f"{pred_date_str}_{ticker}_5d_xgb"
        eval_id_20d = f"{pred_date_str}_{ticker}_20d_xgb"

        pred_date = pd.to_datetime(pred_date_str, utc=True).normalize()
        if ticker not in actual_data:
            continue

        ticker_series = actual_data[ticker]
        future_trades = ticker_series[ticker_series.index > pred_date]

        # 1d
        if eval_id_1d not in processed_keys and len(future_trades) >= 1:
            target_date_1d = future_trades.index[0]
            actual_close_1d = float(future_trades.iloc[0])
            if not pd.isna(actual_close_1d) and actual_close_1d > 0:
                pred_1d = float(row["pred_1d"])
                mae_usd = abs(actual_close_1d - pred_1d)
                mae_pct = (mae_usd / actual_close_1d) * 100
                actual_dir = 1 if actual_close_1d > current_price else (-1 if actual_close_1d < current_price else 0)
                pred_dir = 1 if pred_1d > current_price else (-1 if pred_1d < current_price else 0)
                is_correct = 1 if actual_dir == pred_dir else 0

                new_evaluations.append({
                    "eval_id": eval_id_1d,
                    "prediction_date": pred_date_str,
                    "target_date": target_date_1d.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "horizon": "1d",
                    "current_price": current_price,
                    "predicted_price": pred_1d,
                    "actual_price": actual_close_1d,
                    "mae_usd": mae_usd,
                    "mae_pct": mae_pct,
                    "direction_correct": is_correct
                })
                processed_keys.add(eval_id_1d)

        # 5d
        if eval_id_5d not in processed_keys and len(future_trades) >= 5:
            target_date_5d = future_trades.index[4]
            actual_close_5d = float(future_trades.iloc[4])
            if not pd.isna(actual_close_5d) and actual_close_5d > 0:
                pred_5d = float(row["pred_5d"])
                mae_usd = abs(actual_close_5d - pred_5d)
                mae_pct = (mae_usd / actual_close_5d) * 100
                actual_dir = 1 if actual_close_5d > current_price else (-1 if actual_close_5d < current_price else 0)
                pred_dir = 1 if pred_5d > current_price else (-1 if pred_5d < current_price else 0)
                is_correct = 1 if actual_dir == pred_dir else 0

                new_evaluations.append({
                    "eval_id": eval_id_5d,
                    "prediction_date": pred_date_str,
                    "target_date": target_date_5d.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "horizon": "5d",
                    "current_price": current_price,
                    "predicted_price": pred_5d,
                    "actual_price": actual_close_5d,
                    "mae_usd": mae_usd,
                    "mae_pct": mae_pct,
                    "direction_correct": is_correct
                })
                processed_keys.add(eval_id_5d)

        # 20d
        has_pred_20d = ("pred_20d" in row) and (not pd.isna(row["pred_20d"]))
        if has_pred_20d and eval_id_20d not in processed_keys and len(future_trades) >= 20:
            pred_20d_val = float(row["pred_20d"])
            if pred_20d_val > 0:
                target_date_20d = future_trades.index[19]
                actual_close_20d = float(future_trades.iloc[19])
                if not pd.isna(actual_close_20d) and actual_close_20d > 0:
                    mae_usd = abs(actual_close_20d - pred_20d_val)
                    mae_pct = (mae_usd / actual_close_20d) * 100
                    actual_dir = 1 if actual_close_20d > current_price else (-1 if actual_close_20d < current_price else 0)
                    pred_dir = 1 if pred_20d_val > current_price else (-1 if pred_20d_val < current_price else 0)
                    is_correct = 1 if actual_dir == pred_dir else 0

                    new_evaluations.append({
                        "eval_id": eval_id_20d,
                        "prediction_date": pred_date_str,
                        "target_date": target_date_20d.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "horizon": "20d",
                        "current_price": current_price,
                        "predicted_price": pred_20d_val,
                        "actual_price": actual_close_20d,
                        "mae_usd": mae_usd,
                        "mae_pct": mae_pct,
                        "direction_correct": is_correct
                    })
                    processed_keys.add(eval_id_20d)

    if new_evaluations:
        df_new_eval = pd.DataFrame(new_evaluations)
        df_final_eval = pd.concat([df_eval_existing, df_new_eval], ignore_index=True)

        tg_report = "⚡ КОНТРОЛЬ ЯКОСТІ ШІ (XGBOOST FEEDBACK LOOP)\n"
        tg_report += "━━━━━━━━━━━━━━━━━━━━\n"
        tg_report += f"✅ Оцінено нових дозрілих прогнозів: {len(new_evaluations)}\n"

        for horizon in ["1d", "5d", "20d"]:
            sub = df_new_eval[df_new_eval["horizon"] == horizon]
            if not sub.empty:
                avg_mae_pct = sub["mae_pct"].mean()
                win_rate = sub["direction_correct"].mean() * 100
                tg_report += f"\n🎯 *Горизонт {horizon}:*\n"
                tg_report += f"  • Середня похибка (MAE): `{avg_mae_pct:.2f}%`\n"
                tg_report += f"  • Точність напрямку (Win Rate): `{win_rate:.1f}%`\n"
                tg_report += "  • Результати по активах:\n"
                for _, r in sub.iterrows():
                    dir_emoji = "🎯" if r["direction_correct"] == 1 else "❌"
                    tg_report += f"    {dir_emoji} *{r['ticker']}*: Факт ${r['actual_price']:.2f} | ШІ ${r['predicted_price']:.2f} (MAE: {r['mae_pct']:.2f}%)\n"

        try:
            api = HfApi()
            csv_data = df_final_eval.to_csv(index=False)
            api.upload_file(
                path_or_fileobj=csv_data.encode("utf-8"),
                path_in_repo=eval_file_path,
                repo_id=hf_repo,
                repo_type="dataset",
                token=hf_token
            )
        except Exception as e:
            print(f"⚠️ Не вдалося зберегти {eval_file_path}: {e}")

        send_telegram_report(tg_report)


if __name__ == "__main__":
    evaluate_predictions()