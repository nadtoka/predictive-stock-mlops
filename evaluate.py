import os
import warnings
import pandas as pd
import yfinance as yf
from huggingface_hub import HfApi
from curl_cffi import requests

# Глушимо FutureWarning від Pandas/yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)

def send_telegram_report(text):
    """Надсилає звіт в Телеграм з автоматичним розбиттям на безпечні чанки за рядками"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("ℹ️ TELEGRAM_BOT_TOKEN або TELEGRAM_CHAT_ID не знайдені. Пропускаємо.")
        return
        
    MAX_LEN = 4000
    chunks = []
    current_chunk = ""
    
    # Розбиваємо строго по рядках (\n) для ліквідації пастки довгих повідомлень
    lines = text.split("\n")
    for line in lines:
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
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "Markdown"
        }
        try:
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code != 200:
                print(f"❌ Помилка Telegram API: {res.text}")
        except Exception as e:
            print(f"❌ Не вдалося зв'язатися з Telegram: {e}")

def evaluate_predictions():
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")
    
    if not hf_token or not hf_repo:
        print("❌ Помилка: HF_TOKEN або HF_REPO не знайдені в змінних оточення.")
        return

    file_path = "predictions_history.csv"
    remote_url = f"https://huggingface.co/datasets/{hf_repo}/raw/main/{file_path}"

    try:
        df_history = pd.read_csv(remote_url)
        print(f"📜 Лог прогнозів успішно завантажено. Знайдено рядків: {len(df_history)}")
    except Exception as e:
        print(f"⚠️ Не вдалося завантажити predictions_history.csv з HF: {e}")
        return

    if df_history.empty:
        print("ℹ️ Історія прогнозів порожня. Оцінювати нічого.")
        return

    # Завантажуємо існуючий лог метрик, щоб не дублювати перевірки
    eval_file_path = "evaluation_history.csv"
    remote_eval_url = f"https://huggingface.co/datasets/{hf_repo}/raw/main/{eval_file_path}"
    try:
        df_eval_existing = pd.read_csv(remote_eval_url)
        processed_keys = set(df_eval_existing["eval_id"].tolist())
        print(f"📋 Знайдено базу аудиту. Вже перевірено сутностей: {len(processed_keys)}")
    except Exception:
        df_eval_existing = pd.DataFrame()
        processed_keys = set()
        print("✨ База аудиту не знайдена. Буде створено новий файл метрик...")

    # Оптимізуємо запити до Yahoo Finance (групуємо унікальні тікери)
    tickers = df_history["ticker"].unique()
    actual_data = {}
    
    print("📡 Завантаження реальних історичних цін закриття з yfinance...")
    for ticker in tickers:
        try:
            stock_df = yf.Ticker(ticker).history(period="1mo")
            if not stock_df.empty:
                stock_df.index = pd.to_datetime(stock_df.index, utc=True).normalize()
                actual_data[ticker] = stock_df["Close"]
        except Exception as e:
            print(f"⚠️ Не вдалося отримати реальні котирування для {ticker}: {e}")

    new_evaluations = []
    
    # Парсимо історію прогнозів
    for _, row in df_history.iterrows():
        pred_date_str = str(row["date"])
        ticker = row["ticker"]
        current_price = float(row["current_price"])
        
        eval_id_1d = f"{pred_date_str}_{ticker}_1d"
        eval_id_5d = f"{pred_date_str}_{ticker}_5d"

        pred_date = pd.to_datetime(pred_date_str, utc=True).normalize()
        
        if ticker not in actual_data:
            continue
            
        ticker_series = actual_data[ticker]
        
        # Вираховуємо точні бізнес-дні, коли прогноз мав закритися
        target_date_1d = pred_date + pd.offsets.BusinessDay(1)
        target_date_5d = pred_date + pd.offsets.BusinessDay(5)

        # 1. Валідація 1-денного прогнозу
        if eval_id_1d not in processed_keys and target_date_1d in ticker_series.index:
            actual_close_1d = float(ticker_series.loc[target_date_1d])
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

        # 2. Валідація 5-денного прогнозу
        if eval_id_5d not in processed_keys and target_date_5d in ticker_series.index:
            actual_close_5d = float(ticker_series.loc[target_date_5d])
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

    if new_evaluations:
        df_new_eval = pd.DataFrame(new_evaluations)
        df_final_eval = pd.concat([df_eval_existing, df_new_eval], ignore_index=True)
        
        tg_report = "📊 КОНТРОЛЬ ЯКОСТІ ШІ (FEEDBACK LOOP)\n"
        tg_report += "━━━━━━━━━━━━━━━━━━━━\n"
        tg_report += f"✅ Оцінено нових дозрілих прогнозів: {len(new_evaluations)}\n"
        
        for horizon in ["1d", "5d"]:
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
            print("💾 Синхронізаціяльної матриці оцінки з Hugging Face...")
            api = HfApi()
            csv_data = df_final_eval.to_csv(index=False)
            api.upload_file(
                path_or_fileobj=csv_data.encode("utf-8"),
                path_in_repo=eval_file_path,
                repo_id=hf_repo,
                repo_type="dataset",
                token=hf_token
            )
            print("✅ Матрицю успішно засинкронено в Hugging Face Datasets!")
        except Exception as e:
            print(f"⚠️ Не вдалося зберегти базу оцінки на HF: {e}")
            tg_report += "\n\n⚠️ *Hugging Face:* Помилка синхронізації бази."
            
        send_telegram_report(tg_report)
    else:
        print("ℹ️ Немає нових дозрілих прогнозів для аналізу на сьогодні.")

if __name__ == "__main__":
    evaluate_predictions()