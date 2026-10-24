import os
import warnings
import pandas as pd
from features import send_telegram_report, load_earnings_calendar, load_macro_context

warnings.simplefilter(action="ignore", category=FutureWarning)


def _load_csv_local_or_hf(local_path, hf_filename, hf_repo):
    """Читає CSV спочатку з локального диска (якщо свіжий файл щойно створений), інакше з Hugging Face."""
    if local_path and os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            if not df.empty:
                return df
        except Exception:
            pass

    if hf_repo and hf_filename:
        url = f"https://huggingface.co/datasets/{hf_repo}/raw/main/{hf_filename}"
        try:
            return pd.read_csv(url)
        except Exception as e:
            print(f"ℹ️ Не вдалося завантажити {hf_filename} з HF: {e}")

    return pd.DataFrame()


def run_arbiter():
    hf_repo = os.getenv("HF_REPO", "nadtoka/predictive-stock-dataset")
    target_tickers_env = os.getenv("STOCK_TICKER", "")
    filter_tickers = [t.strip() for t in target_tickers_env.split(",") if t.strip()]

    print("⚖️ Запуск Скрипта-Арбітра (Random Forest vs XGBoost vs Ensemble)...")

    df_eval_rf = _load_csv_local_or_hf("data/evaluation_history_rf_local.csv", "evaluation_history.csv", hf_repo)
    df_eval_xgb = _load_csv_local_or_hf("data/evaluation_history_xgb_local.csv", "evaluation_history_xgb.csv", hf_repo)

    df_pred_rf = _load_csv_local_or_hf("data/latest_predictions_rf.csv", "predictions_history.csv", hf_repo)
    df_pred_xgb = _load_csv_local_or_hf("data/latest_predictions_xgb.csv", "predictions_history_xgb.csv", hf_repo)

    # 1. Head-to-Head аналіз на спільних історичних прогнозах
    rf_map = {}
    if not df_eval_rf.empty:
        for _, r in df_eval_rf.iterrows():
            try:
                key = (str(r["prediction_date"]), str(r["ticker"]), str(r["horizon"]))
                rf_map[key] = {
                    "cur": float(r["current_price"]),
                    "pred": float(r["predicted_price"]),
                    "act": float(r["actual_price"]),
                    "mae_pct": float(r["mae_pct"]),
                    "dir": int(float(r["direction_correct"])),
                }
            except Exception:
                continue

    xgb_map = {}
    if not df_eval_xgb.empty:
        for _, r in df_eval_xgb.iterrows():
            try:
                key = (str(r["prediction_date"]), str(r["ticker"]), str(r["horizon"]))
                xgb_map[key] = {
                    "cur": float(r["current_price"]),
                    "pred": float(r["predicted_price"]),
                    "act": float(r["actual_price"]),
                    "mae_pct": float(r["mae_pct"]),
                    "dir": int(float(r["direction_correct"])),
                }
            except Exception:
                continue

    common_keys = sorted(set(rf_map.keys()) & set(xgb_map.keys()))

    # Статистика по горизонтах
    horizon_stats = {}
    for h in ["1d", "5d", "20d"]:
        h_keys = [k for k in common_keys if k[2] == h]
        if not h_keys:
            continue
        n = len(h_keys)
        rf_mae = sum(rf_map[k]["mae_pct"] for k in h_keys) / n
        xgb_mae = sum(xgb_map[k]["mae_pct"] for k in h_keys) / n
        rf_wr = sum(rf_map[k]["dir"] for k in h_keys) / n * 100.0
        xgb_wr = sum(xgb_map[k]["dir"] for k in h_keys) / n * 100.0

        ens_mae_sum = 0.0
        ens_dir_sum = 0
        rf_wins = 0
        xgb_wins = 0

        for k in h_keys:
            cur = rf_map[k]["cur"]
            act = rf_map[k]["act"]
            p_rf = rf_map[k]["pred"]
            p_xgb = xgb_map[k]["pred"]
            p_ens = 0.5 * (p_rf + p_xgb)

            ens_mae_sum += abs(act - p_ens) / act * 100.0
            act_dir = 1 if act > cur else (-1 if act < cur else 0)
            ens_dir = 1 if p_ens > cur else (-1 if p_ens < cur else 0)
            if act_dir == ens_dir:
                ens_dir_sum += 1

            if rf_map[k]["mae_pct"] < xgb_map[k]["mae_pct"]:
                rf_wins += 1
            elif xgb_map[k]["mae_pct"] < rf_map[k]["mae_pct"]:
                xgb_wins += 1

        horizon_stats[h] = {
            "n": n,
            "rf_mae": rf_mae,
            "xgb_mae": xgb_mae,
            "ens_mae": ens_mae_sum / n,
            "rf_wr": rf_wr,
            "xgb_wr": xgb_wr,
            "ens_wr": ens_dir_sum / n * 100.0,
            "rf_wins": rf_wins,
            "xgb_wins": xgb_wins,
        }

    # 2. Персональний чемпіон по кожному тікеру (на основі 1d горизонту)
    ticker_stats = {}
    all_tickers = sorted({k[1] for k in rf_map.keys()} | {k[1] for k in xgb_map.keys()})
    for t in all_tickers:
        t_keys_rf = [k for k in rf_map.keys() if k[1] == t and k[2] == "1d"]
        t_keys_xgb = [k for k in xgb_map.keys() if k[1] == t and k[2] == "1d"]
        # Беремо останні 14 спільних або доступних оцінок для актуального рейтингу
        t_common = [k for k in common_keys if k[1] == t and k[2] == "1d"][-14:]

        if t_common:
            rf_wr = sum(rf_map[k]["dir"] for k in t_common) / len(t_common) * 100.0
            xgb_wr = sum(xgb_map[k]["dir"] for k in t_common) / len(t_common) * 100.0
            rf_mae = sum(rf_map[k]["mae_pct"] for k in t_common) / len(t_common)
            xgb_mae = sum(xgb_map[k]["mae_pct"] for k in t_common) / len(t_common)
            n_eval = len(t_common)
        else:
            rf_wr = (
                sum(rf_map[k]["dir"] for k in t_keys_rf[-14:]) / len(t_keys_rf[-14:]) * 100.0 if t_keys_rf else 50.0
            )
            xgb_wr = (
                sum(xgb_map[k]["dir"] for k in t_keys_xgb[-14:]) / len(t_keys_xgb[-14:]) * 100.0
                if t_keys_xgb
                else 50.0
            )
            rf_mae = sum(rf_map[k]["mae_pct"] for k in t_keys_rf[-14:]) / len(t_keys_rf[-14:]) if t_keys_rf else 0.0
            xgb_mae = (
                sum(xgb_map[k]["mae_pct"] for k in t_keys_xgb[-14:]) / len(t_keys_xgb[-14:]) if t_keys_xgb else 0.0
            )
            n_eval = max(len(t_keys_rf[-14:]), len(t_keys_xgb[-14:]))

        if xgb_wr > rf_wr or (xgb_wr == rf_wr and xgb_mae <= rf_mae):
            winner = "⚡XGB"
            best_wr = xgb_wr
            best_mae = xgb_mae
        else:
            winner = "🌲RF"
            best_wr = rf_wr
            best_mae = rf_mae

        ticker_stats[t] = {
            "winner": winner,
            "rf_wr": rf_wr,
            "xgb_wr": xgb_wr,
            "best_wr": best_wr,
            "rf_mae": rf_mae,
            "xgb_mae": xgb_mae,
            "best_mae": best_mae,
            "n": n_eval,
        }

    # 3. Свіжі прогнози на сьогодні (беремо останній запис по кожному тікеру індивідуально)
    latest_rf = {}
    if not df_pred_rf.empty and "ticker" in df_pred_rf.columns:
        for _, r in df_pred_rf.sort_values("date").groupby("ticker").tail(1).iterrows():
            latest_rf[str(r["ticker"])] = r

    latest_xgb = {}
    if not df_pred_xgb.empty and "ticker" in df_pred_xgb.columns:
        for _, r in df_pred_xgb.sort_values("date").groupby("ticker").tail(1).iterrows():
            latest_xgb[str(r["ticker"])] = r

    # Якщо у файлі є одиничні тестові рядки новішої дати (наприклад, CI smoke test на 1-3 акції),
    # синхронізуємо по даті, де присутні обидві моделі для цього тікера
    rf_by_date_ticker = {}
    if not df_pred_rf.empty and "ticker" in df_pred_rf.columns:
        for _, r in df_pred_rf.iterrows():
            rf_by_date_ticker[(str(r["date"]), str(r["ticker"]))] = r

    xgb_by_date_ticker = {}
    if not df_pred_xgb.empty and "ticker" in df_pred_xgb.columns:
        for _, r in df_pred_xgb.iterrows():
            xgb_by_date_ticker[(str(r["date"]), str(r["ticker"]))] = r

    active_tickers = filter_tickers if filter_tickers else sorted(set(latest_rf.keys()) | set(latest_xgb.keys()))

    # Визначаємо основну дату звіту (мода/найчастіша остання дата серед усіх тікерів)
    all_last_dates = [str(r["date"]) for r in latest_xgb.values()] + [str(r["date"]) for r in latest_rf.values()]
    report_date = max(set(all_last_dates), key=all_last_dates.count) if all_last_dates else ""

    tg = f"⚖️ *ШІ-АРБІТР & ДАЙДЖЕСТ РИНКУ* `{report_date}`\n"
    tg += "━━━━━━━━━━━━━━━━━━━━\n"

    if horizon_stats:
        tg += "🏆 *Битва моделей (Head-to-Head):*\n"
        for h in ["1d", "5d", "20d"]:
            if h not in horizon_stats:
                continue
            s = horizon_stats[h]
            champ = "⚡XGB" if (s["xgb_wr"] > s["rf_wr"] or s["xgb_mae"] < s["rf_mae"]) else "🌲RF"
            tg += (
                f"• *{h}* ({s['n']} пар): "
                f"🌲RF `{s['rf_wr']:.1f}%` (MAE `{s['rf_mae']:.2f}%`) vs "
                f"⚡XGB `{s['xgb_wr']:.1f}%` (MAE `{s['xgb_mae']:.2f}%`) | "
                f"🤝Ансамбль: MAE `{s['ens_mae']:.2f}%` 👉 Лідер: *{champ}*\n"
            )
        tg += "\n"

    # 4. Свіжі результати аудиту (якщо сьогодні дозріли нові перевірки)
    latest_eval_date = ""
    if not df_eval_rf.empty and "target_date" in df_eval_rf.columns:
        latest_eval_date = str(df_eval_rf["target_date"].max())
    if not df_eval_xgb.empty and "target_date" in df_eval_xgb.columns:
        latest_eval_date = max(latest_eval_date, str(df_eval_xgb["target_date"].max()))

    if latest_eval_date:
        sub_rf_today = (
            df_eval_rf[(df_eval_rf["target_date"].astype(str) == latest_eval_date) & (df_eval_rf["horizon"] == "1d")]
            if not df_eval_rf.empty
            else pd.DataFrame()
        )
        sub_xgb_today = (
            df_eval_xgb[(df_eval_xgb["target_date"].astype(str) == latest_eval_date) & (df_eval_xgb["horizon"] == "1d")]
            if not df_eval_xgb.empty
            else pd.DataFrame()
        )
        if not sub_rf_today.empty or not sub_xgb_today.empty:
            tg += f"🎯 *Останній аудит 1d ({latest_eval_date}):*"
            if not sub_rf_today.empty:
                wr_t_rf = sub_rf_today["direction_correct"].mean() * 100.0
                mae_t_rf = sub_rf_today["mae_pct"].mean()
                tg += f" 🌲RF `{wr_t_rf:.0f}%` (MAE `{mae_t_rf:.2f}%`)"
            if not sub_xgb_today.empty:
                wr_t_xgb = sub_xgb_today["direction_correct"].mean() * 100.0
                mae_t_xgb = sub_xgb_today["mae_pct"].mean()
                tg += f" | ⚡XGB `{wr_t_xgb:.0f}%` (MAE `{mae_t_xgb:.2f}%`)"
            tg += "\n\n"

    # Секція свіжих прогнозів: ділимо на ТОП-Консенсус (обидві моделі згодні + високий WR) та загальний список
    strong_signals = []
    regular_signals = []

    for t in active_tickers:
        r_rf = latest_rf.get(t)
        r_xgb = latest_xgb.get(t)
        if r_rf is None and r_xgb is None:
            continue

        # Якщо в одній моделі дата новіша (наприклад, CI smoke test), пробуємо знайти пару на спільну дату
        if r_rf is not None and r_xgb is not None and str(r_rf["date"]) != str(r_xgb["date"]):
            common_d = min(str(r_rf["date"]), str(r_xgb["date"]))
            if (common_d, t) in rf_by_date_ticker and (common_d, t) in xgb_by_date_ticker:
                r_rf = rf_by_date_ticker[(common_d, t)]
                r_xgb = xgb_by_date_ticker[(common_d, t)]
            elif str(r_rf["date"]) > str(r_xgb["date"]):
                r_xgb = None
            else:
                r_rf = None

        # Відсікаємо старі архівні тікери (наприклад, DJT чи QQQM з літа), яких більше немає в поточному тижні
        rec_date = str(r_rf["date"]) if r_rf is not None else str(r_xgb["date"])
        if report_date and rec_date < report_date and not filter_tickers:
            continue

        cur = float(r_rf["current_price"]) if r_rf is not None else float(r_xgb["current_price"])
        st = ticker_stats.get(
            t,
            {"winner": "⚡XGB", "rf_wr": 50.0, "xgb_wr": 50.0, "best_wr": 50.0, "best_mae": 0.0, "n": 0},
        )

        p1_rf = float(r_rf["pred_1d"]) if r_rf is not None and not pd.isna(r_rf["pred_1d"]) else None
        p5_rf = float(r_rf["pred_5d"]) if r_rf is not None and not pd.isna(r_rf["pred_5d"]) else None
        p20_rf = (
            float(r_rf["pred_20d"])
            if r_rf is not None and "pred_20d" in r_rf and not pd.isna(r_rf["pred_20d"])
            else None
        )

        p1_xgb = float(r_xgb["pred_1d"]) if r_xgb is not None and not pd.isna(r_xgb["pred_1d"]) else None
        p5_xgb = float(r_xgb["pred_5d"]) if r_xgb is not None and not pd.isna(r_xgb["pred_5d"]) else None
        p20_xgb = (
            float(r_xgb["pred_20d"])
            if r_xgb is not None and "pred_20d" in r_xgb and not pd.isna(r_xgb["pred_20d"])
            else None
        )

        # Ваговий консенсус (Ensemble) на основі останніх Win Rate обох моделей
        w_rf = max(0.3, st["rf_wr"] / 100.0) if st["n"] >= 5 else 0.5
        w_xgb = max(0.3, st["xgb_wr"] / 100.0) if st["n"] >= 5 else 0.5

        def _blend(v_rf, v_xgb):
            if v_rf is not None and v_xgb is not None:
                return (v_rf * w_rf + v_xgb * w_xgb) / (w_rf + w_xgb)
            return v_xgb if v_xgb is not None else v_rf

        ens_1d = _blend(p1_rf, p1_xgb)
        ens_5d = _blend(p5_rf, p5_xgb)
        ens_20d = _blend(p20_rf, p20_xgb)

        if ens_1d is None:
            continue

        pct_1d = (ens_1d - cur) / cur * 100.0
        pct_5d = (ens_5d - cur) / cur * 100.0 if ens_5d else 0.0
        pct_20d = (ens_20d - cur) / cur * 100.0 if ens_20d else 0.0

        agree_1d = False
        if p1_rf is not None and p1_xgb is not None:
            agree_1d = (p1_rf > cur and p1_xgb > cur) or (p1_rf < cur and p1_xgb < cur)

        dir_icon = "🟢▲" if pct_1d > 0 else "🔴▼"
        agree_tag = "🤝" if agree_1d else "⚔️"

        # Перевіряємо, чи є найближчим часом квартальний звіт (Earnings Call) у локальному кеші
        earn_tag = ""
        ed = load_earnings_calendar(t, "data")
        if ed is not None and not ed.empty:
            try:
                now_ts = pd.Timestamp.now(tz="UTC").normalize()
                future_ed = ed.index[ed.index >= now_ts]
                if len(future_ed) > 0:
                    days_left = int((future_ed.min() - now_ts).days)
                    if 0 <= days_left <= 7:
                        earn_tag = f" 📅Звіт:{days_left}д!"
            except Exception:
                pass

        line = (
            f"{dir_icon} *{t}* `${cur:.2f}` → 1d:`{pct_1d:+.1f}%` (${ens_1d:.2f}) | "
            f"5d:`{pct_5d:+.1f}%` | 20d:`{pct_20d:+.1f}%` "
            f"[{agree_tag}{st['winner']} WR:{st['best_wr']:.0f}%]{earn_tag}"
        )

        if agree_1d and st["best_wr"] >= 60.0 and st["n"] >= 5:
            strong_signals.append((st["best_wr"], line))
        else:
            regular_signals.append((st["best_wr"], line))

    strong_signals.sort(key=lambda x: x[0], reverse=True)
    regular_signals.sort(key=lambda x: x[0], reverse=True)

    if strong_signals:
        tg += f"🔥 *ТОП КОНСЕНСУС-СИГНАЛИ (Обидві згодні 🤝 + WR ≥ 60%):*\n"
        for _, l in strong_signals:
            tg += f"{l}\n"
        tg += "\n"

    if regular_signals:
        tg += f"📋 *Інші активи в моніторингу ({len(regular_signals)}):*\n"
        for _, l in regular_signals:
            tg += f"{l}\n"

    tg += "\n💡 _Позначки: 🤝 моделі згодні по 1d | ⚔️ моделі розійшлись | в дужках лідер точності по акції за 14д._"

    print("\n" + tg + "\n")
    send_telegram_report(tg)


if __name__ == "__main__":
    run_arbiter()
