"""
╔══════════════════════════════════════════════════════════════════╗
║     MNQ/NQ PREDICTOR — ML DASHBOARD v4.0                        ║
║     Deep OLED Dark · TradingView-style · Multi-Timeframe        ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    streamlit run live-nq.py

Dependencies:
    pip install streamlit streamlit-lightweight-charts-ntf yfinance xgboost
                lightgbm scikit-learn scipy requests beautifulsoup4
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import json
import logging
import hashlib
import pickle
import traceback
import time
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import streamlit as st

import yfinance as yf
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

import requests
from bs4 import BeautifulSoup

# ── Logging ─────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("NQ_PREDICTOR")

# ── Storage ──────────────────────────────────────────────────────
DATA_DIR = Path("nq_data")
DATA_DIR.mkdir(exist_ok=True)
MODEL_CACHE = DATA_DIR / "model_v4.pkl"

SYMBOLS = {
    "NQ":    "NQ=F",
    "VIX":   "^VIX",
    "DXY":   "DX-Y.NYB",
    "US10Y": "^TNX",
    "QQQ":   "QQQ",
    "ES":    "ES=F",
}

# ── Timeframe config ─────────────────────────────────────────────
TF_CONFIG = {
    "1M":  {"interval": "1m",  "period": "5d",   "proj_bars": 30,  "label": "1 Minute"},
    "5M":  {"interval": "5m",  "period": "7d",   "proj_bars": 24,  "label": "5 Minutes"},
    "15M": {"interval": "15m", "period": "30d",  "proj_bars": 16,  "label": "15 Minutes"},
    "30M": {"interval": "30m", "period": "60d",  "proj_bars": 12,  "label": "30 Minutes"},
    "1H":  {"interval": "1h",  "period": "60d",  "proj_bars": 8,   "label": "1 Hour"},
    "4H":  {"interval": "1h",  "period": "60d",  "proj_bars": 6,   "label": "4 Hours",   "resample": "4h"},
    "D":   {"interval": "1d",  "period": "2y",   "proj_bars": 5,   "label": "Daily"},
}

# ── Data classes ─────────────────────────────────────────────────
@dataclass
class Prediction:
    bias: str = "NEUTRAL"
    prob_bull: float = 0.50
    prob_bear: float = 0.50
    confidence: float = 0.50
    current_price: float = 0.0
    atr: float = 0.0
    upper_t1_lo: float = 0.0
    upper_t1_hi: float = 0.0
    upper_t2_lo: float = 0.0
    upper_t2_hi: float = 0.0
    lower_t1_lo: float = 0.0
    lower_t1_hi: float = 0.0
    lower_t2_lo: float = 0.0
    lower_t2_hi: float = 0.0
    upper_t1_conf: float = 0.0
    upper_t2_conf: float = 0.0
    lower_t1_conf: float = 0.0
    lower_t2_conf: float = 0.0
    regime: str = "Normal Market"
    regime_phase: str = "Trend"
    vix: float = 20.0
    vol_regime: str = "Normal"
    trend_prob: float = 0.50
    session_outlook: Dict[str, str] = field(default_factory=dict)
    high_time_est: str = "12:00"
    low_time_est: str = "15:00"
    wf_accuracy: float = 0.0
    wf_auc: float = 0.0
    delta_pct: float = 0.0
    rsi: float = 50.0
    momentum: str = "NEUTRAL"


# ════════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ════════════════════════════════════════════════════════════════
class DataPipeline:
    _cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
    TTL = 300  # 5 min cache

    @classmethod
    def fetch(cls, symbol: str, period: str = "2y",
              interval: str = "1d") -> Optional[pd.DataFrame]:
        key = f"{symbol}_{period}_{interval}"
        if key in cls._cache:
            df, ts = cls._cache[key]
            if (datetime.now() - ts).seconds < cls.TTL:
                return df.copy()
        try:
            df = yf.Ticker(symbol).history(
                period=period, interval=interval, auto_adjust=True)
            if df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
            df = df[df.index <= datetime.now()]
            cls._cache[key] = (df.copy(), datetime.now())
            return df
        except Exception as e:
            logger.warning(f"Fetch error {symbol}: {e}")
            return None

    @classmethod
    def fetch_correlated(cls) -> Dict[str, Optional[pd.DataFrame]]:
        return {k: cls.fetch(v, "2y", "1d")
                for k, v in SYMBOLS.items() if k != "NQ"}

    @classmethod
    def fetch_for_timeframe(cls, tf: str) -> Optional[pd.DataFrame]:
        """Fetch and optionally resample data for the given timeframe key."""
        cfg = TF_CONFIG[tf]
        interval = cfg["interval"]
        period   = cfg["period"]
        df = cls.fetch(SYMBOLS["NQ"], period, interval)
        if df is None or df.empty:
            return None
        # Resample for 4H (yfinance doesn't support it natively)
        if cfg.get("resample"):
            rule = cfg["resample"]
            df = df.resample(rule).agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            }).dropna(subset=["open", "close"])
        return df


# ════════════════════════════════════════════════════════════════
#  FEATURE ENGINE
# ════════════════════════════════════════════════════════════════
class FeatureEngine:

    @staticmethod
    def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(),
                        (l - c.shift(1)).abs()], axis=1).max(axis=1)
        return tr.rolling(p).mean()

    @staticmethod
    def rsi(s: pd.Series, p: int = 14) -> pd.Series:
        d = s.diff()
        g = d.clip(lower=0).rolling(p).mean()
        ls = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - (100 / (1 + g / (ls + 1e-9)))

    @staticmethod
    def adx(df: pd.DataFrame, p: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        up = h.diff(); dn = -l.diff()
        pdm = np.where((up > dn) & (up > 0), up, 0.0)
        ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
        a = FeatureEngine.atr(df, p)
        pdi = 100 * pd.Series(pdm, index=df.index).rolling(p).mean() / (a + 1e-9)
        ndi = 100 * pd.Series(ndm, index=df.index).rolling(p).mean() / (a + 1e-9)
        dx = (np.abs(pdi - ndi) / (pdi + ndi + 1e-9)) * 100
        return dx.rolling(p).mean()

    @classmethod
    def build(cls, df: pd.DataFrame,
              corr: Optional[Dict] = None) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        c = df["close"]

        for n in [1, 2, 5, 10, 20]:
            f[f"ret_{n}d"] = c.pct_change(n)
        f["log_ret"] = np.log(c / c.shift(1))
        f["gap"] = (df["open"] - c.shift(1)) / (c.shift(1) + 1e-9)
        f["body"] = (c - df["open"]) / (df["open"] + 1e-9)
        f["range"] = (df["high"] - df["low"]) / (c + 1e-9)

        body_sz = (c - df["open"]).abs()
        f["upper_wick"] = (df["high"] - df[["open","close"]].max(axis=1)) / (body_sz + 1e-9)
        f["lower_wick"] = (df[["open","close"]].min(axis=1) - df["low"]) / (body_sz + 1e-9)

        atr14 = cls.atr(df, 14)
        atr7  = cls.atr(df, 7)
        f["atr14"] = atr14 / (c + 1e-9)
        f["atr7"]  = atr7  / (c + 1e-9)
        f["vol10"] = f["ret_1d"].rolling(10).std()
        f["vol20"] = f["ret_1d"].rolling(20).std()
        f["vol_ratio"] = f["vol10"] / (f["vol20"] + 1e-9)
        f["vol_regime"] = (f["vol10"] > f["vol10"].rolling(60).quantile(0.75)).astype(int)

        rsi14 = cls.rsi(c, 14)
        f["rsi14"] = rsi14
        f["rsi7"]  = cls.rsi(c, 7)
        f["rsi_slope"] = rsi14.diff(3)

        ema9   = c.ewm(span=9).mean()
        ema21  = c.ewm(span=21).mean()
        ema50  = c.ewm(span=50).mean()
        ema200 = c.ewm(span=200).mean()
        for e, s in [(9, ema9), (21, ema21), (50, ema50), (200, ema200)]:
            f[f"ema{e}_dist"] = (c - s) / (s + 1e-9)

        f["ema9_21_cross"]  = (ema9 - ema21) / (c + 1e-9)
        f["ema21_50_cross"] = (ema21 - ema50) / (c + 1e-9)

        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig  = macd.ewm(span=9).mean()
        f["macd_hist"]  = macd - sig
        f["macd_slope"] = (macd - sig).diff()

        bb_m = c.rolling(20).mean()
        bb_s = c.rolling(20).std()
        f["bb_pos"]   = (c - (bb_m - 2*bb_s)) / ((4*bb_s) + 1e-9)
        f["bb_width"] = (4*bb_s) / (bb_m + 1e-9)

        f["adx14"]   = cls.adx(df, 14)
        f["trending"] = (f["adx14"] > 25).astype(int)

        for n in [5, 10, 20]:
            f[f"mom{n}"] = c / c.shift(n) - 1

        for p in [5, 10, 20]:
            f[f"dist_hi{p}"] = (df["high"].rolling(p).max() - c) / (c + 1e-9)
            f[f"dist_lo{p}"] = (c - df["low"].rolling(p).min()) / (c + 1e-9)

        f["dow"]      = df.index.dayofweek
        f["month"]    = df.index.month
        f["is_mon"]   = (df.index.dayofweek == 0).astype(int)
        f["is_fri"]   = (df.index.dayofweek == 4).astype(int)
        f["month_end"] = df.index.is_month_end.astype(int)
        f["qtr_end"]   = df.index.is_quarter_end.astype(int)

        if corr:
            for name, cdf in corr.items():
                if cdf is not None and not cdf.empty:
                    try:
                        cr = cdf["close"].pct_change(1).reindex(df.index, method="ffill")
                        f[f"ret_{name}"] = cr
                        f[f"corr_{name}"] = f["ret_1d"].rolling(20).corr(cr)
                    except Exception:
                        pass

        f["label"] = (c.shift(-1) > c).astype(int)
        f = f.replace([np.inf, -np.inf], np.nan)
        f = f.dropna(thresh=int(len(f.columns) * 0.5))
        return f


# ════════════════════════════════════════════════════════════════
#  ML ENSEMBLE — Walk-Forward Validated
# ════════════════════════════════════════════════════════════════
class MLEnsemble:

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.wf_metrics: Dict[str, float] = {}
        self.importances: Dict[str, float] = {}

    def _base_models(self) -> Dict[str, Any]:
        return {
            "xgb": xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.04,
                subsample=0.75, colsample_bytree=0.65,
                min_child_weight=8, reg_alpha=0.2, reg_lambda=1.5,
                eval_metric="logloss", verbosity=0,
                random_state=42, n_jobs=-1),
            "lgb": lgb.LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.04,
                subsample=0.75, colsample_bytree=0.65,
                min_child_samples=25, reg_alpha=0.2, reg_lambda=1.5,
                random_state=42, n_jobs=-1, verbose=-1),
            "rf": RandomForestClassifier(
                n_estimators=300, max_depth=5, min_samples_leaf=15,
                max_features="sqrt", random_state=42, n_jobs=-1),
            "et": ExtraTreesClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=15,
                max_features="sqrt", random_state=42, n_jobs=-1),
            "lr": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        }

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > 0.92)]
        X2 = X.drop(columns=drop, errors="ignore")
        rf = RandomForestClassifier(n_estimators=80, max_depth=4,
                                    random_state=42, n_jobs=-1)
        rf.fit(X2.fillna(0), y)
        imp = pd.Series(rf.feature_importances_, index=X2.columns)
        return imp.nlargest(min(45, len(imp))).index.tolist()

    def walk_forward(self, feat: pd.DataFrame, n_splits: int = 6,
                     min_train: int = 250) -> Dict[str, float]:
        df = feat.dropna()
        if len(df) < min_train + 60:
            return {"accuracy": 0.5, "auc": 0.5}
        X = df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number])
        y = df["label"]
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)
        accs, aucs = [], []
        for train_idx, test_idx in tscv.split(X):
            if len(train_idx) < min_train:
                continue
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            sel = self._select_features(Xtr.fillna(0), ytr)
            Xtr, Xte = Xtr[sel].fillna(0), Xte[sel].fillna(0)
            sc = RobustScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)
            fold_probs = []
            for m in list(self._base_models().values()):
                try:
                    m.fit(Xtr_s, ytr)
                    fold_probs.append(m.predict_proba(Xte_s)[:, 1])
                except Exception:
                    pass
            if not fold_probs:
                continue
            avg = np.mean(fold_probs, axis=0)
            accs.append(accuracy_score(yte, (avg > 0.5).astype(int)))
            try:
                aucs.append(roc_auc_score(yte, avg))
            except Exception:
                aucs.append(0.5)
        return {
            "accuracy": float(np.mean(accs)) if accs else 0.5,
            "auc":      float(np.mean(aucs)) if aucs else 0.5,
        }

    def train(self, feat: pd.DataFrame) -> Dict[str, float]:
        df = feat.dropna().copy()
        if len(df) < 150:
            return {}
        X = df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number])
        y = df["label"]
        sel = self._select_features(X.fillna(0), y)
        self.feature_names = sel
        X = X[sel].fillna(0)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        wf = self.walk_forward(feat)
        self.wf_metrics = wf
        for name, model in self._base_models().items():
            try:
                model.fit(Xs, y)
                self.models[name] = model
            except Exception as e:
                logger.warning(f"Train {name}: {e}")
        imps: Dict[str, List[float]] = {f: [] for f in self.feature_names}
        for m in self.models.values():
            if hasattr(m, "feature_importances_"):
                for fn, iv in zip(self.feature_names, m.feature_importances_):
                    imps[fn].append(iv)
        self.importances = {k: float(np.mean(v)) for k, v in imps.items() if v}
        self.is_trained = True
        return wf

    def predict(self, row: Dict[str, float]) -> Tuple[float, float]:
        if not self.is_trained:
            return 0.5, 0.3
        try:
            x = pd.DataFrame([row])
            for col in self.feature_names:
                if col not in x.columns:
                    x[col] = 0.0
            x = x[self.feature_names].fillna(0)
            xs = self.scaler.transform(x)
            probs = []
            for m in self.models.values():
                try:
                    probs.append(m.predict_proba(xs)[0][1])
                except Exception:
                    pass
            if not probs:
                return 0.5, 0.3
            mean_p = float(np.mean(probs))
            std_p  = float(np.std(probs)) if len(probs) > 1 else 0.2
            conf   = max(0.1, min(0.95, 1.0 - std_p * 4))
            return mean_p, conf
        except Exception:
            return 0.5, 0.3


# ════════════════════════════════════════════════════════════════
#  REGIME DETECTOR
# ════════════════════════════════════════════════════════════════
class RegimeDetector:

    @staticmethod
    def detect(df: pd.DataFrame,
               vix_df: Optional[pd.DataFrame] = None) -> Dict:
        try:
            c = df["close"]
            rets = c.pct_change()
            adx_s = FeatureEngine.adx(df, 14)
            adx_v = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 20.0

            vol10 = rets.rolling(10).std().iloc[-1]
            q25 = rets.rolling(252).std().quantile(0.25)
            q75 = rets.rolling(252).std().quantile(0.75)
            q90 = rets.rolling(252).std().quantile(0.90)

            vol_regime = ("Extreme" if vol10 > q90
                          else "High"   if vol10 > q75
                          else "Low"    if vol10 < q25
                          else "Normal")

            vix = 20.0
            risk_mode = "Neutral"
            if vix_df is not None and not vix_df.empty:
                vix = float(vix_df["close"].iloc[-1])
                risk_mode = ("Risk-Off" if vix > 30
                              else "Neutral" if vix > 20
                              else "Risk-On")

            trend_prob = (0.75 if adx_v > 30
                          else 0.55 if adx_v > 20
                          else 0.25)

            last_10 = rets.tail(10).mean()
            bb_m  = c.rolling(20).mean()
            bb_s  = c.rolling(20).std()
            bb_pos = float((c.iloc[-1] - (bb_m.iloc[-1] - 2*bb_s.iloc[-1]))
                           / (4*bb_s.iloc[-1] + 1e-9))

            if adx_v > 25 and abs(last_10) > 0:
                phase = "Trend"
            elif adx_v < 15:
                phase = "Chop"
            elif bb_pos > 0.85:
                phase = "Distribution"
            elif bb_pos < 0.15:
                phase = "Accumulation"
            else:
                phase = "Consolidation"

            label_map = {
                ("High",    "Risk-Off"): "Crisis / Panic",
                ("Extreme", "Risk-Off"): "Extreme Panic",
                ("Normal",  "Risk-On"):  "Bull Trend",
                ("Low",     "Risk-On"):  "Low-Vol Bull",
                ("High",    "Neutral"):  "High-Vol Chop",
                ("Normal",  "Neutral"):  "Normal Market",
                ("Low",     "Neutral"):  "Low-Vol Range",
            }
            regime_label = label_map.get((vol_regime, risk_mode),
                                         f"{vol_regime} Vol / {risk_mode}")

            return {
                "label": regime_label,
                "phase": phase,
                "vix": vix,
                "vol_regime": vol_regime,
                "risk_mode": risk_mode,
                "adx": adx_v,
                "trend_prob": trend_prob,
            }
        except Exception:
            return {"label": "Normal Market", "phase": "Trend",
                    "vix": 20.0, "vol_regime": "Normal",
                    "risk_mode": "Neutral", "adx": 20.0, "trend_prob": 0.5}


# ════════════════════════════════════════════════════════════════
#  TIME-OF-DAY ESTIMATOR
# ════════════════════════════════════════════════════════════════
class TimeEstimator:
    HIGH_DIST = {
        9: 0.12, 10: 0.10, 11: 0.08, 12: 0.06,
        13: 0.09, 14: 0.07, 15: 0.06, 16: 0.08,
        2: 0.06, 3: 0.06, 4: 0.06, 8: 0.08,
        17: 0.05, 18: 0.04, 6: 0.04, 7: 0.05,
    }
    LOW_DIST = {
        9: 0.10, 10: 0.09, 11: 0.07, 12: 0.08,
        13: 0.12, 14: 0.09, 15: 0.07, 16: 0.06,
        2: 0.04, 3: 0.04, 4: 0.04, 8: 0.07,
        17: 0.06, 18: 0.04, 6: 0.03, 7: 0.05,
    }

    @classmethod
    def estimate(cls, bias: str, regime_phase: str,
                 vol_regime: str) -> Tuple[str, str]:
        h_dist = dict(cls.HIGH_DIST)
        l_dist = dict(cls.LOW_DIST)
        if bias == "BULLISH":
            for h in [9, 10]: l_dist[h] = l_dist.get(h, 0) * 1.4
            for h in [13, 14, 15]: h_dist[h] = h_dist.get(h, 0) * 1.3
        elif bias == "BEARISH":
            for h in [9, 10]: h_dist[h] = h_dist.get(h, 0) * 1.4
            for h in [13, 14, 15]: l_dist[h] = l_dist.get(h, 0) * 1.3
        if regime_phase == "Trend":
            h_dist[9] = h_dist.get(9, 0) * 1.2
            l_dist[9] = l_dist.get(9, 0) * 1.2

        def pick_hour(dist: Dict) -> int:
            total = sum(dist.values())
            probs = {k: v/total for k, v in dist.items()}
            hours = list(probs.keys())
            return hours[int(np.argmax([probs[h] for h in hours]))]

        return f"{pick_hour(h_dist):02d}:00 NY", f"{pick_hour(l_dist):02d}:00 NY"


# ════════════════════════════════════════════════════════════════
#  SESSION OUTLOOK — AI-style analysis
# ════════════════════════════════════════════════════════════════
def build_session_outlook(bias: str, regime: Dict, rsi: float) -> Dict[str, str]:
    b   = bias
    vr  = regime.get("vol_regime", "Normal")
    phase = regime.get("phase", "Trend")
    adx = regime.get("adx", 20.0)
    risk = regime.get("risk_mode", "Neutral")

    def asia_text() -> str:
        if vr in ("High", "Extreme"):
            return (
                f"Overnight Asia session is expected to carry elevated volatility — a direct "
                f"consequence of the current {vr.lower()} volatility regime. Expect a session "
                f"range of roughly 60–90 pts as institutional flows continue to reprice risk. "
                f"Traders should be cautious of false breakouts during thin liquidity windows "
                f"(approx. 20:00–01:00 NY). Key Asian support/resistance levels may be tested "
                f"aggressively before NY open reasserts direction."
            )
        elif vr == "Low":
            return (
                "Asia overnight session is anticipated to be exceptionally quiet, reflecting the "
                "current low-volatility regime and compressed ATR. Range likely 15–30 pts with "
                "price coiling near the prior session close. Watch for a potential stop-hunt at "
                "obvious levels before the London open injects meaningful liquidity. No directional "
                "edge is recommended during this window — patience pays."
            )
        else:
            return (
                f"Asia session sets up as a controlled ranging environment with moderate liquidity. "
                f"Current regime ({vr} vol) suggests typical overnight ranges of 30–55 pts around "
                f"the prior day's close. The ML bias is {b}, which may result in a slight directional "
                f"drift toward the {'upside' if b == 'BULLISH' else 'downside' if b == 'BEARISH' else 'median'}. "
                f"Fake breakouts of Asian range highs/lows are historically common and often precede "
                f"the true London/NY impulse move."
            )

    def london_text() -> str:
        if b == "BULLISH":
            return (
                "The London open is statistically the session most likely to establish the intraday "
                "low on a bullish ML day. Expect an initial 'Judas swing' — a brief dip below the "
                f"Asian range low designed to trigger stop-losses — before a sustained push higher "
                f"into the NY overlap. With RSI currently at {rsi:.0f} and a {phase} regime, London "
                "buyers are likely to step in aggressively near Asian lows. Watch for the bullish "
                "engulfing or displacement candle as the entry signal for long positions."
            )
        elif b == "BEARISH":
            return (
                "London open on a bearish ML day historically shows an initial push higher — a "
                "'Judas rally' — before the real downside move develops. With RSI at "
                f"{rsi:.0f} and a {vr.lower()} volatility environment, this false rally is likely "
                "contained within 40–70 pts above the Asia close before sellers reassert control. "
                "Look for bearish market structure shifts on the 15M–1H chart between 03:00–06:00 NY "
                "as a confirmation that distribution is occurring and the larger move lower is imminent."
            )
        else:
            return (
                f"With a NEUTRAL ML bias and an ADX reading of {adx:.0f}, London is expected to "
                "deliver a choppy, two-sided session without a clear directional commitment. The "
                "market may oscillate within a 40–60 pt range, frustrating both bulls and bears. "
                "Focus on identifying the developing range boundaries rather than fading or "
                "breakout trading. Institutional order flow during the 03:30–05:00 NY window will "
                "likely provide the first genuine directional clue of the day."
            )

    def ny_open_text() -> str:
        if b == "BULLISH" and rsi < 65:
            return (
                f"NY Open is the highest-probability window for the primary bull leg to materialize. "
                f"With RSI at {rsi:.0f} (room to run) and the ML model reading {int(max(0,rsi)/100*50+50)}% "
                f"bullish confidence, expect a momentum push through the London high within the first "
                f"60–90 minutes. The 09:30–10:30 NY window historically captures the majority of the "
                f"daily range on trending days. Breakout above the London session high with a "
                f"clean 5M or 15M close is the institutional confirmation entry. Target the upper "
                f"T1 zone first, then evaluate momentum for T2 extension."
            )
        elif b == "BEARISH" and rsi > 35:
            return (
                f"NY Open sets up as a high-risk, high-velocity environment on this bearish ML day. "
                f"RSI at {rsi:.0f} gives plenty of room to extend lower. The 09:30–10:00 NY gap "
                f"open or initial price sweep of London lows is the key trigger — a confirmed break "
                f"with volume expansion below the London low initiates the primary bear leg. "
                f"The {vr.lower()} volatility regime means moves may be fast and spiky; use "
                f"wider stops or wait for a 15M confirmation close below key support before entering "
                f"short. Lower T1 and T2 targets are the primary objectives."
            )
        elif phase == "Chop":
            return (
                "NY Open on a choppy regime day is treacherous. ADX is below threshold, indicating "
                "no genuine trend conviction from the market's perspective either. Expect the first "
                "30 minutes to establish a range that is then violated in both directions, triggering "
                "stops on both sides. Professional traders often sit out the 09:30–10:15 window on "
                "chop days and wait for a 10:15–11:00 NY setup where institutional intent becomes "
                "clearer. Scalping is the dominant strategy; avoid holding positions through news."
            )
        else:
            return (
                f"NY Open carries moderate directional conviction. With the current {phase} regime "
                f"and RSI at {rsi:.0f}, the open is likely to test either the London high or low "
                f"within the first 30 minutes. Wait for the first 30-minute range to establish "
                f"(approx. 09:30–10:00 NY), then trade the breakout in the ML bias direction "
                f"({'long above the range high' if b == 'BULLISH' else 'short below the range low' if b == 'BEARISH' else 'highest-momentum side'}). "
                f"Volume confirmation is essential — a breakout on declining volume is a trap."
            )

    def ny_pm_text() -> str:
        if vr == "Low":
            return (
                "NY afternoon session is expected to drift lazily in the direction of the morning "
                "bias, consistent with a low-volatility regime. Expect 15–30 pt moves max. "
                "Profit-taking by day-traders and algorithmic position-squaring typically creates "
                "gentle, orderly price action from 13:00 onward. The 15:00–15:30 NY window may see "
                "a brief uptick in activity as MOC (Market-on-Close) orders hit. Overall, PM session "
                "is not the time for new directional entries — protect and trail existing winners."
            )
        elif vr in ("High", "Extreme"):
            return (
                f"With a {vr.lower()} volatility regime still active, the NY afternoon carries real "
                "reversal risk. The 13:00–14:00 NY window has historically produced sharp counter-trend "
                "moves on volatile days as algorithmic profit-taking collides with fresh institutional "
                "positioning for the next session. Be alert for a potential 60–100 pt reversal from "
                "the morning's extreme. If you're long on a bullish day and the morning target was "
                "reached, consider reducing exposure before 14:00 NY. Trailing stops are your "
                "best friend during PM session in elevated vol environments."
            )
        else:
            return (
                f"NY PM session sets up with moderate reversal risk given the {vr.lower()} vol "
                f"environment. The primary driver will be whether the morning's ML-driven move "
                f"({b}) achieved its initial target levels. If T1 was reached, expect consolidation "
                "or mild profit-taking into the close. If the morning move was muted or stalled, "
                "there is a higher probability of a late-day continuation push into the last 90 "
                "minutes of trading. Watch the 3:30 PM NY futures close for clues about overnight "
                "positioning intentions — a strong close near the day's extreme favors continuation "
                "in the next Asia/London session."
            )

    return {
        "Asia":    asia_text(),
        "London":  london_text(),
        "NY Open": ny_open_text(),
        "NY PM":   ny_pm_text(),
    }


# ════════════════════════════════════════════════════════════════
#  NEWS SCRAPER (Forex Factory)
# ════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800)
def fetch_ff_news() -> List[Dict]:
    try:
        url = "https://www.forexfactory.com/calendar"
        headers = {"User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}")
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("tr.calendar__row")
        events = []
        last_time = "—"
        for row in rows:
            imp_tag = row.select_one("td.calendar__impact span")
            if not imp_tag:
                continue
            cls_str = imp_tag.get("class", [])
            if not any("high" in str(c).lower() for c in cls_str):
                continue
            cur_tag = row.select_one("td.calendar__currency")
            if not cur_tag or "USD" not in cur_tag.text:
                continue
            time_tag = row.select_one("td.calendar__time")
            if time_tag and time_tag.text.strip():
                last_time = time_tag.text.strip()
            name_tag = row.select_one("td.calendar__event span.calendar__event-title")
            if not name_tag:
                continue
            fc_tag = row.select_one("td.calendar__forecast")
            pr_tag = row.select_one("td.calendar__previous")
            events.append({
                "time":     last_time,
                "event":    name_tag.text.strip(),
                "forecast": fc_tag.text.strip() if fc_tag else "—",
                "previous": pr_tag.text.strip() if pr_tag else "—",
            })
        return events[:8]
    except Exception as e:
        logger.warning(f"FF news scrape failed: {e}")
        return [{"time": "N/A", "event": "Unable to load live data — check connection",
                 "forecast": "—", "previous": "—"}]


# ════════════════════════════════════════════════════════════════
#  LIGHTWEIGHT CHART BUILDER (TradingView-style)
# ════════════════════════════════════════════════════════════════
def build_lwc_chart(df: pd.DataFrame, pred: Prediction,
                    tf: str, show_projection: bool) -> str:
    """
    Returns an HTML string embedding a Lightweight Charts v5 candlestick chart
    with optional future projection cone.
    """
    if df is None or df.empty:
        return "<div style='color:#9CA3AF;padding:40px;text-align:center;'>Chart data unavailable.</div>"

    df = df.dropna(subset=["open","high","low","close"]).copy()
    # Limit display bars for performance
    max_bars = {"1M": 200, "5M": 200, "15M": 150, "30M": 120, "1H": 120, "4H": 100, "D": 200}
    df = df.tail(max_bars.get(tf, 150))

    cp = pred.current_price
    atr = pred.atr
    is_bull = pred.bias == "BULLISH"
    cfg = TF_CONFIG[tf]
    proj_bars = cfg["proj_bars"]

    # ── Convert candles to JS array ───────────────────────────
    candles = []
    for idx, row in df.iterrows():
        # Use UNIX timestamp (seconds) for intraday, date string for daily
        if tf == "D":
            t = idx.strftime("%Y-%m-%d")
        else:
            t = int(idx.timestamp())
        candles.append({
            "time":  t,
            "open":  round(float(row["open"]),  2),
            "high":  round(float(row["high"]),  2),
            "low":   round(float(row["low"]),   2),
            "close": round(float(row["close"]), 2),
        })

    # ── Projection lines ──────────────────────────────────────
    proj_center = []
    proj_upper  = []
    proj_lower  = []

    if show_projection and cp > 0 and len(df) > 0:
        last_ts = df.index[-1]
        # Determine bar frequency in seconds
        if tf == "1M":  bar_secs = 60
        elif tf == "5M":  bar_secs = 300
        elif tf == "15M": bar_secs = 900
        elif tf == "30M": bar_secs = 1800
        elif tf == "1H":  bar_secs = 3600
        elif tf == "4H":  bar_secs = 14400
        else:              bar_secs = 86400  # Daily

        drift_per_bar = (atr / 24) * (bar_secs / 3600) * (1 if is_bull else -1)
        cone_scale    = atr * 0.06  # tight cone

        for i in range(1, proj_bars + 1):
            if tf == "D":
                ts = (last_ts + timedelta(days=i)).strftime("%Y-%m-%d")
            else:
                ts = int((last_ts + timedelta(seconds=bar_secs * i)).timestamp())
            center_val = round(cp + drift_per_bar * i, 2)
            upper_val  = round(cp + cone_scale * i * 1.3, 2)
            lower_val  = round(cp - cone_scale * i * 1.3, 2)
            proj_center.append({"time": ts, "value": center_val})
            proj_upper.append({"time": ts, "value": upper_val})
            proj_lower.append({"time": ts, "value": lower_val})

    # ── Target level lines ────────────────────────────────────
    targets = []
    if pred.upper_t2_hi > 0:
        targets.append({"price": round((pred.upper_t2_lo + pred.upper_t2_hi) / 2, 2),
                         "color": "#00FFCC", "label": f"T2▲ {pred.upper_t2_lo:,.0f}–{pred.upper_t2_hi:,.0f}",
                         "style": 1})
    if pred.upper_t1_hi > 0:
        targets.append({"price": round((pred.upper_t1_lo + pred.upper_t1_hi) / 2, 2),
                         "color": "#00D4AA", "label": f"T1▲ {pred.upper_t1_lo:,.0f}–{pred.upper_t1_hi:,.0f}",
                         "style": 1})
    if cp > 0:
        targets.append({"price": round(cp, 2), "color": "#FFB800",
                         "label": f"NOW {cp:,.2f}", "style": 0})
    if pred.lower_t1_hi > 0:
        targets.append({"price": round((pred.lower_t1_lo + pred.lower_t1_hi) / 2, 2),
                         "color": "#FF6B6B", "label": f"T1▼ {pred.lower_t1_lo:,.0f}–{pred.lower_t1_hi:,.0f}",
                         "style": 1})
    if pred.lower_t2_hi > 0:
        targets.append({"price": round((pred.lower_t2_lo + pred.lower_t2_hi) / 2, 2),
                         "color": "#FF4B4B", "label": f"T2▼ {pred.lower_t2_lo:,.0f}–{pred.lower_t2_hi:,.0f}",
                         "style": 1})

    candles_json   = json.dumps(candles)
    proj_c_json    = json.dumps(proj_center)
    proj_u_json    = json.dumps(proj_upper)
    proj_l_json    = json.dumps(proj_lower)
    targets_json   = json.dumps(targets)
    proj_enabled   = "true" if show_projection else "false"
    bias_clr       = "#00D4AA" if is_bull else "#FF4B4B"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0A0A10; overflow: hidden; }}
  #chart {{ width: 100%; height: 440px; }}
</style>
</head>
<body>
<div id="chart"></div>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
  const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
    width:  document.getElementById('chart').clientWidth,
    height: 440,
    layout: {{
      background: {{ color: '#0A0A10' }},
      textColor:  '#9CA3AF',
      fontFamily: "'JetBrains Mono', monospace",
      fontSize:   11,
    }},
    grid: {{
      vertLines:   {{ color: '#1F2937', style: 1 }},
      horzLines:   {{ color: '#1F2937', style: 1 }},
    }},
    crosshair: {{
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: {{ color: '#4B5563', width: 1, style: 1, labelBackgroundColor: '#1F2937' }},
      horzLine: {{ color: '#4B5563', width: 1, style: 1, labelBackgroundColor: '#1F2937' }},
    }},
    rightPriceScale: {{
      borderColor: '#1F2937',
      scaleMargins: {{ top: 0.08, bottom: 0.08 }},
    }},
    timeScale: {{
      borderColor: '#1F2937',
      timeVisible: true,
      secondsVisible: false,
    }},
    handleScroll: {{ mouseWheel: true, pressedMouseMove: true }},
    handleScale:  {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true }},
  }});

  // Candle series
  const candleSeries = chart.addCandlestickSeries({{
    upColor:          '#00D4AA',
    downColor:        '#FF4B4B',
    borderUpColor:    '#00D4AA',
    borderDownColor:  '#FF4B4B',
    wickUpColor:      '#00D4AA',
    wickDownColor:    '#FF4B4B',
  }});
  candleSeries.setData({candles_json});

  // Target price lines
  const targetDefs = {targets_json};
  targetDefs.forEach(function(t) {{
    candleSeries.createPriceLine({{
      price:           t.price,
      color:           t.color,
      lineWidth:       1,
      lineStyle:       t.style,
      axisLabelVisible: true,
      title:           t.label,
    }});
  }});

  // Projection (volatility cone)
  const projEnabled = {proj_enabled};
  if (projEnabled) {{
    const projCenter = {proj_c_json};
    const projUpper  = {proj_u_json};
    const projLower  = {proj_l_json};

    if (projCenter.length > 0) {{
      // Upper cone boundary
      const upperSeries = chart.addLineSeries({{
        color: 'rgba(0,0,0,0)',
        lineWidth: 1,
        lineStyle: 2,
        lastValueVisible: false,
        priceLineVisible: false,
      }});
      upperSeries.setData(projUpper);

      // Lower cone boundary
      const lowerSeries = chart.addLineSeries({{
        color: 'rgba(0,0,0,0)',
        lineWidth: 1,
        lineStyle: 2,
        lastValueVisible: false,
        priceLineVisible: false,
      }});
      lowerSeries.setData(projLower);

      // Center drift line
      const centerSeries = chart.addLineSeries({{
        color:            '{bias_clr}',
        lineWidth:        2,
        lineStyle:        LightweightCharts.LineStyle.Dashed,
        lastValueVisible: true,
        priceLineVisible: false,
        title:            'Projection',
      }});
      centerSeries.setData(projCenter);
    }}
  }}

  // Responsive resize
  window.addEventListener('resize', function() {{
    chart.applyOptions({{ width: document.getElementById('chart').clientWidth }});
  }});

  // Fit content
  chart.timeScale().fitContent();
}})();
</script>
</body>
</html>
"""
    return html


# ════════════════════════════════════════════════════════════════
#  PREDICTION ENGINE (orchestrator)
# ════════════════════════════════════════════════════════════════
class PredictionEngine:

    def __init__(self):
        self.dp  = DataPipeline()
        self.fe  = FeatureEngine()
        self.rd  = RegimeDetector()
        self.ml  = MLEnsemble()
        self.ready = False

    def _cache_key(self, df_daily: pd.DataFrame) -> str:
        return hashlib.md5(str(df_daily.index[-1]).encode()).hexdigest()[:8]

    def initialize(self) -> bool:
        try:
            df_daily = self.dp.fetch(SYMBOLS["NQ"], "2y", "1d")
            if df_daily is None or len(df_daily) < 200:
                return False
            corr = self.dp.fetch_correlated()
            feat = self.fe.build(df_daily, corr)
            if len(feat) < 100:
                return False
            dk = self._cache_key(df_daily)
            if MODEL_CACHE.exists():
                try:
                    with open(MODEL_CACHE, "rb") as f:
                        cache = pickle.load(f)
                    if cache.get("hash") == dk:
                        self.ml = cache["model"]
                        self.ready = True
                        return True
                except Exception:
                    pass
            self.ml.train(feat)
            try:
                with open(MODEL_CACHE, "wb") as f:
                    pickle.dump({"hash": dk, "model": self.ml}, f)
            except Exception:
                pass
            self.ready = True
            return True
        except Exception as e:
            logger.error(f"Init error: {e}")
            return False

    def predict(self) -> Optional[Prediction]:
        try:
            df_d = self.dp.fetch(SYMBOLS["NQ"], "2y", "1d")
            if df_d is None:
                return None

            corr = self.dp.fetch_correlated()
            vix_df = corr.get("VIX")

            feat = self.fe.build(df_d, corr)
            if feat.empty:
                return None

            latest_feat = feat.drop(columns=["label"], errors="ignore")
            latest_feat = latest_feat.select_dtypes(include=[np.number])
            last_row = latest_feat.iloc[-1].to_dict()

            prob_bull, conf = self.ml.predict(last_row)
            prob_bear = 1 - prob_bull

            cp  = float(df_d["close"].iloc[-1])
            atr = float(FeatureEngine.atr(df_d, 14).iloc[-1])
            rsi_val = float(FeatureEngine.rsi(df_d["close"], 14).iloc[-1])

            regime = self.rd.detect(df_d, vix_df)

            bias = ("BULLISH" if prob_bull > 0.545
                    else "BEARISH" if prob_bear > 0.545
                    else "NEUTRAL")

            # ── Target ranges (narrow, ATR-based) ────────────
            # Range half-width = 0.10 * ATR (tight, realistic)
            half_w = atr * 0.10
            atr_m1 = 0.85
            atr_m2 = 1.60

            def target_range(center):
                lo = round((center - half_w) * 4) / 4
                hi = round((center + half_w) * 4) / 4
                return lo, hi

            def target_conf(center, ml_conf, cp, max_dist):
                """Higher confidence for closer targets."""
                dist = abs(center - cp)
                proximity_bonus = max(0.0, 1.0 - dist / (max_dist + 1e-9))
                base = ml_conf * 0.55 + 0.30
                raw = base + proximity_bonus * 0.25
                return min(0.97, max(0.38, raw))

            max_dist = atr * atr_m2

            u1_c = cp + atr * atr_m1
            u2_c = cp + atr * atr_m2
            l1_c = cp - atr * atr_m1
            l2_c = cp - atr * atr_m2

            u1_lo, u1_hi = target_range(u1_c)
            u2_lo, u2_hi = target_range(u2_c)
            l1_lo, l1_hi = target_range(l1_c)
            l2_lo, l2_hi = target_range(l2_c)

            session_out = build_session_outlook(bias, regime, rsi_val)
            hi_time, lo_time = TimeEstimator.estimate(
                bias, regime["phase"], regime["vol_regime"])

            momentum = ("POSITIVE" if rsi_val > 55
                        else "NEGATIVE" if rsi_val < 45
                        else "NEUTRAL")

            prev_close = float(df_d["close"].iloc[-2]) if len(df_d) > 1 else cp
            delta_pct  = (cp - prev_close) / (prev_close + 1e-9) * 100

            return Prediction(
                bias=bias,
                prob_bull=prob_bull,
                prob_bear=prob_bear,
                confidence=conf,
                current_price=cp,
                atr=atr,
                upper_t1_lo=u1_lo, upper_t1_hi=u1_hi,
                upper_t2_lo=u2_lo, upper_t2_hi=u2_hi,
                lower_t1_lo=l1_lo, lower_t1_hi=l1_hi,
                lower_t2_lo=l2_lo, lower_t2_hi=l2_hi,
                upper_t1_conf=target_conf(u1_c, conf, cp, max_dist),
                upper_t2_conf=target_conf(u2_c, conf, cp, max_dist),
                lower_t1_conf=target_conf(l1_c, conf, cp, max_dist),
                lower_t2_conf=target_conf(l2_c, conf, cp, max_dist),
                regime=regime["label"],
                regime_phase=regime["phase"],
                vix=regime["vix"],
                vol_regime=regime["vol_regime"],
                trend_prob=regime["trend_prob"],
                session_outlook=session_out,
                high_time_est=hi_time,
                low_time_est=lo_time,
                wf_accuracy=self.ml.wf_metrics.get("accuracy", 0.0),
                wf_auc=self.ml.wf_metrics.get("auc", 0.0),
                delta_pct=delta_pct,
                rsi=rsi_val,
                momentum=momentum,
            )
        except Exception as e:
            logger.error(f"Predict error: {traceback.format_exc()}")
            return None


# ════════════════════════════════════════════════════════════════
#  CSS & STREAMLIT UI — Deep OLED Dark
# ════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0A0A10 !important;
    color: #F3F4F6 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] { display: none !important; }

[data-testid="stSidebar"] {
    background: #0D0D18 !important;
    border-right: 1px solid #1F2937 !important;
}
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

[data-testid="stMainBlockContainer"] {
    max-width: 1600px !important;
    padding: 0 1.5rem 2rem 1.5rem !important;
}

hr { border-color: #1F2937 !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0A0A10; }
::-webkit-scrollbar-thumb { background: #2A2A3E; border-radius: 3px; }

[data-testid="stMetric"] {
    background: #12121A !important;
    border: 1px solid #1F2937 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { color: #9CA3AF !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #F3F4F6 !important; font-family: 'JetBrains Mono' !important; }

/* Radio buttons (timeframe selector) */
[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    flex-wrap: wrap !important;
}
[data-testid="stRadio"] label {
    background: #12121A !important;
    border: 1px solid #2A2A3E !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    color: #9CA3AF !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(0,212,170,0.12) !important;
    border-color: #00D4AA !important;
    color: #00D4AA !important;
}
[data-testid="stRadio"] input[type="radio"] { display: none !important; }

/* Buttons */
button[kind="primary"] {
    background: #00D4AA !important;
    border: none !important;
    border-radius: 8px !important;
    color: #000 !important;
    font-weight: 700 !important;
}
button[kind="secondary"] {
    background: #12121A !important;
    border: 1px solid #1F2937 !important;
    border-radius: 8px !important;
    color: #9CA3AF !important;
}

/* iframes (LWC chart) */
iframe { border: none !important; background: transparent !important; }

/* Toggle */
[data-testid="stToggle"] span { color: #9CA3AF !important; font-size: 13px !important; }
</style>
"""


def _header_html(now_ny: datetime) -> str:
    ts_str   = now_ny.strftime("%B %d, %Y  |  NQZ5 (NAS-100 FUTURES)")
    time_str = now_ny.strftime("%H:%M NY")
    return f"""
<div style="padding:20px 4px 18px 4px;border-bottom:1px solid #1F2937;
            margin-bottom:20px;display:flex;justify-content:space-between;
            align-items:flex-end;">
  <div>
    <div style="font-family:'Inter',sans-serif;font-weight:900;font-size:22px;
                color:#F3F4F6;letter-spacing:-0.5px;">MNQ/NQ FUTURES
      <span style="color:#00D4AA;"> | </span>DAILY ML DASHBOARD</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                color:#9CA3AF;margin-top:4px;letter-spacing:1px;">{ts_str}</div>
  </div>
  <div style="text-align:right;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:20px;
                font-weight:700;color:#FFB800;">{time_str}</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                color:#4B5563;letter-spacing:2px;">LIVE DATA · AUTO REFRESH</div>
  </div>
</div>"""


def _bias_panel(pred: Prediction) -> str:
    is_bull = pred.bias == "BULLISH"
    is_bear = pred.bias == "BEARISH"

    if is_bull:
        glow = "rgba(0,212,170,0.18)"; border = "#00D4AA"; color = "#00D4AA"
        arrow = "▲"; shadow = "0 0 32px rgba(0,212,170,0.25)"
    elif is_bear:
        glow = "rgba(255,75,75,0.15)"; border = "#FF4B4B"; color = "#FF4B4B"
        arrow = "▼"; shadow = "0 0 32px rgba(255,75,75,0.20)"
    else:
        glow = "rgba(255,184,0,0.12)"; border = "#FFB800"; color = "#FFB800"
        arrow = "◆"; shadow = "0 0 24px rgba(255,184,0,0.15)"

    conf_pct = int(pred.confidence * 100)
    prob_pct = int(max(pred.prob_bull, pred.prob_bear) * 100)

    return f"""
<div style="background:linear-gradient(135deg,{glow},rgba(10,10,16,0.95));
            border:1.5px solid {border}; border-radius:16px;
            padding:24px 28px; box-shadow:{shadow}; position:relative; overflow:hidden;">
  <div style="position:absolute;top:-20px;right:-20px;width:120px;height:120px;
              border-radius:50%;background:{glow};filter:blur(30px);pointer-events:none;"></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:10px;">⬡ DAILY ML BIAS &amp; TREND</div>
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
    <div style="font-size:52px;line-height:1;color:{color};
                filter:drop-shadow(0 0 12px {color});">{arrow}</div>
    <div>
      <div style="font-family:'Inter',sans-serif;font-weight:900;font-size:36px;
                  color:{color};letter-spacing:-1px;line-height:1;">{pred.bias}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:#9CA3AF;margin-top:3px;">CURRENT BIAS</div>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#9CA3AF;">
        ML MODEL CONFIDENCE</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:13px;
                   font-weight:700;color:{color};">{conf_pct}%</span>
    </div>
    <div style="height:4px;background:rgba(255,255,255,0.07);border-radius:2px;">
      <div style="width:{conf_pct}%;height:4px;background:{color};
                  border-radius:2px;box-shadow:0 0 8px {color};"></div></div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#9CA3AF;">
        SIGNAL PROBABILITY</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:13px;
                   font-weight:700;color:{color};">{prob_pct}%</span>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:2px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#9CA3AF;">
        TREND DIRECTION</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;
                   color:{'#00D4AA' if pred.trend_prob > 0.55 else '#FFB800'};">
        {'UPWARD' if is_bull else 'DOWNWARD' if is_bear else 'SIDEWAYS'}</span>
    </div>
  </div>
</div>"""


def _targets_panel(pred: Prediction) -> str:
    cp = pred.current_price
    d  = pred.delta_pct
    sign = "+" if d >= 0 else ""
    dc   = "#00D4AA" if d >= 0 else "#FF4B4B"

    def conf_badge(c: float, color: str) -> str:
        pct = int(c * 100)
        return (f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                f'font-weight:700;color:{color};background:rgba(255,255,255,0.05);'
                f'border:1px solid {color}44;border-radius:4px;padding:2px 6px;">'
                f'{pct}%</span>')

    def row(label, lo, hi, color, conf=None):
        conf_html = conf_badge(conf, color) if conf is not None else ""
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:9px 14px;border-radius:8px;background:rgba(255,255,255,0.03);'
            f'border:1px solid {color}22;margin-bottom:4px;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:#9CA3AF;letter-spacing:1px;">{label}</span>'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'{conf_html}'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:13px;'
            f'font-weight:700;color:{color};">{lo:,.0f} – {hi:,.0f}</span>'
            f'</div></div>'
        )

    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;
            padding:20px 22px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:14px;">◈ DAILY TARGETS</div>
  {row("UPPER TARGET 2", pred.upper_t2_lo, pred.upper_t2_hi, "#00FFCC", pred.upper_t2_conf)}
  {row("UPPER TARGET 1", pred.upper_t1_lo, pred.upper_t1_hi, "#00D4AA", pred.upper_t1_conf)}
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding:9px 14px;border-radius:8px;background:rgba(255,184,0,0.07);
              border:1.5px solid rgba(255,184,0,0.3);margin-bottom:4px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                 color:#F3F4F6;letter-spacing:1px;">CURRENT PRICE (NQ)</span>
    <span style="font-family:'JetBrains Mono',monospace;">
      <span style="font-size:16px;font-weight:800;color:#FFB800;">{cp:,.2f}</span>
      <span style="font-size:11px;font-weight:600;color:{dc};margin-left:6px;">
        ({sign}{d:.2f}%)</span>
    </span>
  </div>
  {row("LOWER TARGET 1", pred.lower_t1_lo, pred.lower_t1_hi, "#FF6B6B", pred.lower_t1_conf)}
  {row("LOWER TARGET 2", pred.lower_t2_lo, pred.lower_t2_hi, "#FF4B4B", pred.lower_t2_conf)}
</div>"""


def _validation_panel(pred: Prediction) -> str:
    vix = pred.vix
    rsi = pred.rsi
    vol_c = ("#FF4B4B" if pred.vol_regime in ("High","Extreme") else "#00D4AA")
    mom_c = ("#00D4AA" if pred.momentum == "POSITIVE"
              else "#FF4B4B" if pred.momentum == "NEGATIVE" else "#FFB800")
    acc_pct = int(pred.wf_accuracy * 100)
    auc_pct = int(pred.wf_auc * 100)

    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;
            padding:20px 22px;height:100%;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:14px;">◉ MODEL VALIDATION</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">VOLATILITY</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                  font-weight:700;color:{vol_c};">{pred.vol_regime.upper()}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:#9CA3AF;margin-top:2px;">ATR: {pred.atr:.0f} pts</div>
    </div>
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">MOMENTUM</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                  font-weight:700;color:{mom_c};">{pred.momentum}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:#9CA3AF;margin-top:2px;">RSI: {rsi:.0f}</div>
    </div>
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">VIX LEVEL</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                  font-weight:700;color:{'#FF4B4B' if vix > 30 else '#FFB800' if vix > 20 else '#00D4AA'};">
        {vix:.1f}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:#9CA3AF;margin-top:2px;">{pred.regime}</div>
    </div>
    <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;margin-bottom:5px;">WF ACCURACY</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
                  font-weight:700;color:#00D4AA;">{acc_pct}%</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:#9CA3AF;margin-top:2px;">AUC: {auc_pct}%</div>
    </div>
  </div>
  <div style="background:#12121A;border-radius:10px;padding:12px 14px;">
    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9CA3AF;">
        LAST BIAS CALL</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                   color:#00D4AA;font-weight:700;">{pred.bias} — ACTIVE</span>
    </div>
    <div style="display:flex;justify-content:space-between;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9CA3AF;">
        SIGNAL STATUS</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#00D4AA;">
        ⬤ LIVE</span>
    </div>
  </div>
</div>"""


def _regime_panel(pred: Prediction) -> str:
    phase_colors = {
        "Trend":         "#00D4AA",
        "Accumulation":  "#00AAFF",
        "Distribution":  "#FF9500",
        "Chop":          "#FFB800",
        "Consolidation": "#6B7280",
        "Manipulation":  "#FF4B4B",
    }
    clr    = phase_colors.get(pred.regime_phase, "#9CA3AF")
    phases = ["Accumulation", "Manipulation", "Trend",
              "Distribution", "Chop", "Consolidation"]
    dots = ""
    for p in phases:
        active = (p == pred.regime_phase)
        pc = phase_colors.get(p, "#4B5563")
        bg = pc if active else "#1F2937"
        dots += (
            f'<div style="background:{bg};border-radius:6px;padding:5px 10px;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:9px;'
            f'color:{"#000" if active else "#9CA3AF"};font-weight:{"700" if active else "400"};'
            f'white-space:nowrap;">{p}</div>'
        )

    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;
            padding:20px 22px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:14px;">▣ MARKET REGIME &amp; PHASE</div>
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
    <div style="font-family:'Inter',sans-serif;font-weight:800;font-size:22px;
                color:{clr};line-height:1;">{pred.regime_phase.upper()}</div>
    <div style="background:rgba(255,255,255,0.04);border:1px solid #2A2A3E;
                border-radius:8px;padding:4px 12px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                   color:#9CA3AF;">{pred.regime}</span>
    </div>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:6px;">{dots}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px;">
    <div style="background:#12121A;border-radius:8px;padding:10px 12px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;">TREND PROB</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                  font-weight:700;color:#00D4AA;">{int(pred.trend_prob*100)}%</div>
    </div>
    <div style="background:#12121A;border-radius:8px;padding:10px 12px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;letter-spacing:2px;">VOL REGIME</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                  font-weight:700;color:{'#FF4B4B' if pred.vol_regime in ('High','Extreme') else '#00D4AA'};">
        {pred.vol_regime.upper()}</div>
    </div>
  </div>
</div>"""


def _time_panel(pred: Prediction) -> str:
    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;
            padding:20px 22px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:14px;">◷ TIME PROJECTIONS</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
    <div style="background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.20);
                border-radius:10px;padding:14px 16px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#00D4AA;letter-spacing:2px;margin-bottom:6px;">HIGH EXPECTED</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                  font-weight:700;color:#00D4AA;">{pred.high_time_est}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;margin-top:4px;">Statistical estimate</div>
    </div>
    <div style="background:rgba(255,75,75,0.06);border:1px solid rgba(255,75,75,0.20);
                border-radius:10px;padding:14px 16px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#FF6B6B;letter-spacing:2px;margin-bottom:6px;">LOW EXPECTED</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                  font-weight:700;color:#FF6B6B;">{pred.low_time_est}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#9CA3AF;margin-top:4px;">Statistical estimate</div>
    </div>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
              color:#4B5563;margin-top:12px;text-align:center;">
    Based on historical intraday volatility distribution
  </div>
</div>"""


def _session_panel(pred: Prediction) -> str:
    icons = {"Asia": "🌏", "London": "🇬🇧", "NY Open": "🗽", "NY PM": "🌆"}
    h = datetime.utcnow().hour
    current = ("Asia" if 0 <= h < 7
                else "London" if 7 <= h < 12
                else "NY Open" if 12 <= h < 17
                else "NY PM")

    rows = ""
    for sess, outlook in pred.session_outlook.items():
        active = (sess == current)
        border = "rgba(0,212,170,0.35)" if active else "#1F2937"
        bg     = "rgba(0,212,170,0.06)" if active else "#12121A"
        icon   = icons.get(sess, "◈")
        rows += f"""
<div style="background:{bg};border:1px solid {border};border-radius:10px;
            padding:14px 16px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <span style="font-size:15px;">{icon}</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                 color:{'#00D4AA' if active else '#9CA3AF'};
                 font-weight:{'700' if active else '500'};
                 letter-spacing:2px;">{sess.upper()}{"  ← NOW" if active else ""}</span>
  </div>
  <div style="font-family:'Inter',sans-serif;font-size:12px;color:#D1D5DB;
              line-height:1.65;padding-left:23px;">{outlook}</div>
</div>"""

    return f"""
<div style="background:#0D0D18;border:1px solid #1F2937;border-radius:16px;
            padding:20px 22px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:14px;">◎ SESSION OUTLOOK — AI ANALYSIS</div>
  {rows}
</div>"""


def _news_panel(events: List[Dict]) -> str:
    if not events:
        return ""
    rows = ""
    for e in events:
        rows += f"""
<div style="display:flex;align-items:flex-start;gap:12px;
            padding:10px 14px;border-bottom:1px solid #0E0E1A;
            background:rgba(255,75,75,0.03);">
  <div style="min-width:56px;font-family:'JetBrains Mono',monospace;
              font-size:10px;color:#FF6B6B;padding-top:1px;">{e['time']}</div>
  <div style="flex:1;">
    <div style="font-family:'Inter',sans-serif;font-size:12px;
                font-weight:600;color:#F3F4F6;margin-bottom:2px;">{e['event']}</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#9CA3AF;">
      Forecast: <span style="color:#D1D5DB;">{e['forecast']}</span>
      &nbsp;&nbsp;Prev: <span style="color:#9CA3AF;">{e['previous']}</span>
    </div>
  </div>
  <div style="width:8px;height:8px;border-radius:50%;background:#FF4B4B;
              box-shadow:0 0 6px rgba(255,75,75,0.7);margin-top:3px;flex-shrink:0;"></div>
</div>"""

    return f"""
<div style="background:#0D0D18;border:1px solid #1E0A0A;border-radius:16px;overflow:hidden;">
  <div style="padding:16px 20px 12px 20px;border-bottom:1px solid #0E0E1A;
              display:flex;align-items:center;gap:10px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                 color:#FF6B6B;letter-spacing:3px;text-transform:uppercase;">
      ⚡ HIGH IMPACT USD NEWS</span>
    <span style="background:#FF4B4B;color:#000;font-family:'JetBrains Mono',monospace;
                 font-size:8px;font-weight:700;padding:2px 6px;border-radius:4px;">
      FOREX FACTORY</span>
  </div>
  {rows}
</div>"""


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:10px;
            color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;
            margin-bottom:20px;">⬡ NQ PREDICTOR v4</div>
""", unsafe_allow_html=True)

    show_proj = st.sidebar.toggle("Show Future Projection", value=True)
    auto_refresh = st.sidebar.toggle("Auto Refresh (5 min)", value=False)

    st.sidebar.markdown("<hr style='border-color:#1F2937;margin:18px 0;'>",
                        unsafe_allow_html=True)

    retrain = st.sidebar.button("🔄 Retrain Model", use_container_width=True)
    if retrain:
        try:
            MODEL_CACHE.unlink()
        except Exception:
            pass
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("<hr style='border-color:#1F2937;margin:18px 0;'>",
                        unsafe_allow_html=True)
    st.sidebar.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:9px;
            color:#4B5563;letter-spacing:2px;line-height:1.8;">
MODELS<br>
XGBoost · LightGBM<br>
RandomForest · ExtraTrees<br>
LogisticRegression (meta)<br><br>
VALIDATION<br>
Walk-Forward · TimeSeriesSplit<br>
6-fold · No lookahead<br><br>
DATA<br>
NQ Futures · VIX · DXY<br>
US10Y · QQQ · ES Futures
</div>
""", unsafe_allow_html=True)

    st.sidebar.markdown("""
<div style="position:absolute;bottom:20px;left:0;right:0;padding:0 16px;
            font-family:'JetBrains Mono',monospace;font-size:8px;
            color:#2A2A3E;text-align:center;line-height:1.6;">
PROBABILISTIC MODEL ONLY<br>
NOT FINANCIAL ADVICE<br>
FOR RESEARCH PURPOSES ONLY
</div>
""", unsafe_allow_html=True)

    return show_proj, auto_refresh


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="MNQ/NQ ML Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    show_proj, auto_refresh = render_sidebar()

    if auto_refresh:
        st.markdown('<meta http-equiv="refresh" content="300">', unsafe_allow_html=True)

    now_utc = datetime.utcnow()
    now_ny  = now_utc - timedelta(hours=5)

    st.markdown(_header_html(now_ny), unsafe_allow_html=True)

    # ── Initialize engine ──────────────────────────────────────
    if "engine" not in st.session_state:
        st.session_state.engine = PredictionEngine()

    engine: PredictionEngine = st.session_state.engine

    if not engine.ready:
        with st.spinner("🔧 Initializing ML Engine — fetching data & training models…"):
            ok = engine.initialize()
        if not ok:
            st.error("❌ Failed to initialize engine. Check internet connection.")
            return
        st.rerun()

    # ── Compute prediction ─────────────────────────────────────
    with st.spinner("📡 Computing today's prediction…"):
        pred = engine.predict()

    if pred is None:
        st.error("❌ Prediction failed. Try retraining via the sidebar.")
        return

    # ── Timeframe selector ─────────────────────────────────────
    st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9CA3AF;
            letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;">
  📈  NQ (NAS-100) CHART — SELECT TIMEFRAME
</div>""", unsafe_allow_html=True)

    tf_options = list(TF_CONFIG.keys())
    # Default to 1H
    default_idx = tf_options.index("1H")
    selected_tf = st.radio(
        label="Timeframe",
        options=tf_options,
        index=default_idx,
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Fetch chart data for selected TF ──────────────────────
    with st.spinner(f"Loading {selected_tf} chart data…"):
        df_chart = DataPipeline.fetch_for_timeframe(selected_tf)

    # ── Fetch news ─────────────────────────────────────────────
    news = fetch_ff_news()

    # ════════════════════════════════════════════════════════
    # LAYOUT: Row 1 — Bias  |  Chart
    # ════════════════════════════════════════════════════════
    left, right = st.columns([1, 1.85], gap="medium")

    with left:
        st.markdown(_bias_panel(pred), unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(_targets_panel(pred), unsafe_allow_html=True)

    with right:
        if df_chart is not None and not df_chart.empty:
            chart_html = build_lwc_chart(df_chart, pred, selected_tf, show_proj)
            st.components.v1.html(chart_html, height=460, scrolling=False)
        else:
            st.warning(f"Chart data unavailable for {selected_tf} timeframe.")

    # ════════════════════════════════════════════════════════
    # LAYOUT: Row 2 — Validation | Regime | Time
    # ════════════════════════════════════════════════════════
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(_validation_panel(pred), unsafe_allow_html=True)
    with c2:
        st.markdown(_regime_panel(pred), unsafe_allow_html=True)
    with c3:
        st.markdown(_time_panel(pred), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # LAYOUT: Row 3 — Sessions | News
    # ════════════════════════════════════════════════════════
    s1, s2 = st.columns([1.1, 0.9], gap="medium")
    with s1:
        st.markdown(_session_panel(pred), unsafe_allow_html=True)
    with s2:
        st.markdown(_news_panel(news), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # Footer
    # ════════════════════════════════════════════════════════
    st.markdown("""
<div style="text-align:center;padding:28px 0 8px 0;color:#2A2A3E;
            font-size:10px;border-top:1px solid #1F2937;margin-top:28px;
            font-family:'JetBrains Mono',monospace;letter-spacing:2px;">
  MNQ/NQ ML PREDICTOR · PROBABILISTIC ONLY · NOT FINANCIAL ADVICE · FOR RESEARCH
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
