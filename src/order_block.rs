// =============================================================================
// ORDER BLOCK DETECTION & TP/SL CALCULATION
// =============================================================================
//
// Smart Money Concepts Order Block tabanlı TP/SL sistemi.
//
// Order Block Tanımı:
// - Bullish OB: Güçlü yukarı BOS öncesindeki son bearish mum (demand zone)
// - Bearish OB: Güçlü aşağı BOS öncesindeki son bullish mum (supply zone)
//
// TP/SL Mantığı:
// - LONG: SL = Bullish OB low altı (buffer ile), TP = En yakın unmitigated bearish OB
// - SHORT: SL = Bearish OB high üstü (buffer ile), TP = En yakın unmitigated bullish OB
//

use crate::types::{Candle, SignalType};
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// =============================================================================
// ORDER BLOCK STRUCT
// =============================================================================

/// Bir Order Block'u temsil eder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBlock {
    /// Bullish (demand) veya Bearish (supply)
    pub ob_type: OrderBlockType,
    /// OB mumunun high'ı
    pub high: Decimal,
    /// OB mumunun low'u
    pub low: Decimal,
    /// OB mumunun open'ı
    pub open: Decimal,
    /// OB mumunun close'u
    pub close: Decimal,
    /// OB'nin oluştuğu mum indeksi
    pub candle_idx: usize,
    /// OB hala geçerli mi (fiyat tarafından mitigation edilmemiş)
    pub is_valid: bool,
    /// OB zaten test edildi mi (fiyat OB zone'a dokundu)
    pub is_tested: bool,
    /// BOS candle range (gücü ölçmek için)
    pub bos_strength: Decimal,
    /// OB zone'un orta noktası (entry için)
    pub midpoint: Decimal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderBlockType {
    /// Demand zone - güçlü yukarı hareket öncesi son bearish mum
    Bullish,
    /// Supply zone - güçlü aşağı hareket öncesi son bullish mum
    Bearish,
}

impl OrderBlock {
    pub fn new_bullish(candle: &Candle, candle_idx: usize, bos_strength: Decimal) -> Self {
        let midpoint = (candle.high + candle.low) / Decimal::from(2);
        Self {
            ob_type: OrderBlockType::Bullish,
            high: candle.high,
            low: candle.low,
            open: candle.open,
            close: candle.close,
            candle_idx,
            is_valid: true,
            is_tested: false,
            bos_strength,
            midpoint,
        }
    }

    pub fn new_bearish(candle: &Candle, candle_idx: usize, bos_strength: Decimal) -> Self {
        let midpoint = (candle.high + candle.low) / Decimal::from(2);
        Self {
            ob_type: OrderBlockType::Bearish,
            high: candle.high,
            low: candle.low,
            open: candle.open,
            close: candle.close,
            candle_idx,
            is_valid: true,
            is_tested: false,
            bos_strength,
            midpoint,
        }
    }

    /// OB zone genişliği
    pub fn zone_width(&self) -> Decimal {
        self.high - self.low
    }

    /// Fiyat OB zone içinde mi?
    pub fn contains_price(&self, price: Decimal) -> bool {
        price >= self.low && price <= self.high
    }

    /// Bu OB'yi mitigation et (invalidate)
    pub fn mitigate(&mut self) {
        self.is_valid = false;
    }

    /// Bu OB'yi test edilmiş olarak işaretle
    pub fn mark_tested(&mut self) {
        self.is_tested = true;
    }
}

// =============================================================================
// ORDER BLOCK TRACKER
// =============================================================================

/// Order Block'ları tespit edip takip eden yapı
pub struct OrderBlockTracker {
    /// Aktif bullish order block'lar (demand zones)
    pub bullish_obs: VecDeque<OrderBlock>,
    /// Aktif bearish order block'lar (supply zones)
    pub bearish_obs: VecDeque<OrderBlock>,
    /// Son birkaç mum (OB tespiti için lookback)
    candle_history: VecDeque<Candle>,
    /// Candle index history (for mapping)
    candle_idx_history: VecDeque<usize>,
    /// Minimum BOS displacement (ATR çarpanı)
    pub min_displacement_atr: Decimal,
    /// Maksimum takip edilen OB sayısı
    pub max_obs: usize,
    /// OB yaşı limiti (kaç mum sonra expire)
    pub max_ob_age: usize,
}

impl OrderBlockTracker {
    pub fn new() -> Self {
        Self {
            bullish_obs: VecDeque::new(),
            bearish_obs: VecDeque::new(),
            candle_history: VecDeque::new(),
            candle_idx_history: VecDeque::new(),
            min_displacement_atr: Decimal::from_f64(1.0).unwrap(), // BOS mumu en az 1x ATR olmalı
            max_obs: 10,                                           // En fazla 10 OB takip et
            max_ob_age: 200,                                       // 200 mumdan eski OB'leri sil
        }
    }

    /// Yeni mum geldiğinde OB tespiti ve güncelleme
    pub fn update(
        &mut self,
        candle: &Candle,
        candle_idx: usize,
        atr: Option<Decimal>,
        bos_up: bool,   // Bu mumda yukarı BOS oldu mu?
        bos_down: bool, // Bu mumda aşağı BOS oldu mu?
    ) {
        // Mum tarihçesini güncelle
        self.candle_history.push_back(candle.clone());
        self.candle_idx_history.push_back(candle_idx);
        if self.candle_history.len() > 20 {
            self.candle_history.pop_front();
            self.candle_idx_history.pop_front();
        }

        let current_atr = atr.unwrap_or(Decimal::ONE);
        let candle_range = candle.high - candle.low;

        // ─────────────────────────────────────────────────────────
        // BULLISH ORDER BLOCK TESPİTİ
        // Koşul: Yukarı BOS + displacement mumu > ATR threshold
        // OB = BOS öncesindeki son bearish mum
        // ─────────────────────────────────────────────────────────
        if bos_up && candle_range >= current_atr * self.min_displacement_atr {
            // Son bearish mumu geriye doğru ara (max 5 mum geriye bak)
            if let Some(ob) = self.find_last_bearish_candle(candle_idx, &current_atr) {
                self.bullish_obs.push_back(ob);
                // Max OB sayısını aşma
                while self.bullish_obs.len() > self.max_obs {
                    self.bullish_obs.pop_front();
                }
            }
        }

        // ─────────────────────────────────────────────────────────
        // BEARISH ORDER BLOCK TESPİTİ
        // Koşul: Aşağı BOS + displacement mumu > ATR threshold
        // OB = BOS öncesindeki son bullish mum
        // ─────────────────────────────────────────────────────────
        if bos_down && candle_range >= current_atr * self.min_displacement_atr {
            // Son bullish mumu geriye doğru ara
            if let Some(ob) = self.find_last_bullish_candle(candle_idx, &current_atr) {
                self.bearish_obs.push_back(ob);
                while self.bearish_obs.len() > self.max_obs {
                    self.bearish_obs.pop_front();
                }
            }
        }

        // ─────────────────────────────────────────────────────────
        // MİTİGATİON KONTROLÜ
        // Fiyat OB zone'a girerse → mitigate et
        // ─────────────────────────────────────────────────────────
        self.check_mitigation(candle, candle_idx);
    }

    /// Geriye doğru son bearish mumu bul (Bullish OB için)
    fn find_last_bearish_candle(&self, _current_idx: usize, _atr: &Decimal) -> Option<OrderBlock> {
        let history_len = self.candle_history.len();
        if history_len < 2 {
            return None;
        }

        // Son 5 mumu geriye doğru kontrol et (BOS mumu hariç, ondan önceki mumlar)
        let lookback = 5.min(history_len - 1);
        for i in (0..lookback).rev() {
            let idx = history_len - 2 - i; // -2: BOS mumu kendisi -1, ondan öncesi -2
            if idx < history_len {
                let hist_candle = &self.candle_history[idx];
                let hist_idx = self.candle_idx_history[idx];

                // Bearish mum mu? (close < open)
                if hist_candle.is_bearish() {
                    let bos_candle = self.candle_history.back().unwrap();
                    let bos_strength = bos_candle.high - bos_candle.low;
                    return Some(OrderBlock::new_bullish(hist_candle, hist_idx, bos_strength));
                }
            }
        }

        None
    }

    /// Geriye doğru son bullish mumu bul (Bearish OB için)
    fn find_last_bullish_candle(&self, _current_idx: usize, _atr: &Decimal) -> Option<OrderBlock> {
        let history_len = self.candle_history.len();
        if history_len < 2 {
            return None;
        }

        let lookback = 5.min(history_len - 1);
        for i in (0..lookback).rev() {
            let idx = history_len - 2 - i;
            if idx < history_len {
                let hist_candle = &self.candle_history[idx];
                let hist_idx = self.candle_idx_history[idx];

                // Bullish mum mu? (close > open)
                if hist_candle.is_bullish() {
                    let bos_candle = self.candle_history.back().unwrap();
                    let bos_strength = bos_candle.high - bos_candle.low;
                    return Some(OrderBlock::new_bearish(hist_candle, hist_idx, bos_strength));
                }
            }
        }

        None
    }

    /// Mitigation kontrolü: Fiyat OB zone'u geçtiyse invalidate et
    fn check_mitigation(&mut self, candle: &Candle, current_idx: usize) {
        // Bullish OB'ler: Fiyat OB low'un altına düşerse → mitigate
        for ob in self.bullish_obs.iter_mut() {
            if !ob.is_valid {
                continue;
            }

            // Yaş kontrolü
            if current_idx.saturating_sub(ob.candle_idx) > self.max_ob_age {
                ob.mitigate();
                continue;
            }

            // Test kontrolü: Fiyat OB zone'a dokundu
            if candle.low <= ob.high && !ob.is_tested {
                ob.mark_tested();
            }

            // Mitigation: Fiyat OB low'un altına kapanış yaptı
            if candle.close < ob.low {
                ob.mitigate();
            }
        }

        // Bearish OB'ler: Fiyat OB high'ın üstüne çıkarsa → mitigate
        for ob in self.bearish_obs.iter_mut() {
            if !ob.is_valid {
                continue;
            }

            // Yaş kontrolü
            if current_idx.saturating_sub(ob.candle_idx) > self.max_ob_age {
                ob.mitigate();
                continue;
            }

            // Test kontrolü
            if candle.high >= ob.low && !ob.is_tested {
                ob.mark_tested();
            }

            // Mitigation: Fiyat OB high üzerine kapanış yaptı
            if candle.close > ob.high {
                ob.mitigate();
            }
        }

        // Geçersiz OB'leri temizle (periyodik)
        if current_idx % 50 == 0 {
            self.bullish_obs.retain(|ob| ob.is_valid);
            self.bearish_obs.retain(|ob| ob.is_valid);
        }
    }

    // =========================================================================
    // TP / SL HESAPLAMA FONKSİYONLARI
    // =========================================================================

    /// LONG pozisyon için en iyi SL seviyesi (Order Block tabanlı)
    /// En yakın geçerli bullish OB'nin low'u - buffer
    pub fn get_long_sl(
        &self,
        entry_price: Decimal,
        atr: Decimal,
        fallback_pivot_low: Option<Decimal>,
    ) -> Decimal {
        let buffer = atr * Decimal::from_f64(0.5).unwrap(); // %50 of ATR as buffer (Daha esnek SL)
        let min_sl_distance = atr * Decimal::from_f64(1.5).unwrap(); // Min 1.5 ATR distance

        // Entry'nin altındaki en yakın geçerli bullish OB'yi bul
        let ob_sl = self
            .bullish_obs
            .iter()
            .filter(|ob| ob.is_valid && ob.low < entry_price)
            .max_by_key(|ob| ob.low) // Entry'e en yakın olanı seç
            .map(|ob| ob.low - buffer);

        // OB bulunamazsa pivot low'a düş, o da yoksa ATR tabanlı SL (2.0x) - Daha esnek
        let sl = ob_sl
            .or(fallback_pivot_low.map(|p| p - buffer))
            .unwrap_or_else(|| entry_price - (atr * Decimal::from_f64(2.0).unwrap()));

        // Minimum mesafe koruması
        if (entry_price - sl) < min_sl_distance {
            entry_price - min_sl_distance
        } else {
            sl
        }
    }

    /// SHORT pozisyon için en iyi SL seviyesi (Order Block tabanlı)
    /// En yakın geçerli bearish OB'nin high'ı + buffer
    pub fn get_short_sl(
        &self,
        entry_price: Decimal,
        atr: Decimal,
        fallback_pivot_high: Option<Decimal>,
    ) -> Decimal {
        let buffer = atr * Decimal::from_f64(0.5).unwrap(); // %50 of ATR as buffer (Daha esnek SL)
        let min_sl_distance = atr * Decimal::from_f64(1.5).unwrap();

        // Entry'nin üstündeki en yakın geçerli bearish OB'yi bul
        let ob_sl = self
            .bearish_obs
            .iter()
            .filter(|ob| ob.is_valid && ob.high > entry_price)
            .min_by_key(|ob| ob.high) // Entry'e en yakın olanı seç
            .map(|ob| ob.high + buffer);

        // OB bulunamazsa pivot high'a düş, o da yoksa ATR tabanlı SL (2.0x) - Daha esnek
        let sl = ob_sl
            .or(fallback_pivot_high.map(|p| p + buffer))
            .unwrap_or_else(|| entry_price + (atr * Decimal::from_f64(2.0).unwrap()));

        // Minimum mesafe koruması
        if (sl - entry_price) < min_sl_distance {
            entry_price + min_sl_distance
        } else {
            sl
        }
    }

    /// LONG pozisyon için en iyi TP seviyesi (Order Block tabanlı)
    /// En yakın geçerli bearish OB (supply zone) = direnç
    pub fn get_long_tp(
        &self,
        entry_price: Decimal,
        sl_price: Decimal,
        min_rr: Decimal,
        atr: Decimal,
        fallback_pivot_highs: &VecDeque<Decimal>,
    ) -> Decimal {
        let min_profit_dist = atr * Decimal::from_f64(1.0).unwrap(); // Min 1.0 ATR distance
        let risk = (entry_price - sl_price).abs();

        // En yakın geçerli bearish OB (supply zone) → direnç noktası
        let ob_tp = self
            .bearish_obs
            .iter()
            .filter(|ob| ob.is_valid && ob.low > entry_price + min_profit_dist)
            .min_by_key(|ob| ob.low) // Entry'e en yakın supply zone
            .map(|ob| ob.low); // Supply zone'un alt sınırı = TP

        // Pivot tabanlı fallback
        let pivot_tp = fallback_pivot_highs
            .iter()
            .filter(|&&p| p > entry_price + min_profit_dist)
            .min()
            .copied();

        // OB TP > Pivot TP > Fallback RR
        let tp = ob_tp
            .or(pivot_tp)
            .unwrap_or_else(|| entry_price + risk * min_rr);

        // RR minimum kontrolü: TP en az min_rr kadar uzakta olmalı
        let min_tp = entry_price + risk * min_rr;
        if tp < min_tp {
            min_tp
        } else {
            tp
        }
    }

    /// SHORT pozisyon için en iyi TP seviyesi (Order Block tabanlı)
    /// En yakın geçerli bullish OB (demand zone) = destek
    pub fn get_short_tp(
        &self,
        entry_price: Decimal,
        sl_price: Decimal,
        min_rr: Decimal,
        atr: Decimal,
        fallback_pivot_lows: &VecDeque<Decimal>,
    ) -> Decimal {
        let min_profit_dist = atr * Decimal::from_f64(1.0).unwrap();
        let risk = (sl_price - entry_price).abs();

        // En yakın geçerli bullish OB (demand zone) → destek noktası
        let ob_tp = self
            .bullish_obs
            .iter()
            .filter(|ob| ob.is_valid && ob.high < entry_price - min_profit_dist)
            .max_by_key(|ob| ob.high) // Entry'e en yakın demand zone
            .map(|ob| ob.high); // Demand zone'un üst sınırı = TP

        // Pivot tabanlı fallback
        let pivot_tp = fallback_pivot_lows
            .iter()
            .filter(|&&p| p < entry_price - min_profit_dist)
            .max()
            .copied();

        let tp = ob_tp
            .or(pivot_tp)
            .unwrap_or_else(|| entry_price - risk * min_rr);

        // RR minimum kontrolü
        let min_tp = entry_price - risk * min_rr;
        if tp > min_tp {
            min_tp
        } else {
            tp
        }
    }

    // =========================================================================
    // AKILLI TP/SL HESAPLAMA (BİRLEŞİK)
    // =========================================================================

    /// Order Block tabanlı SL/TP hesapla
    /// Tüm mantığı birleştiren ana fonksiyon
    pub fn calculate_ob_sl_tp(
        &self,
        direction: &SignalType,
        entry_price: Decimal,
        atr: Decimal,
        fallback_pivot_low: Option<Decimal>,
        fallback_pivot_high: Option<Decimal>,
        pivot_high_history: &VecDeque<Decimal>,
        pivot_low_history: &VecDeque<Decimal>,
    ) -> (Decimal, Decimal) {
        // Geniş R'lar(Risk/Reward) için minimum rR oranını artırıyoruz
        let min_rr = Decimal::from_f64(2.0).unwrap();

        match direction {
            SignalType::LONG => {
                let sl = self.get_long_sl(entry_price, atr, fallback_pivot_low);
                let tp = self.get_long_tp(entry_price, sl, min_rr, atr, pivot_high_history);
                (sl, tp)
            }
            SignalType::SHORT => {
                let sl = self.get_short_sl(entry_price, atr, fallback_pivot_high);
                let tp = self.get_short_tp(entry_price, sl, min_rr, atr, pivot_low_history);
                (sl, tp)
            }
        }
    }

    // =========================================================================
    // İSTATİSTİK / BİLGİ
    // =========================================================================

    /// Aktif (geçerli) bullish OB sayısı
    pub fn valid_bullish_count(&self) -> usize {
        self.bullish_obs.iter().filter(|ob| ob.is_valid).count()
    }

    /// Aktif (geçerli) bearish OB sayısı
    pub fn valid_bearish_count(&self) -> usize {
        self.bearish_obs.iter().filter(|ob| ob.is_valid).count()
    }

    /// En yakın bullish OB (destek)
    pub fn nearest_bullish_ob(&self, price: Decimal) -> Option<&OrderBlock> {
        self.bullish_obs
            .iter()
            .filter(|ob| ob.is_valid && ob.high <= price)
            .max_by_key(|ob| ob.high)
    }

    /// En yakın bearish OB (direnç)
    pub fn nearest_bearish_ob(&self, price: Decimal) -> Option<&OrderBlock> {
        self.bearish_obs
            .iter()
            .filter(|ob| ob.is_valid && ob.low >= price)
            .min_by_key(|ob| ob.low)
    }

    /// Debug: Tüm aktif OB'leri listele
    pub fn summary(&self) -> String {
        let bull_count = self.valid_bullish_count();
        let bear_count = self.valid_bearish_count();
        format!(
            "OB Tracker: {} bullish (demand) | {} bearish (supply)",
            bull_count, bear_count
        )
    }
}

impl Default for OrderBlockTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candle(open: f64, high: f64, low: f64, close: f64) -> Candle {
        Candle {
            open_time: Utc::now(),
            open: Decimal::from_f64(open).unwrap(),
            high: Decimal::from_f64(high).unwrap(),
            low: Decimal::from_f64(low).unwrap(),
            close: Decimal::from_f64(close).unwrap(),
            volume: Decimal::from(1000),
            close_time: None,
        }
    }

    #[test]
    fn test_bullish_ob_detection() {
        let mut tracker = OrderBlockTracker::new();
        let atr = Some(Decimal::from(100));

        // Mum 1: Bearish (potansiyel bullish OB)
        let c1 = make_candle(100.0, 105.0, 95.0, 96.0);
        tracker.update(&c1, 1, atr, false, false);

        // Mum 2: Neutral
        let c2 = make_candle(96.0, 98.0, 94.0, 97.0);
        tracker.update(&c2, 2, atr, false, false);

        // Mum 3: Güçlü bullish BOS (range > ATR)
        let c3 = make_candle(97.0, 210.0, 96.0, 200.0);
        tracker.update(&c3, 3, atr, true, false);

        assert!(tracker.valid_bullish_count() > 0);
    }

    #[test]
    fn test_bearish_ob_detection() {
        let mut tracker = OrderBlockTracker::new();
        let atr = Some(Decimal::from(100));

        // Mum 1: Bullish (potansiyel bearish OB)
        let c1 = make_candle(200.0, 210.0, 198.0, 208.0);
        tracker.update(&c1, 1, atr, false, false);

        // Mum 2: Güçlü bearish BOS
        let c2 = make_candle(208.0, 209.0, 90.0, 95.0);
        tracker.update(&c2, 2, atr, false, true);

        assert!(tracker.valid_bearish_count() > 0);
    }

    #[test]
    fn test_ob_mitigation() {
        let mut tracker = OrderBlockTracker::new();
        let atr = Some(Decimal::from(100));

        // Bullish OB oluştur
        let c1 = make_candle(100.0, 105.0, 95.0, 96.0);
        tracker.update(&c1, 1, atr, false, false);
        let c2 = make_candle(96.0, 210.0, 95.0, 200.0);
        tracker.update(&c2, 2, atr, true, false);

        assert_eq!(tracker.valid_bullish_count(), 1);

        // Fiyat OB low altına düşsün → mitigate
        let c3 = make_candle(200.0, 201.0, 80.0, 85.0);
        tracker.update(&c3, 3, atr, false, false);

        assert_eq!(tracker.valid_bullish_count(), 0);
    }

    #[test]
    fn test_long_sl_tp_calculation() {
        let mut tracker = OrderBlockTracker::new();
        let atr = Decimal::from(100);

        // Bullish OB at low=95
        let c1 = make_candle(100.0, 105.0, 95.0, 96.0);
        tracker.update(&c1, 1, Some(atr), false, false);
        let c2 = make_candle(96.0, 210.0, 95.0, 200.0);
        tracker.update(&c2, 2, Some(atr), true, false);

        // Bearish OB at high=310
        let c3 = make_candle(300.0, 310.0, 298.0, 308.0);
        tracker.update(&c3, 3, Some(atr), false, false);
        let c4 = make_candle(308.0, 309.0, 100.0, 105.0);
        tracker.update(&c4, 4, Some(atr), false, true);

        let entry = Decimal::from(200);
        let pivot_highs = VecDeque::new();
        let pivot_lows = VecDeque::new();

        let (sl, tp) = tracker.calculate_ob_sl_tp(
            &SignalType::LONG,
            entry,
            atr,
            None,
            None,
            &pivot_highs,
            &pivot_lows,
        );

        // SL should be below entry
        assert!(sl < entry, "SL {} should be below entry {}", sl, entry);
        // TP should be above entry
        assert!(tp > entry, "TP {} should be above entry {}", tp, entry);
    }
}
