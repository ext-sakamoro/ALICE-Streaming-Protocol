//! 適応ビットレート制御 (Adaptive Bitrate Control)
//!
//! AIMD (Additive Increase Multiplicative Decrease) アルゴリズムによる
//! ネットワーク状況に応じた動的ビットレート調整。
//!
//! # 使い方
//!
//! ```rust,ignore
//! use libasp::bitrate::BitrateController;
//!
//! let mut ctrl = BitrateController::new(500_000); // 初期 500 kbps
//! ctrl.on_ack(1024, 50); // 1024バイト送信成功、RTT 50ms
//! ctrl.on_loss();         // パケットロス検出 → ビットレート半減
//! let target = ctrl.target_bps();
//! ```

use crate::types::QualityLevel;

/// ビットレートコントローラの設定。
#[derive(Debug, Clone, Copy)]
pub struct BitrateConfig {
    /// 最小ビットレート (bps)。
    pub min_bps: u64,
    /// 最大ビットレート (bps)。
    pub max_bps: u64,
    /// 加算的増加ステップ (bps/ACK)。
    pub additive_increase_bps: u64,
    /// 乗法的減少係数 (0.0〜1.0)。ロス時にこの割合まで減少。
    pub multiplicative_decrease: f64,
    /// 品質レベルごとのビットレート閾値 (Low, Medium, High, Ultra) [bps]。
    pub quality_thresholds: [u64; 4],
}

impl Default for BitrateConfig {
    fn default() -> Self {
        Self {
            min_bps: 50_000,
            max_bps: 50_000_000,
            additive_increase_bps: 50_000,
            multiplicative_decrease: 0.5,
            quality_thresholds: [100_000, 500_000, 2_000_000, 10_000_000],
        }
    }
}

/// AIMD ビットレートコントローラ。
#[derive(Debug, Clone)]
pub struct BitrateController {
    /// 現在のターゲットビットレート (bps)。
    current_bps: u64,
    /// 設定。
    config: BitrateConfig,
    /// 帯域幅推定値 (bps)。直近の測定に基づく。
    estimated_bandwidth_bps: u64,
    /// 累計送信バイト数。
    total_bytes_sent: u64,
    /// 累計ロス回数。
    total_losses: u64,
    /// 累計ACK回数。
    total_acks: u64,
}

impl BitrateController {
    /// 初期ビットレートを指定して作成。
    #[must_use]
    pub fn new(initial_bps: u64) -> Self {
        Self {
            current_bps: initial_bps,
            config: BitrateConfig::default(),
            estimated_bandwidth_bps: initial_bps,
            total_bytes_sent: 0,
            total_losses: 0,
            total_acks: 0,
        }
    }

    /// カスタム設定で作成。
    #[must_use]
    pub fn with_config(initial_bps: u64, config: BitrateConfig) -> Self {
        Self {
            current_bps: initial_bps.clamp(config.min_bps, config.max_bps),
            config,
            estimated_bandwidth_bps: initial_bps,
            total_bytes_sent: 0,
            total_losses: 0,
            total_acks: 0,
        }
    }

    /// 現在のターゲットビットレート (bps)。
    #[must_use]
    pub const fn target_bps(&self) -> u64 {
        self.current_bps
    }

    /// 推定帯域幅 (bps)。
    #[must_use]
    pub const fn estimated_bandwidth(&self) -> u64 {
        self.estimated_bandwidth_bps
    }

    /// 累計ロス回数。
    #[must_use]
    pub const fn total_losses(&self) -> u64 {
        self.total_losses
    }

    /// 累計ACK回数。
    #[must_use]
    pub const fn total_acks(&self) -> u64 {
        self.total_acks
    }

    /// ACK受信時の処理: 帯域幅推定を更新し、ビットレートを加算的に増加。
    ///
    /// # Arguments
    ///
    /// - `bytes_acked`: ACKされたバイト数
    /// - `rtt_ms`: 往復遅延時間 (ミリ秒)
    pub fn on_ack(&mut self, bytes_acked: u64, rtt_ms: u64) {
        self.total_acks += 1;
        self.total_bytes_sent += bytes_acked;

        // 帯域幅推定: bytes / rtt → bps
        if rtt_ms > 0 {
            let sample_bps = bytes_acked.saturating_mul(8).saturating_mul(1000) / rtt_ms;
            // EWMA (α = 0.125)
            self.estimated_bandwidth_bps = self.estimated_bandwidth_bps / 8 * 7 + sample_bps / 8;
        }

        // 加算的増加 (Additive Increase)
        self.current_bps =
            (self.current_bps + self.config.additive_increase_bps).min(self.config.max_bps);
    }

    /// パケットロス検出時の処理: ビットレートを乗法的に減少。
    pub fn on_loss(&mut self) {
        self.total_losses += 1;

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let new_bps = (self.current_bps as f64 * self.config.multiplicative_decrease) as u64;
        self.current_bps = new_bps.max(self.config.min_bps);
    }

    /// 現在のビットレートに基づく推奨品質レベル。
    #[must_use]
    pub const fn recommended_quality(&self) -> QualityLevel {
        let t = &self.config.quality_thresholds;
        if self.current_bps >= t[3] {
            QualityLevel::Ultra
        } else if self.current_bps >= t[2] {
            QualityLevel::High
        } else if self.current_bps >= t[1] {
            QualityLevel::Medium
        } else {
            QualityLevel::Low
        }
    }

    /// ロス率 (0.0〜1.0)。ACK+ロスの合計が0の場合は0.0。
    #[must_use]
    pub fn loss_rate(&self) -> f64 {
        let total = self.total_acks + self.total_losses;
        if total == 0 {
            return 0.0;
        }
        self.total_losses as f64 / total as f64
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_controller() {
        let ctrl = BitrateController::new(500_000);
        assert_eq!(ctrl.target_bps(), 500_000);
        assert_eq!(ctrl.total_losses(), 0);
        assert_eq!(ctrl.total_acks(), 0);
    }

    #[test]
    fn additive_increase_on_ack() {
        let mut ctrl = BitrateController::new(500_000);
        ctrl.on_ack(1024, 50);
        assert!(ctrl.target_bps() > 500_000);
        assert_eq!(ctrl.total_acks(), 1);
    }

    #[test]
    fn multiplicative_decrease_on_loss() {
        let mut ctrl = BitrateController::new(1_000_000);
        ctrl.on_loss();
        assert_eq!(ctrl.target_bps(), 500_000);
        assert_eq!(ctrl.total_losses(), 1);
    }

    #[test]
    fn min_bitrate_floor() {
        let mut ctrl = BitrateController::new(100_000);
        for _ in 0..20 {
            ctrl.on_loss();
        }
        assert!(ctrl.target_bps() >= ctrl.config.min_bps);
    }

    #[test]
    fn max_bitrate_ceiling() {
        let mut ctrl = BitrateController::new(49_000_000);
        for _ in 0..100 {
            ctrl.on_ack(65536, 10);
        }
        assert!(ctrl.target_bps() <= ctrl.config.max_bps);
    }

    #[test]
    fn bandwidth_estimation() {
        let mut ctrl = BitrateController::new(500_000);
        // 10KB in 10ms = 8Mbps
        ctrl.on_ack(10_000, 10);
        assert!(ctrl.estimated_bandwidth() > 0);
    }

    #[test]
    fn quality_low() {
        let ctrl = BitrateController::new(50_000);
        assert_eq!(ctrl.recommended_quality(), QualityLevel::Low);
    }

    #[test]
    fn quality_medium() {
        let ctrl = BitrateController::new(500_000);
        assert_eq!(ctrl.recommended_quality(), QualityLevel::Medium);
    }

    #[test]
    fn quality_high() {
        let ctrl = BitrateController::new(5_000_000);
        assert_eq!(ctrl.recommended_quality(), QualityLevel::High);
    }

    #[test]
    fn quality_ultra() {
        let ctrl = BitrateController::new(20_000_000);
        assert_eq!(ctrl.recommended_quality(), QualityLevel::Ultra);
    }

    #[test]
    fn loss_rate_empty() {
        let ctrl = BitrateController::new(500_000);
        assert!((ctrl.loss_rate()).abs() < f64::EPSILON);
    }

    #[test]
    fn loss_rate_mixed() {
        let mut ctrl = BitrateController::new(500_000);
        for _ in 0..8 {
            ctrl.on_ack(1024, 50);
        }
        for _ in 0..2 {
            ctrl.on_loss();
        }
        assert!((ctrl.loss_rate() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn aimd_convergence() {
        // AIMD で上昇→ロス→下降→上昇のサイクルを検証
        let mut ctrl = BitrateController::new(500_000);
        for _ in 0..10 {
            ctrl.on_ack(4096, 20);
        }
        let peak = ctrl.target_bps();
        ctrl.on_loss();
        let trough = ctrl.target_bps();
        assert!(trough < peak);
        assert!(trough >= ctrl.config.min_bps);
    }

    #[test]
    fn custom_config() {
        let config = BitrateConfig {
            min_bps: 10_000,
            max_bps: 1_000_000,
            additive_increase_bps: 100_000,
            multiplicative_decrease: 0.75,
            quality_thresholds: [50_000, 200_000, 500_000, 800_000],
        };
        let mut ctrl = BitrateController::with_config(500_000, config);
        assert_eq!(ctrl.target_bps(), 500_000);
        ctrl.on_loss();
        assert_eq!(ctrl.target_bps(), 375_000);
    }

    #[test]
    fn with_config_clamps_initial() {
        let config = BitrateConfig {
            min_bps: 100_000,
            max_bps: 1_000_000,
            ..BitrateConfig::default()
        };
        let ctrl = BitrateController::with_config(50_000, config);
        assert_eq!(ctrl.target_bps(), 100_000);
    }

    #[test]
    fn zero_rtt_ack() {
        let mut ctrl = BitrateController::new(500_000);
        // RTT=0 の場合、帯域幅推定はスキップされるがビットレートは増加
        ctrl.on_ack(1024, 0);
        assert!(ctrl.target_bps() > 500_000);
    }
}
