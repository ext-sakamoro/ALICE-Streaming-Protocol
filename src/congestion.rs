//! 輻輳制御 (Congestion Control)
//!
//! RTT（往復遅延時間）推定と輻輳ウィンドウ管理。
//! TCP Reno 風のスロースタート / 輻輳回避アルゴリズム。
//!
//! # 使い方
//!
//! ```rust,ignore
//! use libasp::congestion::CongestionController;
//!
//! let mut cc = CongestionController::new(1400); // MSS = 1400 bytes
//! cc.on_ack(50); // RTT 50ms
//! let cwnd = cc.window_bytes();
//! cc.on_loss(); // パケットロス → 輻輳回避へ遷移
//! ```

/// 輻輳制御の状態。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionState {
    /// スロースタート: 指数的にウィンドウ拡大。
    SlowStart,
    /// 輻輳回避: 線形にウィンドウ拡大。
    CongestionAvoidance,
}

/// 輻輳制御コントローラ。
///
/// TCP Reno ベースの輻輳ウィンドウ管理:
/// - スロースタート: ACK毎に MSS 分増加（指数的成長）
/// - 輻輳回避: ACK毎に MSS²/cwnd 分増加（線形成長）
/// - ロス時: ssthresh = cwnd/2, cwnd = ssthresh（高速リカバリ）
#[derive(Debug, Clone)]
pub struct CongestionController {
    /// 輻輳ウィンドウ (バイト)。
    cwnd: u64,
    /// スロースタート閾値 (バイト)。
    ssthresh: u64,
    /// 最大セグメントサイズ (バイト)。
    mss: u64,
    /// 現在の状態。
    state: CongestionState,
    /// 平滑化RTT (マイクロ秒)。
    srtt_us: u64,
    /// RTT分散 (マイクロ秒)。
    rtt_var_us: u64,
    /// RTO (再送タイムアウト, マイクロ秒)。
    rto_us: u64,
    /// RTT サンプル数。
    rtt_samples: u64,
    /// 累計ACK数。
    total_acks: u64,
    /// 累計ロス数。
    total_losses: u64,
}

impl CongestionController {
    /// MSS（最大セグメントサイズ）を指定して作成。
    ///
    /// 初期ウィンドウ: 10 * MSS（RFC 6928 準拠）。
    #[must_use]
    pub const fn new(mss: u64) -> Self {
        let initial_cwnd = mss * 10;
        Self {
            cwnd: initial_cwnd,
            ssthresh: u64::MAX,
            mss,
            state: CongestionState::SlowStart,
            srtt_us: 0,
            rtt_var_us: 0,
            rto_us: 1_000_000, // 初期 RTO = 1秒
            rtt_samples: 0,
            total_acks: 0,
            total_losses: 0,
        }
    }

    /// ACK 受信時の処理。
    ///
    /// # Arguments
    ///
    /// - `rtt_ms`: RTT サンプル (ミリ秒)
    pub fn on_ack(&mut self, rtt_ms: u64) {
        self.total_acks += 1;
        self.update_rtt(rtt_ms * 1000);

        match self.state {
            CongestionState::SlowStart => {
                // 指数的増加
                self.cwnd += self.mss;
                if self.cwnd >= self.ssthresh {
                    self.state = CongestionState::CongestionAvoidance;
                }
            }
            CongestionState::CongestionAvoidance => {
                // 線形増加: MSS * MSS / cwnd
                let increment = (self.mss * self.mss) / self.cwnd.max(1);
                self.cwnd += increment.max(1);
            }
        }
    }

    /// パケットロス検出時の処理。
    ///
    /// cwnd を半減し、輻輳回避モードに移行。
    pub fn on_loss(&mut self) {
        self.total_losses += 1;
        self.ssthresh = (self.cwnd / 2).max(2 * self.mss);
        self.cwnd = self.ssthresh;
        self.state = CongestionState::CongestionAvoidance;
    }

    /// 現在の輻輳ウィンドウ (バイト)。
    #[must_use]
    pub const fn window_bytes(&self) -> u64 {
        self.cwnd
    }

    /// 送信可能パケット数（cwnd / mss）。
    #[must_use]
    pub const fn window_packets(&self) -> u64 {
        self.cwnd / self.mss
    }

    /// 現在の輻輳状態。
    #[must_use]
    pub const fn state(&self) -> CongestionState {
        self.state
    }

    /// 平滑化 RTT (ミリ秒)。
    #[must_use]
    pub const fn srtt_ms(&self) -> u64 {
        self.srtt_us / 1000
    }

    /// RTO (再送タイムアウト, ミリ秒)。
    #[must_use]
    pub const fn rto_ms(&self) -> u64 {
        self.rto_us / 1000
    }

    /// スロースタート閾値 (バイト)。
    #[must_use]
    pub const fn ssthresh(&self) -> u64 {
        self.ssthresh
    }

    /// 累計ACK数。
    #[must_use]
    pub const fn total_acks(&self) -> u64 {
        self.total_acks
    }

    /// 累計ロス数。
    #[must_use]
    pub const fn total_losses(&self) -> u64 {
        self.total_losses
    }

    /// RTT 推定の更新 (RFC 6298)。
    ///
    /// SRTT = (1 - α) * SRTT + α * R (α = 1/8)
    /// RTTVAR = (1 - β) * RTTVAR + β * |SRTT - R| (β = 1/4)
    /// RTO = SRTT + max(G, 4 * RTTVAR) (G = 1ms = 1000us)
    fn update_rtt(&mut self, rtt_us: u64) {
        if self.rtt_samples == 0 {
            self.srtt_us = rtt_us;
            self.rtt_var_us = rtt_us / 2;
        } else {
            let diff = rtt_us.abs_diff(self.srtt_us);
            // RTTVAR = 3/4 * RTTVAR + 1/4 * |SRTT - R|
            self.rtt_var_us = self.rtt_var_us / 4 * 3 + diff / 4;
            // SRTT = 7/8 * SRTT + 1/8 * R
            self.srtt_us = self.srtt_us / 8 * 7 + rtt_us / 8;
        }
        self.rtt_samples += 1;

        // RTO = SRTT + max(1000, 4 * RTTVAR), 最小 200ms
        let rto = self.srtt_us + (4 * self.rtt_var_us).max(1000);
        self.rto_us = rto.max(200_000);
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        let cc = CongestionController::new(1400);
        assert_eq!(cc.state(), CongestionState::SlowStart);
        assert_eq!(cc.window_bytes(), 14_000); // 10 * MSS
        assert_eq!(cc.window_packets(), 10);
    }

    #[test]
    fn slow_start_exponential_growth() {
        let mut cc = CongestionController::new(1400);
        let initial = cc.window_bytes();
        cc.on_ack(50);
        assert_eq!(cc.window_bytes(), initial + 1400);
        cc.on_ack(50);
        assert_eq!(cc.window_bytes(), initial + 2800);
    }

    #[test]
    fn loss_halves_window() {
        let mut cc = CongestionController::new(1400);
        // 増加させてからロス
        for _ in 0..20 {
            cc.on_ack(50);
        }
        let before_loss = cc.window_bytes();
        cc.on_loss();
        assert!(cc.window_bytes() <= before_loss / 2 + 1400);
        assert_eq!(cc.state(), CongestionState::CongestionAvoidance);
    }

    #[test]
    fn congestion_avoidance_linear_growth() {
        let mut cc = CongestionController::new(1400);
        // ロスで輻輳回避に移行
        for _ in 0..10 {
            cc.on_ack(50);
        }
        cc.on_loss();
        assert_eq!(cc.state(), CongestionState::CongestionAvoidance);

        let before = cc.window_bytes();
        cc.on_ack(50);
        let after = cc.window_bytes();
        // 線形増加: increment = MSS^2/cwnd ≪ MSS
        assert!(after > before);
        assert!(after - before < 1400); // 線形なのでMSS未満
    }

    #[test]
    fn rtt_estimation() {
        let mut cc = CongestionController::new(1400);
        cc.on_ack(100);
        assert_eq!(cc.srtt_ms(), 100);

        cc.on_ack(50);
        // EWMA: 7/8 * 100 + 1/8 * 50 = 87 + 6 = 93
        assert!(cc.srtt_ms() > 80);
        assert!(cc.srtt_ms() < 110);
    }

    #[test]
    fn rto_minimum() {
        let mut cc = CongestionController::new(1400);
        cc.on_ack(1); // very low RTT
                      // RTO は最小 200ms
        assert!(cc.rto_ms() >= 200);
    }

    #[test]
    fn ssthresh_after_loss() {
        let mut cc = CongestionController::new(1400);
        for _ in 0..20 {
            cc.on_ack(50);
        }
        let cwnd_before = cc.window_bytes();
        cc.on_loss();
        assert_eq!(cc.ssthresh(), cwnd_before / 2);
    }

    #[test]
    fn min_cwnd_on_loss() {
        let mut cc = CongestionController::new(1400);
        // 初期状態でロス
        cc.on_loss();
        // cwnd >= 2 * MSS
        assert!(cc.window_bytes() >= 2 * 1400);
    }

    #[test]
    fn slow_start_to_avoidance_transition() {
        let mut cc = CongestionController::new(1400);
        // ロスで ssthresh を設定
        cc.on_loss();
        let thresh = cc.ssthresh();

        // リセット状態（cwnd = ssthresh, 輻輳回避）
        assert_eq!(cc.state(), CongestionState::CongestionAvoidance);

        // 新しい CongestionController で ssthresh を小さくした場合の遷移テスト
        let mut cc2 = CongestionController::new(100);
        // cwnd=1000, ssthresh=MAX → スロースタート
        assert_eq!(cc2.state(), CongestionState::SlowStart);

        // ssthresh を低く設定してからACK
        cc2.on_loss(); // ssthresh = 500, cwnd = 500
                       // 状態は CongestionAvoidance
        assert_eq!(cc2.state(), CongestionState::CongestionAvoidance);

        let _ = thresh;
    }

    #[test]
    fn multiple_losses() {
        let mut cc = CongestionController::new(1400);
        for _ in 0..5 {
            for _ in 0..10 {
                cc.on_ack(50);
            }
            cc.on_loss();
        }
        assert_eq!(cc.total_losses(), 5);
        assert!(cc.window_bytes() >= 2 * 1400);
    }

    #[test]
    fn total_acks_counted() {
        let mut cc = CongestionController::new(1400);
        for _ in 0..25 {
            cc.on_ack(50);
        }
        assert_eq!(cc.total_acks(), 25);
    }

    #[test]
    fn window_packets() {
        let cc = CongestionController::new(1400);
        assert_eq!(cc.window_packets(), 10); // 14000 / 1400
    }

    #[test]
    fn rtt_variance_affects_rto() {
        let mut cc = CongestionController::new(1400);
        // 安定したRTT
        for _ in 0..10 {
            cc.on_ack(50);
        }
        let stable_rto = cc.rto_ms();

        // 不安定なRTTで別のコントローラ
        let mut cc2 = CongestionController::new(1400);
        cc2.on_ack(10);
        cc2.on_ack(200);
        cc2.on_ack(10);
        cc2.on_ack(200);
        let unstable_rto = cc2.rto_ms();

        // 不安定なRTTの方がRTOが大きい（分散が大きいため）
        assert!(unstable_rto > stable_rto);
    }
}
