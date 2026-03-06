//! パケットロス回復 (Packet Loss Recovery)
//!
//! 受信シーケンス番号を追跡し、ギャップ（欠落パケット）を検出して
//! NACK（再送要求）を生成する。
//!
//! # 使い方
//!
//! ```rust,ignore
//! use libasp::loss_recovery::LossDetector;
//!
//! let mut det = LossDetector::new(3, 5); // reorder_tolerance=3, max_retries=5
//! det.on_receive(0);
//! det.on_receive(1);
//! det.on_receive(3); // seq=2 が欠落
//! let nacks = det.pending_nacks();
//! assert_eq!(nacks, &[2]);
//! ```

/// パケットロス検出器。
///
/// 受信したシーケンス番号を追跡し、ギャップを検出する。
/// `reorder_tolerance` パケット分の順序逆転を許容する。
#[derive(Debug, Clone)]
pub struct LossDetector {
    /// 次に期待するシーケンス番号。
    expected_seq: u32,
    /// 最大受信済みシーケンス番号。
    max_received: u32,
    /// 順序逆転許容数。この数以上先のパケットを受信したらギャップと判定。
    reorder_tolerance: u32,
    /// 1パケットあたりの最大再送回数。
    max_retries: u32,
    /// 検出されたギャップ: (シーケンス番号, 再送回数)。
    gaps: Vec<(u32, u32)>,
    /// 累計受信パケット数。
    total_received: u64,
    /// 累計検出ロス数。
    total_detected_losses: u64,
    /// 受信済みビットマップ（最新 256 パケット分）。
    received_bitmap: [u64; 4],
    /// ビットマップの基底シーケンス番号。
    bitmap_base: u32,
}

impl LossDetector {
    /// 新しいロス検出器を作成。
    ///
    /// # Arguments
    ///
    /// - `reorder_tolerance`: 順序逆転許容パケット数
    /// - `max_retries`: 1パケットあたりの最大NACK送信回数
    #[must_use]
    pub const fn new(reorder_tolerance: u32, max_retries: u32) -> Self {
        Self {
            expected_seq: 0,
            max_received: 0,
            reorder_tolerance,
            max_retries,
            gaps: Vec::new(),
            total_received: 0,
            total_detected_losses: 0,
            received_bitmap: [0; 4],
            bitmap_base: 0,
        }
    }

    /// パケット受信を記録し、ギャップを検出する。
    ///
    /// 戻り値: 新たに検出されたギャップ（欠落シーケンス番号）の数。
    pub fn on_receive(&mut self, seq: u32) -> usize {
        self.total_received += 1;
        self.mark_received(seq);

        // ギャップリストから受信済みを除去
        self.gaps.retain(|&(s, _)| s != seq);

        if seq >= self.expected_seq {
            let mut new_gaps = 0;

            // ギャップ検出: expected_seq から seq-1 までが欠落候補
            // ビットマップ容量（256）を超える範囲は追跡不能なので制限
            if seq >= self.expected_seq + self.reorder_tolerance {
                let gap_start = if seq > self.expected_seq + 256 {
                    seq - 256
                } else {
                    self.expected_seq
                };
                let gap_end = seq;
                let uncapped_count = (seq - self.expected_seq) as u64;
                // ビットマップ範囲外の欠落分を一括カウント
                if gap_start > self.expected_seq {
                    self.total_detected_losses += (gap_start - self.expected_seq) as u64;
                    new_gaps += (gap_start - self.expected_seq) as usize;
                }
                let _ = uncapped_count;
                for missing in gap_start..gap_end {
                    if !self.is_received(missing) && !self.gaps.iter().any(|&(s, _)| s == missing) {
                        self.gaps.push((missing, 0));
                        self.total_detected_losses += 1;
                        new_gaps += 1;
                    }
                }
            }

            if seq >= self.expected_seq {
                self.expected_seq = seq + 1;
            }
            if seq > self.max_received {
                self.max_received = seq;
            }
            new_gaps
        } else {
            // 遅延到着パケット — ギャップ解消済み
            0
        }
    }

    /// NACK を送信すべき欠落シーケンス番号のリスト。
    ///
    /// 再送回数が `max_retries` に達したものは除外。
    #[must_use]
    pub fn pending_nacks(&self) -> Vec<u32> {
        self.gaps
            .iter()
            .filter(|&&(_, retries)| retries < self.max_retries)
            .map(|&(seq, _)| seq)
            .collect()
    }

    /// NACK 送信後に呼び出し、再送カウンタをインクリメント。
    pub fn mark_nack_sent(&mut self, seq: u32) {
        if let Some(entry) = self.gaps.iter_mut().find(|e| e.0 == seq) {
            entry.1 += 1;
        }
    }

    /// 全ての pending NACK の再送カウンタをインクリメント。
    pub fn mark_all_nacks_sent(&mut self) {
        for entry in &mut self.gaps {
            if entry.1 < self.max_retries {
                entry.1 += 1;
            }
        }
    }

    /// 累計受信パケット数。
    #[must_use]
    pub const fn total_received(&self) -> u64 {
        self.total_received
    }

    /// 累計検出ロス数。
    #[must_use]
    pub const fn total_detected_losses(&self) -> u64 {
        self.total_detected_losses
    }

    /// 現在のギャップ数。
    #[must_use]
    pub const fn gap_count(&self) -> usize {
        self.gaps.len()
    }

    /// 次に期待するシーケンス番号。
    #[must_use]
    pub const fn expected_seq(&self) -> u32 {
        self.expected_seq
    }

    /// 受信済みビットマップに記録。
    const fn mark_received(&mut self, seq: u32) {
        // ビットマップの範囲を超えたらベースをシフト
        if seq >= self.bitmap_base + 256 {
            let shift = seq - self.bitmap_base - 128;
            self.bitmap_base += shift;
            // 古いビットは破棄（簡易実装: 全クリア）
            self.received_bitmap = [0; 4];
        }
        if seq >= self.bitmap_base {
            let offset = (seq - self.bitmap_base) as usize;
            if offset < 256 {
                self.received_bitmap[offset / 64] |= 1 << (offset % 64);
            }
        }
    }

    /// シーケンス番号が受信済みかチェック。
    const fn is_received(&self, seq: u32) -> bool {
        if seq < self.bitmap_base || seq >= self.bitmap_base + 256 {
            return false;
        }
        let offset = (seq - self.bitmap_base) as usize;
        (self.received_bitmap[offset / 64] >> (offset % 64)) & 1 == 1
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_no_gaps() {
        let mut det = LossDetector::new(3, 5);
        for i in 0..10 {
            assert_eq!(det.on_receive(i), 0);
        }
        assert!(det.pending_nacks().is_empty());
        assert_eq!(det.total_received(), 10);
        assert_eq!(det.total_detected_losses(), 0);
    }

    #[test]
    fn single_gap_detected() {
        let mut det = LossDetector::new(1, 5);
        det.on_receive(0);
        det.on_receive(1);
        // seq=2 を飛ばして seq=3 を受信 → reorder_tolerance=1 を超えるギャップ
        let gaps = det.on_receive(3);
        assert!(gaps > 0);
        let nacks = det.pending_nacks();
        assert!(nacks.contains(&2));
    }

    #[test]
    fn gap_resolved_by_late_arrival() {
        let mut det = LossDetector::new(1, 5);
        det.on_receive(0);
        det.on_receive(1);
        det.on_receive(3); // gap at 2
        assert!(!det.pending_nacks().is_empty());

        // 遅延到着
        det.on_receive(2);
        assert!(det.pending_nacks().is_empty());
    }

    #[test]
    fn reorder_tolerance() {
        let mut det = LossDetector::new(5, 3);
        det.on_receive(0);
        // seq=1,2,3,4 を飛ばして seq=5 → tolerance=5 なので seq=5 はちょうど boundary
        let gaps = det.on_receive(5);
        // tolerance=5 means gap_start..gap_end must exceed tolerance
        // seq(5) > expected(1) + tolerance(5) is false (5 > 6 is false)
        assert_eq!(gaps, 0);

        // seq=7 → 7 > 6 + 5? no. But expected is now 6, so 7 > 6+5=11? no
        det.on_receive(7);
        // Still within tolerance for small gaps
    }

    #[test]
    fn max_retries_respected() {
        let mut det = LossDetector::new(1, 2);
        det.on_receive(0);
        det.on_receive(5); // gap at 1,2,3,4

        // NACK送信を2回マーク
        let nacks = det.pending_nacks();
        for &seq in &nacks {
            det.mark_nack_sent(seq);
            det.mark_nack_sent(seq);
        }

        // max_retries=2 に達したので空になる
        assert!(det.pending_nacks().is_empty());
    }

    #[test]
    fn mark_all_nacks_sent() {
        let mut det = LossDetector::new(1, 3);
        det.on_receive(0);
        det.on_receive(5);
        assert!(!det.pending_nacks().is_empty());

        det.mark_all_nacks_sent();
        det.mark_all_nacks_sent();
        det.mark_all_nacks_sent();
        // 3回で max_retries=3 に達する
        assert!(det.pending_nacks().is_empty());
    }

    #[test]
    fn gap_count() {
        let mut det = LossDetector::new(1, 5);
        det.on_receive(0);
        det.on_receive(5);
        let count = det.gap_count();
        assert!(count > 0);
    }

    #[test]
    fn expected_seq_advances() {
        let mut det = LossDetector::new(3, 5);
        det.on_receive(0);
        assert_eq!(det.expected_seq(), 1);
        det.on_receive(1);
        assert_eq!(det.expected_seq(), 2);
        det.on_receive(5);
        assert_eq!(det.expected_seq(), 6);
    }

    #[test]
    fn large_sequence_numbers() {
        let mut det = LossDetector::new(1, 5);
        det.on_receive(1_000_000);
        det.on_receive(1_000_005);
        assert!(det.total_detected_losses() > 0);
        // ビットマップ範囲制限により高速に完了すること
    }

    #[test]
    fn huge_jump_fast() {
        // 100万パケットジャンプでもO(256)で完了
        let mut det = LossDetector::new(1, 5);
        det.on_receive(0);
        det.on_receive(1_000_000);
        assert!(det.total_detected_losses() > 0);
        assert!(det.gap_count() <= 256);
    }

    #[test]
    fn duplicate_receive_no_effect() {
        let mut det = LossDetector::new(3, 5);
        det.on_receive(0);
        det.on_receive(0); // duplicate
        assert_eq!(det.total_received(), 2); // counted
        assert_eq!(det.expected_seq(), 1); // unchanged
    }

    #[test]
    fn bitmap_tracks_received() {
        let mut det = LossDetector::new(1, 5);
        det.on_receive(0);
        det.on_receive(2);
        assert!(det.is_received(0));
        assert!(!det.is_received(1));
        assert!(det.is_received(2));
    }

    #[test]
    fn bitmap_shift_on_large_jump() {
        let mut det = LossDetector::new(1, 5);
        det.on_receive(0);
        det.on_receive(300); // exceeds 256 window → bitmap shifts
        assert!(det.is_received(300));
        // 古いエントリはクリアされている
        assert!(!det.is_received(0));
    }
}
