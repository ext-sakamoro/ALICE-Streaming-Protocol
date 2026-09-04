//! Fuzz target: bincode 経路 (from_bytes_bincode) の panic-freedom + DoS 耐性
//!
//! libasp は legacy な Rust-only serialization として bincode 経路も持つ
//! (`AspPacket::from_bytes_bincode`) canonical CI template 罠 #6 に該当:
//! bincode 2.x では length prefix 上限なしで capacity overflow panic (DoS)
//! を起こしうる 攻撃者制御 byte 列で `Vec::with_capacity(huge)` → abort を
//! 起こさないことを保証する
//!
//! NOTE: from_bytes_bincode は `bincode-compat` feature gated bincode-compat が
//! disable の環境では build error にならないよう `#[cfg]` gate する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠 (罠 #6)

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // 入力サイズ upper bound (fuzz corpus 巨大化予防、8 MB 相当)
    if data.len() > 8 * 1024 * 1024 {
        return;
    }

    // bincode-compat feature 有効時のみ from_bytes_bincode を呼ぶ
    // (feature disable の環境では nop fuzz、build error 予防)
    #[cfg(feature = "bincode-compat")]
    {
        let _ = libasp::AspPacket::from_bytes_bincode(data);
    }

    // feature 有効/無効に関わらず FlatBuffers 経路も同 byte 列で走らせておく
    // (bincode を持たない CI matrix でも fuzz corpus が無駄にならない)
    let _ = libasp::AspPacket::from_bytes(data);
});
