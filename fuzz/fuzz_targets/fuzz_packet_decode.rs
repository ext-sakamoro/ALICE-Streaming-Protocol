//! Fuzz target: AspPacket::from_bytes(bytes) の panic-freedom
//!
//! libasp (ALICE-Streaming-Protocol) は network から受信した任意の byte 列を
//! `AspPacket::from_bytes` で parse する 攻撃者制御の任意 byte 列 (I / D / C / S
//! packet 相当、あるいは任意の garbage) を食わせても panic せず AspResult で
//! 返ることを保証する これは network-facing surface の DoS 耐性の要
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use libasp::AspPacket;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // 入力サイズ upper bound (fuzz corpus 巨大化予防、8 MB 相当)
    // 実 network packet は MTU 相当だが、fuzz では大きめに取る
    if data.len() > 8 * 1024 * 1024 {
        return;
    }

    // AspPacket::from_bytes は Result を返し、panic すべきではない
    let _ = AspPacket::from_bytes(data);
});
