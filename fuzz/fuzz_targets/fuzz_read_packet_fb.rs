//! Fuzz target: FlatBuffers 経路 (read_packet / get_packet_type) の panic-freedom
//!
//! libasp の default serialization は FlatBuffers (zero-copy)、cross-language
//! で最重要 network parser 攻撃者制御の任意 byte 列に対し flatbuffers verifier
//! が panic せず Result で返ることを保証する
//!
//! 加えて get_packet_type は packet 種別 dispatch の前段 discriminator として
//! 使われるため、任意 byte で panic しないことは特に重要
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use libasp::flatbuffers_api::{get_packet_type, read_packet};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // 入力サイズ upper bound (fuzz corpus 巨大化予防、8 MB 相当)
    if data.len() > 8 * 1024 * 1024 {
        return;
    }

    // 1. get_packet_type: packet 種別判定 (dispatch 前段) — 任意 byte で Result
    let _ = get_packet_type(data);

    // 2. read_packet: full parse — verifier 込みで panic-free であること
    let _ = read_packet(data);
});
