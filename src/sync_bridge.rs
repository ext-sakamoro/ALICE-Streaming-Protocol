//! ALICE-Streaming-Protocol × ALICE-Sync Bridge
//!
//! Embed Sync events (entity motion, spawn, despawn) inside ASP D-packets.
//! Enables multiplayer game state synchronization over the ASP transport layer.

use alice_sync::{Event, EventKind};
use crate::{DPacketPayload, MotionVector, AspPacket};

/// Convert Sync motion events into ASP D-packet motion vectors.
///
/// Each Sync motion event maps to an ASP MotionVector:
/// - `entity` → block position (block_x = entity % width, block_y = entity / width)
/// - `delta` → motion vector (dx, dy) with SAD = magnitude
///
/// # Arguments
///
/// - `events`: Sync events to embed
/// - `ref_sequence`: ASP reference frame sequence number
/// - `grid_width`: Virtual grid width for entity → block mapping
pub fn sync_events_to_d_packet(
    events: &[Event],
    ref_sequence: u32,
    grid_width: u16,
) -> AspPacket {
    let mut payload = DPacketPayload::new(ref_sequence);

    for event in events {
        match &event.kind {
            EventKind::Motion { entity, delta } => {
                let bx = (*entity % grid_width as u32) as u16;
                let by = (*entity / grid_width as u32) as u16;
                let dx = delta[0]; // Already i16
                let dy = delta[1];
                let sad = ((dx as i32).abs() + (dy as i32).abs()) as u32;
                payload.add_motion_vector(MotionVector::new(bx, by, dx, dy, sad));
            }
            _ => {} // Only motion events map to D-packets
        }
    }

    AspPacket::create_d_packet(ref_sequence + 1, payload)
        .expect("D-packet creation should not fail")
}

/// Extract Sync-compatible motion data from an ASP D-packet's motion vectors.
///
/// Inverse of `sync_events_to_d_packet`. Reconstructs approximate
/// entity IDs and motion deltas from ASP MotionVectors.
///
/// # Arguments
///
/// - `motion_vectors`: ASP motion vectors from a D-packet
/// - `grid_width`: Virtual grid width (must match encoder)
pub fn d_packet_to_sync_motions(
    motion_vectors: &[MotionVector],
    grid_width: u16,
) -> Vec<(u32, [i16; 3])> {
    motion_vectors
        .iter()
        .map(|mv| {
            let entity = mv.block_y as u32 * grid_width as u32 + mv.block_x as u32;
            let delta = [mv.dx, mv.dy, 0];
            (entity, delta)
        })
        .collect()
}

/// Count how many sync events are motion-type (D-packet compatible).
pub fn count_motion_events(events: &[Event]) -> usize {
    events.iter().filter(|e| matches!(&e.kind, EventKind::Motion { .. })).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_to_d_packet_roundtrip() {
        let events = vec![
            Event::new(EventKind::Motion {
                entity: 5,
                delta: [1024, 2048, 0],
            }),
            Event::new(EventKind::Motion {
                entity: 10,
                delta: [-1024, 0, 0],
            }),
        ];

        let grid_width = 16;
        let packet = sync_events_to_d_packet(&events, 100, grid_width);
        assert!(!packet.is_keyframe());

        // Extract motion vectors back
        let d_payload = match &packet.payload {
            crate::AspPayload::DPacket(d) => d,
            _ => panic!("Expected D-packet"),
        };

        let motions = d_packet_to_sync_motions(&d_payload.motion_vectors, grid_width);
        assert_eq!(motions.len(), 2);

        // Entity IDs should roundtrip
        assert_eq!(motions[0].0, 5);
        assert_eq!(motions[1].0, 10);
    }

    #[test]
    fn test_non_motion_events_skipped() {
        let events = vec![
            Event::new(EventKind::Spawn {
                entity: 1,
                kind: 0,
                pos: [0, 0, 0],
            }),
            Event::new(EventKind::Motion {
                entity: 2,
                delta: [100, 200, 300],
            }),
        ];

        assert_eq!(count_motion_events(&events), 1);
    }
}
