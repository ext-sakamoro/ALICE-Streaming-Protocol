//! ALICE-Streaming-Protocol × ALICE-Physics Bridge
//!
//! Encode physics body state deltas as ASP D-packets for streaming
//! deterministic physics simulations over the network.

use crate::{AspPacket, DPacketPayload, MotionVector};
use alice_physics::{Fix128, PhysicsWorld, RigidBody, Vec3Fix};

/// Snapshot of physics body positions for delta computation.
pub struct PhysicsSnapshot {
    /// Body positions at snapshot time (f32 for delta computation).
    positions: Vec<[f32; 3]>,
}

impl PhysicsSnapshot {
    /// Capture current body positions from a physics world.
    pub fn capture(world: &PhysicsWorld) -> Self {
        let positions = world
            .bodies
            .iter()
            .map(|body| {
                let (x, y, z) = body.position.to_f32();
                [x, y, z]
            })
            .collect();
        Self { positions }
    }

    /// Number of bodies in this snapshot.
    pub fn body_count(&self) -> usize {
        self.positions.len()
    }

    /// Compute deltas between this snapshot and the current world state,
    /// and encode them as an ASP D-packet.
    ///
    /// Bodies with zero delta are skipped for bandwidth efficiency.
    ///
    /// # Arguments
    ///
    /// - `world`: Current physics world state
    /// - `ref_sequence`: ASP reference frame sequence number
    /// - `grid_width`: Virtual grid width for body → block mapping
    pub fn delta_to_d_packet(
        &self,
        world: &PhysicsWorld,
        ref_sequence: u32,
        grid_width: u16,
    ) -> AspPacket {
        let mut payload = DPacketPayload::new(ref_sequence);

        for (i, body) in world.bodies.iter().enumerate() {
            if i >= self.positions.len() {
                break; // New bodies added since snapshot
            }

            let (cx, cy, _cz) = body.position.to_f32();
            let prev = &self.positions[i];
            let dx = cx - prev[0];
            let dy = cy - prev[1];

            // Skip zero-delta bodies (threshold for float comparison)
            if dx.abs() < 1e-6 && dy.abs() < 1e-6 {
                continue;
            }

            // Map body index to grid position
            let bx = (i as u32 % grid_width as u32) as u16;
            let by = (i as u32 / grid_width as u32) as u16;

            // Scale deltas to i16 range
            let mvx = (dx * 100.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            let mvy = (dy * 100.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            let sad = ((mvx as i32).abs() + (mvy as i32).abs()) as u32;

            payload.add_motion_vector(MotionVector::new(bx, by, mvx, mvy, sad));
        }

        AspPacket::create_d_packet(ref_sequence + 1, payload)
            .expect("D-packet creation should not fail")
    }
}

/// Apply D-packet motion deltas back to approximate body positions.
///
/// Used on the receiver side to reconstruct physics state from streamed packets.
///
/// Returns a list of (body_index, delta_x, delta_y) tuples.
pub fn d_packet_to_body_deltas(
    motion_vectors: &[MotionVector],
    grid_width: u16,
) -> Vec<(usize, f32, f32)> {
    motion_vectors
        .iter()
        .map(|mv| {
            let body_idx = mv.block_y as usize * grid_width as usize + mv.block_x as usize;
            let dx = mv.dx as f32 / 100.0;
            let dy = mv.dy as f32 / 100.0;
            (body_idx, dx, dy)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alice_physics::PhysicsConfig;

    #[test]
    fn test_physics_snapshot_delta() {
        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);

        // Add two bodies
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(0, 0, 0),
            Fix128::ONE,
        ));
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(10, 0, 0),
            Fix128::ONE,
        ));

        // Snapshot before simulation
        let snapshot = PhysicsSnapshot::capture(&world);
        assert_eq!(snapshot.body_count(), 2);

        // Simulate (bodies will fall due to gravity)
        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..10 {
            world.step(dt);
        }

        // Encode delta as D-packet
        let packet = snapshot.delta_to_d_packet(&world, 100, 16);
        assert!(!packet.is_keyframe());
    }

    #[test]
    fn test_zero_delta_skipped() {
        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);

        // Static body — won't move
        world.add_body(RigidBody::new_static(Vec3Fix::ZERO));

        let snapshot = PhysicsSnapshot::capture(&world);

        // No movement, D-packet should have no motion vectors
        let packet = snapshot.delta_to_d_packet(&world, 1, 16);
        let d_payload = match &packet.payload {
            crate::AspPayload::DPacket(d) => d,
            _ => panic!("Expected D-packet"),
        };
        assert!(d_payload.motion_vectors.is_empty());
    }
}
