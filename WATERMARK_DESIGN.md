# ALICE Streaming Protocol — Steganographic Watermark Design

**Status:** Design Document (not yet implemented)
**Author:** Moroya Sakamoto
**Date:** 2026-02-24

## 1. Purpose

Embed an invisible, cryptographically-signed watermark into ALICE Streaming
Protocol (ASP) streams. This enables:

- **Origin attribution**: Prove that a stream was produced by an ALICE-powered system.
- **License violation detection**: Identify unauthorized commercial use of AGPL-licensed code.
- **Tampering evidence**: Detect if stream metadata has been stripped or altered.

## 2. Threat Model

| Threat | Description | Watermark Defense |
|--------|-------------|-------------------|
| **Strip & Rebrand** | Competitor removes ALICE headers, claims their own tech | Watermark survives header stripping |
| **Proxy Laundering** | SaaS wraps ASP streams, removes attribution | Watermark embedded in payload, not headers |
| **Selective Extraction** | Extract algorithms, discard protocol framing | Watermark in algorithmic output, not framing |

## 3. Design

### 3.1 Embedding Location

ASP streams consist of:

```
[Magic: 0x41535031] [Header: variable] [Payload chunks...]
```

The watermark is embedded in **payload chunk padding bytes**. ASP chunks are
aligned to 64-byte boundaries; unused padding bytes carry the watermark.

### 3.2 Watermark Structure

```
┌──────────────────────────────────────────────┐
│  Watermark (32 bytes, spread across chunks)  │
├──────────────────────────────────────────────┤
│  [0..4]   Magic: 0xA11CE_W4  (4 bytes)      │
│  [4..12]  Timestamp: Unix epoch ns (8 bytes) │
│  [12..16] Instance ID: FNV-1a hash (4 bytes) │
│  [16..32] Ed25519 signature (16 bytes, truncated) │
└──────────────────────────────────────────────┘
```

- **Magic**: Identifies the watermark within padding noise.
- **Timestamp**: When the stream was produced.
- **Instance ID**: Hash of the server's identity (from ALICE-Auth).
- **Signature**: Ed25519 signature over `[magic || timestamp || instance_id]`,
  truncated to 16 bytes. Not cryptographically secure alone, but sufficient
  for attribution evidence combined with the full signature stored server-side.

### 3.3 Spreading Algorithm

The 32-byte watermark is spread across chunk padding using a PRNG seeded
with a per-stream key:

```
seed = FNV-1a(stream_id || instance_secret)
for each watermark_byte:
    chunk_index = prng.next() % total_chunks
    padding_offset = prng.next() % padding_size
    chunk[chunk_index].padding[padding_offset] = watermark_byte
```

This makes the watermark invisible to casual inspection and resistant to
partial-stream extraction.

### 3.4 Detection Algorithm

Given a suspected ASP stream and the instance secret:

1. Reconstruct the PRNG seed from stream metadata.
2. Extract bytes from predicted padding locations.
3. Check for magic bytes `0xA11CE_W4`.
4. If found, extract timestamp and instance ID.
5. Verify against server-side signature log.

### 3.5 Robustness

| Attack | Survival |
|--------|----------|
| Header stripping | Watermark is in payload padding, not headers |
| Re-encoding | Watermark survives if padding alignment is preserved |
| Truncation | Partial watermark still contains magic + timestamp |
| Random corruption | PRNG spread ensures most bytes survive |
| Full re-mux | Destroyed (acceptable — requires deep protocol knowledge) |

## 4. Implementation Plan

### Phase 1: Padding Infrastructure

- Add configurable chunk alignment (already 64-byte default).
- Reserve padding bytes in chunk serialization.
- Add `watermark` feature flag to Cargo.toml.

### Phase 2: Watermark Embedding

- Implement PRNG-based byte spreading.
- Integrate with ALICE-Auth for Ed25519 signing.
- Add `StreamWriter::with_watermark(instance_secret)` API.

### Phase 3: Detection Tool

- CLI tool: `alice-watermark detect <stream_file> --secret <key>`
- Library API: `watermark::detect(stream_bytes, secret) -> Option<WatermarkInfo>`
- Integration with alicelaw.net for automated violation reporting.

## 5. Privacy Considerations

- The watermark does **not** contain user data or PII.
- The instance ID is a one-way hash, not reversible to the server identity
  without the instance secret.
- The timestamp reveals when the stream was produced, which is already
  present in ASP headers.

## 6. License

This design is part of the ALICE Streaming Protocol (AGPL-3.0).
Implementation will be feature-gated and optional.

---

Reference: https://alicelaw.net
