Protocol derived from Improbable-AI/VisionProTeleop at 4c549905c2a8b214d79f7cd88e535101a1ce32af (MIT).

Fields 4–12 provide R1 tracking validity and original anchor timestamps. Fields 13–23 add versioned diagnostics: raw head validity, video requirement/readiness/reason, source loss counters, local write wait and per-RPC packet count. The receiver checks `diagnostics_version == 1` before interpreting these fields; omitted fields from older senders must not be treated as a new tracking failure.

Loss counters preserve source invalidation events between transmitted samples. Stream identity resets their baseline after reconnect. `last_write_ms` measures local gRPC write submission wait, not RTT or delivery acknowledgement; `packets_sent` belongs to the current RPC stream. Head/video failure causes remain latched in operator status until explicit realignment succeeds. Anchor timestamps remain unchanged when packets are coalesced.

Regenerate handtracking_pb2.py using grpcio-tools==1.78.0:

```sh
python -m grpc_tools.protoc -I tools/visionpro_protocol --python_out=tools/visionpro_protocol tools/visionpro_protocol/handtracking.proto
```

The matching native source and Swift bindings are in `lizuju/VisionProTeleop`, tag `production-20260928-visionpro`.
