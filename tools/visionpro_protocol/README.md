Protocol derived from Improbable-AI/VisionProTeleop at 4c549905c2a8b214d79f7cd88e535101a1ce32af (MIT). Fields 4–12 provide R1 tracking validity and timestamps.

Regenerate handtracking_pb2.py using grpcio-tools==1.78.0:

```sh
python -m grpc_tools.protoc -I tools/visionpro_protocol --python_out=tools/visionpro_protocol tools/visionpro_protocol/handtracking.proto
```
