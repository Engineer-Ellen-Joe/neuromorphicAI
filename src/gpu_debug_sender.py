"""
GPU 디버거 연동을 위한 ZeroMQ 메시지 송신 모듈.

이 모듈은 CuPy GPU 배열의 메타데이터와 CUDA IPC 핸들을 추출하여,
독립적으로 실행되는 Vulkan 디버거에 ZeroMQ PUB/SUB 소켓을 통해 전송하는
역할을 합니다.
"""

import zmq
import cupy as cp
import json
from cupy.cuda import runtime

class GpuDebugSender:
    """ZMQ PUB 소켓을 통해 GPU 버퍼 정보를 전송하는 클래스."""

    def __init__(self, port=5555):
        """ZMQ 컨텍스트와 PUB 소켓을 초기화하고 지정된 포트에 바인딩합니다."""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{port}")
            print(f"[GpuDebugSender] ZeroMQ PUB socket bound to port {port}")
        except zmq.ZMQError as e:
            print(f"[GpuDebugSender] Error initializing ZMQ: {e}")
            self.context = None
            self.socket = None

    def send_buffer(self, buffer: cp.ndarray, buffer_name: str):
        """CuPy 배열의 메타데이터와 IPC 핸들을 전송합니다."""
        if not self.socket or not isinstance(buffer, cp.ndarray):
            # print(f"[GpuDebugSender] Skipping send: socket not ready or buffer is not cp.ndarray (isinstance: {isinstance(buffer, cp.ndarray)}, type: {type(buffer)}) ")
            return

        try:
            # print(f"[GpuDebugSender] Debugging buffer '{buffer_name}': type={type(buffer)}, isinstance(cp.ndarray)={isinstance(buffer, cp.ndarray)}, hasattr(get_ipc_handle)={hasattr(buffer, 'get_ipc_handle')}")
            # 1. IPC 핸들 추출 (cupy.cuda.runtime.ipcGetMemHandle 사용)
            ipc_handle = runtime.ipcGetMemHandle(buffer.data.ptr)

            # 2. CUDA 장치 UUID 가져오기
            device_id = buffer.device.id
            device_properties = runtime.getDeviceProperties(device_id)
            # print(f"[GpuDebugSender] Debugging device_properties: type={type(device_properties)}, content={device_properties}") # Debug print removed
            uuid_bytes = device_properties['uuid'][:16] # Access as dictionary key and slice to 16 bytes
            # print(f"[GpuDebugSender] Debugging uuid_bytes: type={type(uuid_bytes)}, len={len(uuid_bytes)}, content={uuid_bytes}") # Debug print removed
            cuda_uuid_str = ''.join([f'{b:02x}' for b in uuid_bytes])

            # 3. 메타데이터 준비 (JSON으로 직렬화)
            metadata = {
                'name': buffer_name,
                'shape': buffer.shape,
                'dtype': str(buffer.dtype),
                'size': buffer.nbytes,
                'cuda_uuid': cuda_uuid_str # CUDA UUID 추가
            }
            metadata_json = json.dumps(metadata).encode('utf-8')
            # print(f"[GpuDebugSender] Sending metadata: {metadata_json.decode('utf-8')}") # Debug print

            # 4. IPC 핸들을 원시 바이트로 변환
            handle_bytes = bytes(ipc_handle)

            # 5. 멀티파트 메시지 전송
            self.socket.send_multipart([
                buffer_name.encode('utf-8'),
                metadata_json,
                handle_bytes
            ])

        except Exception as e:
            print(f"[GpuDebugSender] Error sending buffer '{buffer_name}': {e}")

    def close(self):
        """소켓과 컨텍스트를 안전하게 닫습니다."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("[GpuDebugSender] ZeroMQ context terminated.")

# 다른 모듈에서 쉽게 사용할 수 있도록 전역 인스턴스 생성
sender = GpuDebugSender()
