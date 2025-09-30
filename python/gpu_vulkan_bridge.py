# gpu_vulkan_bridge.py
import ctypes
import cupy as cp
import ctypes.wintypes as wintypes

nvcuda = ctypes.CDLL("nvcuda.dll")
kernel32 = ctypes.windll.kernel32

# --- Ctypes Structs & Constants ---
class CUuuid(ctypes.Structure):
    _fields_ = [('bytes', ctypes.c_byte * 16)]

class _CUDA_HANDLE_WIN32(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_wchar_p)]

class _CUDA_HANDLE_NAMED(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p)]

class _CUDA_HANDLE_UNION(ctypes.Union):
    _fields_ = [("win32Handle", _CUDA_HANDLE_WIN32), ("win32NamedHandle", _CUDA_HANDLE_NAMED)]

class CUDA_EXTERNAL_MEMORY_HANDLE_DESC(ctypes.Structure):
    _fields_ = [("handle", _CUDA_HANDLE_UNION), ("size", ctypes.c_ulonglong), ("flags", ctypes.c_uint), ("type", ctypes.c_int)]

class CUDA_EXTERNAL_MEMORY_BUFFER_DESC(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_ulonglong), ("size", ctypes.c_ulonglong), ("flags", ctypes.c_uint)]

CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2
CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
CUDA_EXTERNAL_MEMORY_DEDICATED = 0x1

# --- Driver Prototypes ---
nvcuda.cuInit.argtypes = [ctypes.c_uint]; nvcuda.cuInit.restype = ctypes.c_int
nvcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]; nvcuda.cuDeviceGetCount.restype = ctypes.c_int
nvcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]; nvcuda.cuDeviceGet.restype = ctypes.c_int
nvcuda.cuDeviceGetUuid.argtypes = [ctypes.POINTER(CUuuid), ctypes.c_int]; nvcuda.cuDeviceGetUuid.restype = ctypes.c_int
nvcuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]; nvcuda.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
nvcuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]; nvcuda.cuCtxSetCurrent.restype = ctypes.c_int
nvcuda.cuImportExternalMemory.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUDA_EXTERNAL_MEMORY_HANDLE_DESC)]; nvcuda.cuImportExternalMemory.restype = ctypes.c_int
nvcuda.cuExternalMemoryGetMappedBuffer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(CUDA_EXTERNAL_MEMORY_BUFFER_DESC)]; nvcuda.cuExternalMemoryGetMappedBuffer.restype = ctypes.c_int
nvcuda.cuDestroyExternalMemory.argtypes = [ctypes.c_void_p]; nvcuda.cuDestroyExternalMemory.restype = ctypes.c_int

_initialized_cuda_device = -1

def init_cuda_for_vulkan_device(vulkan_uuid_str):
    global _initialized_cuda_device
    if _initialized_cuda_device != -1:
        return # Already initialized

    rc = nvcuda.cuInit(0)
    if rc != 0: raise RuntimeError(f"cuInit failed: {rc}")

    # Convert Vulkan UUID string to bytes
    vulkan_uuid_bytes = bytes.fromhex(vulkan_uuid_str)

    # Find matching CUDA device by UUID
    device_count = ctypes.c_int()
    nvcuda.cuDeviceGetCount(ctypes.byref(device_count))
    
    matching_device = -1
    for i in range(device_count.value):
        cuda_uuid = CUuuid()
        nvcuda.cuDeviceGetUuid(ctypes.byref(cuda_uuid), i)
        cuda_uuid_bytes = bytes(cuda_uuid.bytes)
        
        if cuda_uuid_bytes == vulkan_uuid_bytes:
            matching_device = i
            break

    if matching_device == -1:
        raise RuntimeError("Could not find a CUDA device matching the Vulkan device UUID.")

    print(f"[Bridge] Found matching CUDA device at index {matching_device}")
    _initialized_cuda_device = matching_device

    # Get and set the primary context for the matched device
    ctx = ctypes.c_void_p()
    rc = nvcuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), matching_device)
    if rc != 0: raise RuntimeError(f"cuDevicePrimaryCtxRetain failed for device {matching_device}: {rc}")
    
    rc = nvcuda.cuCtxSetCurrent(ctx)
    if rc != 0: raise RuntimeError(f"cuCtxSetCurrent failed: {rc}")

    print(f"[Bridge] Successfully set primary context for CUDA device {matching_device}")

def import_and_map(exported_info):
    handle_val = int(exported_info.handle)
    alloc_size = int(exported_info.allocation_size)
    print(f"[bridge] import_and_map: handle=0x{handle_val:x}, alloc_size={alloc_size}, alignment={exported_info.alignment}, handle_type={exported_info.handle_type}")

    # check handle validity in-process
    flags = ctypes.c_uint(0)
    ok = kernel32.GetHandleInformation(ctypes.c_void_p(handle_val), ctypes.byref(flags))
    print(f"[bridge] GetHandleInformation => ok={ok}, flags={flags.value}, WinErr={kernel32.GetLastError()}")

    mem_desc = CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
    mem_desc.handle.win32Handle.handle = ctypes.c_void_p(handle_val)
    mem_desc.handle.win32Handle.name = None
    mem_desc.size = ctypes.c_ulonglong(alloc_size)

    tried = []
    ext_mem = ctypes.c_void_p()
    # combinations: types × flags
    types = [CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32, CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT]
    flags_list = [0, CUDA_EXTERNAL_MEMORY_DEDICATED]

    last_err = None
    for flags_val in flags_list:
        for t in types:
            mem_desc.flags = ctypes.c_uint(flags_val)
            mem_desc.type = t
            tried.append((t, flags_val))
            kernel32.SetLastError(0)
            res = nvcuda.cuImportExternalMemory(ctypes.byref(ext_mem), ctypes.byref(mem_desc))
            winerr = kernel32.GetLastError()
            print(f"[bridge] try type={t} flags={flags_val} => cuImportExternalMemory res={res}, WinErr={winerr}")
            if res == 0:
                print(f"[bridge] cuImportExternalMemory succeeded with type={t} flags={flags_val}")
                last_err = None
                break
            last_err = (res, winerr)
        if last_err is None:
            break

    if last_err is not None:
        # try local DuplicateHandle as last resort
        dup = ctypes.c_void_p()
        okdup = kernel32.DuplicateHandle(kernel32.GetCurrentProcess(),
                                         ctypes.c_void_p(handle_val),
                                         kernel32.GetCurrentProcess(),
                                         ctypes.byref(dup),
                                         0,
                                         False,
                                         2)  # DUPLICATE_SAME_ACCESS
        print(f"[bridge] Local DuplicateHandle => ok={okdup}, dup={dup.value if dup else None}, WinErr={kernel32.GetLastError()}")
        if okdup and dup.value:
            mem_desc.handle.win32Handle.handle = ctypes.c_void_p(dup.value)
            mem_desc.type = CUDA_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
            mem_desc.flags = ctypes.c_uint(0)
            kernel32.SetLastError(0)
            res_dup = nvcuda.cuImportExternalMemory(ctypes.byref(ext_mem), ctypes.byref(mem_desc))
            print(f"[bridge] cuImportExternalMemory with local dup => res={res_dup}, WinErr={kernel32.GetLastError()}")
            if res_dup == 0:
                last_err = None
            else:
                last_err = (res_dup, kernel32.GetLastError())

    if last_err is not None:
        # cleanup: tell vkdbg to destroy export buffer (so Vulkan won't warn on device destroy)
        try:
            import vkdbg
            if hasattr(exported_info, "internal_id"):
                try:
                    vkdbg.destroy_exportable_buffer(exported_info.internal_id)
                    print("[bridge] requested vkdbg.destroy_exportable_buffer on failure")
                except Exception as e:
                    print("[bridge] failed to call vkdbg.destroy_exportable_buffer:", e)
        except Exception:
            pass

        raise RuntimeError(f"cuImportExternalMemory failed after trying {tried}; last={last_err}")

    # map buffer
    buf_desc = CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
    buf_desc.offset = ctypes.c_ulonglong(0)
    buf_desc.size = ctypes.c_ulonglong(alloc_size)
    buf_desc.flags = ctypes.c_uint(0)

    dev_ptr = ctypes.c_void_p()
    r2 = nvcuda.cuExternalMemoryGetMappedBuffer(ctypes.byref(dev_ptr), ext_mem, ctypes.byref(buf_desc))
    print(f"[bridge] cuExternalMemoryGetMappedBuffer => {r2}, WinErr={kernel32.GetLastError()}")
    if r2 != 0:
        nvcuda.cuDestroyExternalMemory(ext_mem)
        raise RuntimeError(f"cuExternalMemoryGetMappedBuffer failed: {r2}")

    unowned = cp.cuda.UnownedMemory(dev_ptr.value, alloc_size, None)
    ptr = cp.cuda.MemoryPointer(unowned, 0)
    arr = cp.ndarray((alloc_size,), dtype=cp.uint8, memptr=ptr)
    return arr, ext_mem

def destroy_mapped(ext_mem):
    if ext_mem:
        nvcuda.cuDestroyExternalMemory(ext_mem)