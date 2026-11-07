"""Check compare_quant_kernels Rust fixtures against a pinned GGML CPU library.

Usage: python compare_quant_kernels.py MODEL FIXTURES GGML_CPU_LIBRARY OUTPUT
Calls GGML's own activation quantizers and dot products through ctypes.
"""
import ctypes as C
import json
import mmap
import struct
import sys
from pathlib import Path


def main():
    model, fixture_path, library, output = map(Path, sys.argv[1:])
    fixtures = json.loads(fixture_path.read_text())
    lib = C.CDLL(str(library.resolve()))
    lib.ggml_backend_cpu_init.restype = C.c_void_p
    backend = lib.ggml_backend_cpu_init()
    lib.ggml_backend_free.argtypes = [C.c_void_p]
    results = []
    with model.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as weights:
        for fixture in fixtures:
            cols = fixture['columns']
            dtype = fixture['dtype']
            is_k = dtype == 18
            block_size, block_bytes = (256, 292) if is_k else (32, 34)
            x = (C.c_float * cols)(*fixture['input'])
            quantized = C.create_string_buffer(cols // block_size * block_bytes)
            quantize = getattr(lib, 'quantize_row_q8_K' if is_k else 'quantize_row_q8_0')
            quantize.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
            quantize(x, quantized, cols)
            qbytes = quantized.raw
            scales, qs = [], []
            for b in range(cols // block_size):
                base = b * block_bytes
                scales.append(struct.unpack_from('<f' if is_k else '<e', qbytes, base)[0])
                qs.extend(struct.unpack_from(f'{block_size}b', qbytes, base + (4 if is_k else 2)))
            scale_errors = [abs(a-b) for a,b in zip(scales, fixture['scales'])]
            byte_mismatches = sum(a != b for a,b in zip(qs, fixture['quantized']))
            dot_name = {8: 'ggml_vec_dot_q8_0_q8_0', 18: 'ggml_vec_dot_iq3_xxs_q8_K', 20: 'ggml_vec_dot_iq4_nl_q8_0'}[dtype]
            dot = getattr(lib, dot_name)
            dot.argtypes = [C.c_int, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int]
            reference = []
            for row in fixture['rows']:
                start = fixture['data_offset'] + row * fixture['row_bytes']
                row_data = C.create_string_buffer(weights[start:start + fixture['row_bytes']])
                result = C.c_float()
                dot(cols, C.byref(result), 0, row_data, 0, quantized, 0, 1)
                reference.append(result.value)
            errors = [abs(a-b) for a,b in zip(reference, fixture['native'])]
            dedicated_errors = [abs(a-b) for a,b in zip(reference, fixture['dedicated'])]
            results.append({'tensor': fixture['tensor'], 'variant': fixture['variant'], 'dtype': dtype,
                            'activation_byte_mismatches': byte_mismatches,
                            'max_activation_scale_error': max(scale_errors),
                            'reference': reference, 'native': fixture['native'],
                            'max_dot_absolute_error': max(errors),
                            'max_dedicated_dot_absolute_error': max(dedicated_errors, default=0),
                            'within_tolerance': all(e <= 5e-5 * max(1, abs(r)) for e,r in zip(errors, reference))
                            and all(e <= 5e-5 * max(1, abs(r)) for e,r in zip(dedicated_errors, reference))})
    lib.ggml_backend_free(backend)
    report = {'method': 'Actual model rows, three deterministic activation vectors, GGML CPU quantizers and dot products. No model fine-tuning involved.',
              'fixtures': len(results), 'dot_products': sum(len(r['reference']) for r in results),
              'all_within_tolerance': all(r['within_tolerance'] for r in results),
              'max_dot_absolute_error': max(r['max_dot_absolute_error'] for r in results),
              'activation_byte_mismatches': sum(r['activation_byte_mismatches'] for r in results),
              'max_activation_scale_error': max(r['max_activation_scale_error'] for r in results),
              'results': results}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
