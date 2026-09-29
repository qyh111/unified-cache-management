# GLM5.3 layout captures

Source: native no-weight runner diagnostics on 2026-09-29, GLM-5.3-Flash
official config, TP1, no MTP; UCM base `476fa42c40dca1eee2e8823c792f2e0dfebcfcf7`.
vLLM source: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
Ascend source: `b64b4d714484feaa6ca71edc99b98318fe6d2f0d`.

CUDA-path allocation ran on CPU using the model-check CPU harness, with the
FP8 linear compute-kernel factory mocked to raise if executed. NPU allocation
ran on 910B3. Neither capture executed forward or loaded checkpoint weights.
These are layout evidence, not numerical inference or store validation.

The JSON retains registered shapes, strides, dtypes, allocation sizes, native
group IDs and descriptors. Absolute device pointers are replaced with view
offsets into each backing storage. Test tensors rebase those offsets onto a
synthetic origin; they do not allocate gigabyte buffers or dereference pointers.
Different captures/storage allocations must not be treated as one allocation
just because the synthetic origin is the same.

Logical block size: 4352 tokens. CUDA has 36 physical blocks and six groups;
NPU has 30 blocks and five groups. Tail is group 1 in both captures. State
group layer counts are 9/9/8/8 and 12/11/11 respectively.

The address oracle uses native tensor row indexing independently of the page
resolver. Policy tests use native State component views on NPU and full declared
pages on CUDA. Production model selection, transient-group handling, ghost row
compilation and end-to-end scheduler/worker integration remain separate work.
