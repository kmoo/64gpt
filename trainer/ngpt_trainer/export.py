import struct
import numpy as np
from ngpt_trainer.vocab import Vocab
from ngpt_trainer.quantize import QuantizedGRU

MAGIC = b'NGPT'
FORMAT_VERSION = 1
FORMAT_VERSION_TRIE = 2  # M12.1 phase 4: GRU payload + a trailing word-trie
                         # section (docs/ideas-coherence-rescue-plan.md fix
                         # 4). A NEW version, not a change to version 1's
                         # exact-size formula, because every pre-M12.1
                         # vector file (m2_gru.bin..m12_1_gru.bin from
                         # phases 2-3) is version 1 and must keep parsing
                         # byte-for-byte unchanged -- see core/ngpt_gru.cpp.
MODEL_TYPE_GRU = 1

def payload(q: QuantizedGRU, vocab: Vocab, trie_nodes=None) -> bytes:
    H, V = q.H, q.V
    payload = struct.pack('>HHBB', H, V, q.k_w, q.k_out)
    payload += vocab.to_bytes()
    payload += q.lut_sigmoid.astype('>i2').tobytes()
    payload += q.lut_tanh.astype('>i2').tobytes()
    payload += q.W_ih.astype('>i1').tobytes()
    payload += q.W_hh.astype('>i1').tobytes()
    payload += q.b_ih.astype('>i4').tobytes()
    payload += q.b_hh.astype('>i4').tobytes()
    payload += q.W_out.astype('>i1').tobytes()
    payload += q.b_out.astype('>i4').tobytes()
    if trie_nodes is not None:
        payload += struct.pack('>I', len(trie_nodes))
        for char, flags, first_child, next_sibling in trie_nodes:
            payload += struct.pack('>BBHH', char, flags, first_child, next_sibling)
    return payload

def build_blob(q: QuantizedGRU, vocab: Vocab, trie_nodes=None) -> bytes:
    p = payload(q, vocab, trie_nodes)
    version = FORMAT_VERSION_TRIE if trie_nodes is not None else FORMAT_VERSION
    return MAGIC + struct.pack('>HHI', version, MODEL_TYPE_GRU, len(p)) + p

def parse_blob(data: bytes) -> (QuantizedGRU, Vocab):
    if data[:4] != MAGIC:
        raise ValueError("Invalid magic number")
    version, model_type, payload_length = struct.unpack('>HHI', data[4:12])
    if version != FORMAT_VERSION or model_type != MODEL_TYPE_GRU:
        raise ValueError("Unsupported version or model type")
    if len(data) != 12 + payload_length:
        raise ValueError("Payload length mismatch")
    
    payload = data[12:]
    H, V, k_w, k_out = struct.unpack('>HHBB', payload[:6])
    vocab = Vocab.from_bytes(payload[6:6+V])
    offset = 6 + V

    lut_sigmoid = np.frombuffer(payload[offset:offset+512], dtype='>i2').astype(np.int16)
    offset += 512

    lut_tanh = np.frombuffer(payload[offset:offset+512], dtype='>i2').astype(np.int16)
    offset += 512

    W_ih = np.frombuffer(payload[offset:offset+3*H*V], dtype=np.int8).reshape((3*H, V))
    offset += 3*H*V

    W_hh = np.frombuffer(payload[offset:offset+3*H*H], dtype=np.int8).reshape((3*H, H))
    offset += 3*H*H

    b_ih = np.frombuffer(payload[offset:offset+3*H*4], dtype='>i4').astype(np.int32)
    offset += 3*H*4

    b_hh = np.frombuffer(payload[offset:offset+3*H*4], dtype='>i4').astype(np.int32)
    offset += 3*H*4

    W_out = np.frombuffer(payload[offset:offset+V*H], dtype=np.int8).reshape((V, H))
    offset += V*H

    b_out = np.frombuffer(payload[offset:offset+V*4], dtype='>i4').astype(np.int32)
    
    q = QuantizedGRU(H=H, V=V, k_w=k_w, k_out=k_out, W_ih=W_ih, W_hh=W_hh, b_ih=b_ih, b_hh=b_hh, W_out=W_out, b_out=b_out, lut_sigmoid=lut_sigmoid, lut_tanh=lut_tanh)
    return q, vocab

def trace_bytes(steps, H) -> bytes:
    payload = struct.pack('>I', len(steps))
    for input_id, h, argmax_id in steps:  # matches ref_impl.trace() order
        payload += struct.pack('>HH', input_id, argmax_id)
        payload += h.astype('>i2').tobytes()
    return payload
