from vox_agent.audio import (
    JitterBuffer,
    VoiceActivityDetector,
    downsample,
    mulaw_decode,
    mulaw_encode,
    rms_energy,
    synth_utterance,
)
from vox_agent.audio.pcm import pcm_bytes_to_ints


def test_synth_has_energy():
    pcm = synth_utterance("hello there friend", 16000)
    assert len(pcm) > 0
    loud = any(rms_energy(pcm[i : i + 640]) > 500 for i in range(0, len(pcm) - 640, 640))
    assert loud


def test_vad_detects_speech_and_endpoints():
    sr = 16000
    fb = int(sr * 0.02) * 2
    pcm = synth_utterance("book an appointment please", sr)
    # append 600ms trailing silence so the endpoint fires
    pcm += b"\x00" * (fb * 30)
    vad = VoiceActivityDetector(threshold=500, frame_ms=20, end_silence_ms=480)
    starts = ends = 0
    for i in range(0, len(pcm) - fb, fb):
        ev = vad.process(pcm[i : i + fb])
        starts += ev.kind == "speech_start"
        ends += ev.kind == "speech_end"
    assert starts >= 1
    assert ends >= 1


def test_mulaw_roundtrip_preserves_shape():
    pcm = synth_utterance("test tone", 8000)
    enc = mulaw_encode(pcm)
    dec = mulaw_decode(enc)
    assert len(enc) == len(pcm) // 2
    assert len(dec) == len(enc) * 2
    assert any(abs(x) > 0 for x in pcm_bytes_to_ints(dec))


def test_downsample_shrinks():
    pcm = synth_utterance("hi there", 16000)
    ds = downsample(pcm, 16000, 8000)
    assert 0 < len(ds) < len(pcm)


def test_jitter_buffer_reorders():
    jb = JitterBuffer(frame_ms=20, target_ms=40, max_ms=200)
    jb.push(1, b"b")
    jb.push(0, b"a")
    out = []
    for _ in range(6):
        x = jb.pop()
        if x is not None:
            out.append(x)
    assert out[:2] == [b"a", b"b"]


def test_jitter_buffer_drops_late_packet():
    jb = JitterBuffer(frame_ms=20, target_ms=20, max_ms=100)
    jb.push(0, b"a")
    jb.pop()  # plays seq 0
    jb.push(0, b"late")  # arrives after we've moved on
    assert jb.dropped == 1
