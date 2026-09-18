"""Test evidence packing and prompt assembly."""

from dsc_legalqa.data.schema import RetrievalHit
from dsc_legalqa.evidence.packer import EvidencePacker
from dsc_legalqa.generation.prompt import build_generation_prompt, SYSTEM_INSTRUCTION_VIETNAMESE


def test_evidence_packer():
    packer = EvidencePacker()
    hits = [
        RetrievalHit("c1", "d1", "12", "Noi dung dieu 12", 0.9, 1, "HYBRID"),
        RetrievalHit("c2", "d1", "13", "Noi dung dieu 13", 0.8, 2, "HYBRID"),
    ]
    packed = packer.pack("Q001", hits)
    assert packed.query_id == "Q001"
    assert packed.chunk_ids == ("c1", "c2")
    assert "[E1] Noi dung dieu 12" in packed.formatted_context
    assert "[E2] Noi dung dieu 13" in packed.formatted_context


def test_generation_prompt_structure():
    prompt = build_generation_prompt("Cau hoi test?", "[E1] Can cu test")
    assert "[ANSWER_CONTROL]" in prompt
    assert "[/ANSWER_CONTROL]" in prompt
    assert "Căn cứ pháp lý:" in prompt
    assert "Câu hỏi:" in prompt
    assert "Câu trả lời:" in prompt
    assert "[E1] Can cu test" in prompt
