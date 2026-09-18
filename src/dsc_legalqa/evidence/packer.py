"""Deterministic legal evidence packing for generator context."""

from dsc_legalqa.data.schema import PackedEvidence, RetrievalHit


class EvidencePacker:
    """Formats 1..3 evidence chunks into standard [E1] ... [E2] ... context."""

    def pack(self, query_id: str, hits: list[RetrievalHit]) -> PackedEvidence:
        if not (1 <= len(hits) <= 3):
            raise ValueError(f"EvidencePacker requires 1..3 evidence chunks, got {len(hits)}")

        blocks: list[str] = []
        chunk_ids: list[str] = []

        for idx, hit in enumerate(hits, start=1):
            marker = f"E{idx}"
            chunk_ids.append(hit.chunk_id)
            blocks.append(f"[{marker}] {hit.text.strip()}")

        context_text = "\n\n".join(blocks)
        token_estimate = len(context_text.split())

        return PackedEvidence(
            query_id=query_id,
            chunk_ids=tuple(chunk_ids),
            formatted_context=context_text,
            token_count_estimate=token_estimate,
        )
