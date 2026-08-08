import asyncio

from app.core.subtitle_hub import SubtitleHub


def test_preview_events_are_broadcast_but_not_remembered():
    hub = SubtitleHub(max_history=80)

    async def _run():
        queue = await hub.subscribe()
        await hub.publish({
            "type": "original_preview",
            "segment_id": 101,
            "text": "draft",
        })
        delivered = queue.get_nowait()
        preview_snapshot = await hub.snapshot()
        await hub.publish({
            "type": "original",
            "segment_id": 101,
            "text": "final",
        })
        final_snapshot = await hub.snapshot()
        return delivered, preview_snapshot, final_snapshot

    delivered, preview_snapshot, final_snapshot = asyncio.run(_run())

    assert delivered["type"] == "original_preview"
    assert preview_snapshot["segments"] == []
    assert final_snapshot["segments"][0]["original_text"] == "final"
