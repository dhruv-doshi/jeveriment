from PIL import Image

from jev_eval.io import write_json
from jev_eval.reporting import plots


def test_render_only_saved_measurements(tmp_path):
    assert plots(tmp_path) == []
    write_json(
        tmp_path / "summary.json",
        {
            "fixture": {
                "ndcg@10": 0.5,
                "recall@10": 1.0,
                "mrr@10": 0.5,
                "judged@10": 0.2,
            }
        },
    )
    rendered = plots(tmp_path)
    assert len(rendered) == 4
    for path in rendered:
        with Image.open(path) as image:
            assert image.size == (1200, 600)
