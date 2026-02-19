import tempfile
import unittest
from pathlib import Path

from streamlit_utils import (
    build_train_val_split,
    compute_class_counts,
    parse_label_line,
    write_data_yaml,
    write_split_files,
)


class StreamlitUtilsTests(unittest.TestCase):
    def test_parse_label_line_valid(self):
        parsed = parse_label_line("2 0.5 0.5 0.2 0.2\n")
        self.assertIsNotNone(parsed)
        class_id, bbox = parsed
        self.assertEqual(class_id, 2)
        self.assertEqual(bbox, [0.5, 0.5, 0.2, 0.2])

    def test_parse_label_line_invalid(self):
        self.assertIsNone(parse_label_line("bad-line"))
        self.assertIsNone(parse_label_line("0 only two fields"))

    def test_compute_class_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            label1 = p / "1.txt"
            label2 = p / "2.txt"
            label1.write_text("0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n", encoding="utf-8")
            label2.write_text("1 0.2 0.2 0.1 0.1\n8 0.1 0.1 0.1 0.1\n", encoding="utf-8")

            counts, total = compute_class_counts([label1, label2], num_classes=5)
            self.assertEqual(total, 3)
            self.assertEqual(counts[0], 1)
            self.assertEqual(counts[1], 2)
            self.assertEqual(counts[2], 0)

    def test_build_train_val_split_nonempty(self):
        images = [Path(f"/tmp/{i}.jpg") for i in range(10)]
        train, val = build_train_val_split(images, train_ratio=0.8, seed=123)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 2)
        self.assertEqual(set(train + val), set(images))

    def test_write_split_and_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            train_images = [base / "images" / "1.jpg", base / "images" / "2.jpg"]
            val_images = [base / "images" / "3.jpg"]
            (base / "images").mkdir(parents=True, exist_ok=True)
            for p in train_images + val_images:
                p.write_bytes(b"")

            train_txt, val_txt = write_split_files(base, train_images, val_images)
            self.assertTrue(train_txt.exists())
            self.assertTrue(val_txt.exists())

            data_yaml = base / "data.yaml"
            write_data_yaml(
                target_path=data_yaml,
                base_dir=base,
                train_ref=train_txt,
                val_ref=val_txt,
                class_names=["a", "b"],
            )
            self.assertTrue(data_yaml.exists())
            content = data_yaml.read_text(encoding="utf-8")
            self.assertIn("nc: 2", content)
            self.assertIn("names:", content)


if __name__ == "__main__":
    unittest.main()
