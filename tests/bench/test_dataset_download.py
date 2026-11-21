import autotunenet.bench.tasks.data as ds


class _FakeDataset:
    """Behaves like a torchvision dataset: missing until downloaded."""
    downloaded = False

    def __init__(self, root, train, download, transform=None):
        if download:
            type(self).downloaded = True
        if not type(self).downloaded:
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        self.root = root


def test_missing_dataset_is_downloaded_with_a_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ds, "DATA_DIR", tmp_path / "data")
    monkeypatch.setitem(ds.DATASET_INFO, _FakeDataset, ("FakeSet", "about 1 MB", "a test server"))
    _FakeDataset.downloaded = False

    dataset = ds.load_dataset(_FakeDataset, train=True)

    out = capsys.readouterr().out
    assert "FakeSet (training split) not found" in out
    assert "Downloading FakeSet (about 1 MB) from a test server" in out
    assert "FakeSet ready" in out
    assert dataset.root == str(tmp_path / "data")


def test_present_dataset_is_reused_silently(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ds, "DATA_DIR", tmp_path)
    _FakeDataset.downloaded = True

    ds.load_dataset(_FakeDataset, train=False)

    assert capsys.readouterr().out == ""


def test_data_dir_does_not_depend_on_the_working_directory():
    assert ds.DATA_DIR.is_absolute()
