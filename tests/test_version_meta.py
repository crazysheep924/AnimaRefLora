from anima_reflora.version_meta import _pkg_version, collect_runtime_versions


def test_collect_runtime_versions_keys():
    versions = collect_runtime_versions(None)
    expected_keys = {
        "python", "torch", "cuda", "cuda_device",
        "sd_scripts_path", "sd_scripts_commit",
        "lycoris", "safetensors", "transformers", "dghs_imgutils",
        "anima_reflora", "docker_image_tag", "runpod_pod_id",
    }
    assert expected_keys == set(versions.keys())


def test_collect_runtime_versions_torch_present():
    versions = collect_runtime_versions(None)
    assert isinstance(versions["python"], str)
    assert isinstance(versions["torch"], str)


def test_pkg_version_known():
    assert isinstance(_pkg_version("torch"), str)


def test_pkg_version_unknown():
    assert _pkg_version("nonexistent-package-xyz-999") is None
