"""
Config 单元测试 —— 重点测 data_dir 解析逻辑

resolve_data_dir() + Config.resolve() 是纯逻辑（+ 文件系统副作用），
使用 tmp_path 隔离测试。

注意：load_config 测试都传显式不存在的 path，避免被项目根 config.yaml 干扰。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from paper_review.config import Config, resolve_data_dir

# 显式不存在的 config 路径，避免被项目根的 config.yaml 干扰
_EXPLICIT_NONE_PATH = "/nonexistent/.paper-review/config.yaml"


class TestResolveDataDir:
    """resolve_data_dir() 的行为"""

    def test_explicit_path_returns_as_is(self, tmp_path):
        """显式 data_dir 直接返回。"""
        d = tmp_path / "my-data"
        d.mkdir(parents=True)
        result = resolve_data_dir(str(d))
        assert result == d.resolve()

    def test_dot_paper_review_exists_uses_it(self, tmp_path):
        """./.paper-review/ 存在时使用它。"""
        dot = tmp_path / ".paper-review"
        dot.mkdir(parents=True)
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = resolve_data_dir()
        assert result == dot.resolve()

    def test_dot_paper_review_not_exists_falls_back(self, tmp_path):
        """./.paper-review/ 不存在时 fallback 到 ~/.paper-review/ 并自动创建。"""
        home = tmp_path / "home"
        with patch("pathlib.Path.home", return_value=home):
            with patch("pathlib.Path.cwd", return_value=tmp_path / "some-project"):
                result = resolve_data_dir()
        expected = home / ".paper-review"
        assert result == expected
        assert expected.exists()  # 自动创建

    def test_fallback_creates_parents(self, tmp_path):
        """fallback 路径自动创建目录（包含中间目录）。"""
        home = tmp_path / "deep" / "home"
        with patch("pathlib.Path.home", return_value=home):
            result = resolve_data_dir()
        assert result == home / ".paper-review"
        assert result.exists()

    def test_explicit_overrides_all(self, tmp_path):
        """显式 data_dir 覆盖一切，即使 .paper-review 存在。"""
        dot = tmp_path / ".paper-review"
        dot.mkdir(parents=True)
        explicit = tmp_path / "explicit"
        explicit.mkdir(parents=True)
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = resolve_data_dir(str(explicit))
        assert result == explicit.resolve()
        assert result != dot.resolve()


class TestConfigResolve:
    """Config.resolve() 的行为"""

    def test_empty_paths_get_derived_from_data_dir(self, tmp_path):
        """空的 index_dir/pdf_dir 从 data_dir 自动推导。"""
        dd = tmp_path / "data"
        dd.mkdir(parents=True)
        cfg = Config(index_dir="", pdf_dir="")
        resolved = cfg.resolve(data_dir_override=str(dd))
        assert resolved.index_dir == str(dd / "index")
        assert resolved.pdf_dir == str(dd / "pdfs")

    def test_explicit_paths_survive_resolve(self, tmp_path):
        """显式设置的 index_dir/pdf_dir 不被覆盖。"""
        dd = tmp_path / "data"
        dd.mkdir(parents=True)
        cfg = Config(index_dir="/custom/index", pdf_dir="/custom/pdfs")
        resolved = cfg.resolve(data_dir_override=str(dd))
        assert resolved.index_dir == "/custom/index"
        assert resolved.pdf_dir == "/custom/pdfs"

    def test_resolve_returns_new_instance(self, tmp_path):
        """resolve() 不修改原对象。"""
        dd = tmp_path / "data"
        dd.mkdir(parents=True)
        cfg = Config(index_dir="", pdf_dir="")
        resolved = cfg.resolve(data_dir_override=str(dd))
        assert cfg is not resolved  # 新实例
        assert cfg.index_dir == ""  # 原对象不变

    def test_model_cache_stays_independent(self, tmp_path):
        """model_cache_dir 不受 data_dir 影响。"""
        dd = tmp_path / "data"
        dd.mkdir(parents=True)
        cfg = Config(model_cache_dir="/custom/cache/models")
        resolved = cfg.resolve(data_dir_override=str(dd))
        assert resolved.model_cache_dir == "/custom/cache/models"

    def test_resolve_auto_data_dir(self, tmp_path):
        """data_dir_override 为 None 时自动解析 data_dir。"""
        dot = tmp_path / ".paper-review"
        dot.mkdir(parents=True)
        cfg = Config(index_dir="", pdf_dir="")
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            resolved = cfg.resolve()
        assert resolved.index_dir == str(dot / "index")
        assert resolved.pdf_dir == str(dot / "pdfs")


class TestLoadConfigDataDir:
    """load_config() 的 data_dir 参数"""

    def test_load_config_with_data_dir(self, tmp_path):
        """load_config(data_dir=...) 传入 resolve_data_dir。"""
        from paper_review.config import load_config

        dd = tmp_path / "data"
        dd.mkdir(parents=True)
        cfg = load_config(path=_EXPLICIT_NONE_PATH, data_dir=str(dd))
        assert cfg.index_dir == str(dd / "index")
        assert cfg.pdf_dir == str(dd / "pdfs")
        assert cfg.model_cache_dir == str(Path.home() / ".cache" / "paper-review" / "models")

    def test_load_config_no_data_dir_auto(self, tmp_path):
        """load_config() 无 data_dir 时自动解析。"""
        from paper_review.config import load_config

        dot = tmp_path / ".paper-review"
        dot.mkdir(parents=True)
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            cfg = load_config(path=_EXPLICIT_NONE_PATH)
        assert cfg.index_dir == str(dot / "index")
        assert cfg.pdf_dir == str(dot / "pdfs")

    def test_load_config_fallback_auto_create(self, tmp_path):
        """load_config() fallback 到 ~/.paper-review/ 时自动创建。"""
        from paper_review.config import load_config

        home = tmp_path / "home"
        cwd = tmp_path / "project"
        cwd.mkdir(parents=True)
        with patch("pathlib.Path.home", return_value=home):
            with patch("pathlib.Path.cwd", return_value=cwd):
                cfg = load_config(path=_EXPLICIT_NONE_PATH)
        expected_ix = str(home / ".paper-review" / "index")
        assert cfg.index_dir == expected_ix
        assert (home / ".paper-review").exists()

    def test_load_config_yaml_overrides_data_dir_derived(self, tmp_path):
        """Config YAML 中的显式 index_dir 不被 data_dir 覆盖。"""
        from paper_review.config import load_config

        dd = tmp_path / "data"
        dd.mkdir(parents=True)
        # 在 {data_dir}/ 下放一个 config.yaml
        yaml_path = dd / "config.yaml"
        yaml_path.write_text("index_dir: /custom/from/yaml\n")
        cfg = load_config(path=str(yaml_path), data_dir=str(dd))
        assert cfg.index_dir == "/custom/from/yaml"
        # pdf_dir 空 → 自动推导
        assert cfg.pdf_dir == str(dd / "pdfs")

    def test_load_config_yaml_data_dir_field_sets_derived_paths(self, tmp_path):
        """YAML 中 data_dir 字段被 load_config() 读取，自动推导子路径。"""
        from paper_review.config import load_config

        dd = tmp_path / "yaml-data-dir"
        dd.mkdir(parents=True)
        yaml_path = tmp_path / "my.yaml"
        yaml_path.write_text(f"data_dir: {dd}\n")
        cfg = load_config(path=str(yaml_path))
        # index_dir/pdf_dir 应从 YAML 的 data_dir 推导
        assert cfg.index_dir == str(dd / "index")
        assert cfg.pdf_dir == str(dd / "pdfs")

    def test_load_config_yaml_data_dir_with_explicit_index(self, tmp_path):
        """YAML 中同时指定 data_dir 和 index_dir，index_dir 优先。"""
        from paper_review.config import load_config

        dd = tmp_path / "base"
        dd.mkdir(parents=True)
        yaml_path = tmp_path / "mix.yaml"
        yaml_path.write_text(f"data_dir: {dd}\nindex_dir: /custom/index\n")
        cfg = load_config(path=str(yaml_path))
        assert cfg.index_dir == "/custom/index"
        # pdf_dir 应来自 data_dir 推导
        assert cfg.pdf_dir == str(dd / "pdfs")
