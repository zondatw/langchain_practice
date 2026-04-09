import importlib
import sys
import types
import unittest
from unittest import mock

from project_helper.settings import VectorDb


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class EntryPointSmokeTest(unittest.TestCase):
    def tearDown(self):
        for module_name in ("main", "web", "project_helper.__main__"):
            sys.modules.pop(module_name, None)

    def test_root_main_delegates_to_package_cli(self):
        cli_main = mock.Mock()

        with mock.patch.dict(sys.modules, {"project_helper.cli": _stub_module("project_helper.cli", main=cli_main)}):
            module = importlib.import_module("main")
            module.main()

        cli_main.assert_called_once_with()

    def test_package_main_delegates_to_package_cli(self):
        cli_main = mock.Mock()

        with mock.patch.dict(sys.modules, {"project_helper.cli": _stub_module("project_helper.cli", main=cli_main)}):
            module = importlib.import_module("project_helper.__main__")
            module.main()

        cli_main.assert_called_once_with()

    def test_web_main_uses_module_runtime_constants(self):
        assistant_instance = mock.Mock()
        assistant_class = mock.Mock(return_value=assistant_instance)
        instrument_assistant = mock.Mock()
        metrics_server = types.SimpleNamespace(start=mock.Mock())
        configure_logging = mock.Mock()
        settings = types.SimpleNamespace(
            project_path="/tmp/demo-project",
            runtime=mock.sentinel.runtime,
            qdrant=mock.sentinel.qdrant,
            zhtw_mcp=mock.sentinel.zhtw_mcp,
        )
        load_settings = mock.Mock(return_value=settings)
        app = mock.Mock()

        stub_modules = {
            "gradio": _stub_module("gradio"),
            "project_helper.assistant": _stub_module("project_helper.assistant", RustProjectAssistant=assistant_class),
            "project_helper.metrics": _stub_module(
                "project_helper.metrics",
                MetricsServer=metrics_server,
                instrument_assistant=instrument_assistant,
            ),
            "project_helper.settings": _stub_module(
                "project_helper.settings",
                VectorDb=VectorDb,
                load_settings=load_settings,
            ),
            "project_helper.logging_utils": _stub_module(
                "project_helper.logging_utils",
                configure_logging=configure_logging,
            ),
        }

        with mock.patch.dict(sys.modules, stub_modules):
            module = importlib.import_module("web")
            with mock.patch.object(module, "create_gr", return_value=app) as create_gr:
                with mock.patch("builtins.print"):
                    module.main()

        configure_logging.assert_called_once_with()
        load_settings.assert_called_once_with()
        assistant_class.assert_called_once_with(
            project_path="/tmp/demo-project",
            runtime_settings=mock.sentinel.runtime,
            qdrant_settings=mock.sentinel.qdrant,
            zhtw_mcp_settings=mock.sentinel.zhtw_mcp,
        )
        instrument_assistant.assert_called_once_with(assistant_instance)
        metrics_server.start.assert_called_once_with(port=9090, host="0.0.0.0")
        create_gr.assert_called_once_with(assistant=assistant_instance)
        app.launch.assert_called_once_with(server_name="0.0.0.0", server_port=7860, theme="soft")


if __name__ == "__main__":
    unittest.main()
