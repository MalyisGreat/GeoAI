from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _purge_package(package_name: str) -> None:
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)


def _load_package_from_init(module_name: str, package_root: Path) -> None:
    init_path = (package_root / "__init__.py").resolve()
    if not init_path.exists():
        raise RuntimeError(f"Cannot bootstrap {module_name!r}: missing {init_path}")
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(package_root.resolve())],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create import spec for {module_name!r} from {init_path}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(init_path)
    module.__package__ = module_name
    module.__path__ = [str(package_root.resolve())]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def bootstrap_project_src(
    project_root: Path,
    package_name: str = "geobot",
    required_subpackages: tuple[str, ...] = ("data", "eval", "model", "train", "utils"),
) -> Path:
    project_root = project_root.resolve()
    src_root = (project_root / "src").resolve()
    package_root = (src_root / package_name).resolve()
    init_path = package_root / "__init__.py"
    if not init_path.exists():
        raise RuntimeError(f"Cannot bootstrap {package_name!r}: missing {init_path}")

    src_entry = str(src_root)
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != src_root]
    sys.path.insert(0, src_entry)

    loaded = sys.modules.get(package_name)
    loaded_path = None
    if loaded is not None:
        module_file = getattr(loaded, "__file__", None)
        if module_file:
            loaded_path = Path(module_file).resolve()
    if loaded_path != init_path.resolve():
        _purge_package(package_name)
        importlib.invalidate_caches()
        _load_package_from_init(package_name, package_root)

    imported = sys.modules.get(package_name)
    imported_file = getattr(imported, "__file__", None)
    if imported_file is None or Path(imported_file).resolve() != init_path.resolve():
        raise RuntimeError(f"Failed to resolve local {package_name!r} package from {init_path}")
    imported_path = [Path(entry).resolve() for entry in getattr(imported, "__path__", [])]
    if package_root.resolve() not in imported_path:
        imported.__path__ = [str(package_root)]
    for subpackage in required_subpackages:
        _load_package_from_init(f"{package_name}.{subpackage}", package_root / subpackage)
    return src_root
