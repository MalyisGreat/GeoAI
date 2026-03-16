from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _purge_package(package_name: str) -> None:
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)


def bootstrap_project_src(project_root: Path, package_name: str = "geobot") -> Path:
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
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[str(package_root)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot create import spec for {package_name!r} from {init_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

    imported = sys.modules.get(package_name)
    imported_file = getattr(imported, "__file__", None)
    if imported_file is None or Path(imported_file).resolve() != init_path.resolve():
        raise RuntimeError(f"Failed to resolve local {package_name!r} package from {init_path}")
    return src_root
