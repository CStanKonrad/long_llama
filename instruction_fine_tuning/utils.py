from typing import Dict, Any


def metrics_assign_group(metrics_dict: Dict[str, Any], group: str, index: int = 0):
    result = {}
    for k, v in metrics_dict.items():
        groups = k.split("/")
        abs_index = index % len(groups)
        groups = groups[:abs_index] + [group] + groups[abs_index:]
        new_k = "/".join(groups)
        result[new_k] = v
    return result


def non_numeric_to_str(metrics_dict: Dict[str, Any]):
    result = {}
    for k, v in metrics_dict.items():
        if not isinstance(v, int) and not isinstance(v, float):
            result[k] = str(v)
        else:
            result[k] = v
    return result


def get_packages():
    """For getting the list of installed packages"""
    import pkg_resources

    result = {}
    for pkg in pkg_resources.working_set:
        pkg_name, pkg_ver = str(pkg.project_name), str(pkg.version)
        result[pkg_name] = pkg_ver
    return result
