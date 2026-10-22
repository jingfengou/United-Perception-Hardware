"""Locate the first differing full tensor in aligned read-only GPU traces."""
import argparse
import hashlib
import json
from pathlib import Path


def tensors(value, prefix=""):
    if isinstance(value, dict):
        if {"sha256", "shape", "dtype"} <= value.keys():
            return {prefix: value}
        result = {}
        for key, child in value.items():
            result.update(tensors(child, prefix + "/" + key))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(tensors(child, prefix + "/" + str(index)))
        return result
    return {}


def numerical_rows(rows):
    omitted = []
    selected = []
    aliases = ("detach", "clone", "view", "_unsafe_view", "transpose", "t", "permute",
               "slice", "select", "as_strided", "expand", "unsqueeze", "squeeze", "alias",
               "empty", "empty_strided", "empty_like", "new_empty", "new_empty_strided", "lift_fresh")
    for row in rows:
        values = tensors(row["output"]).values()
        has_gpu = any(value["device_type"] != "cpu" for value in values)
        copied = False
        if row["operation"] == "aten.copy_.default":
            args = row["inputs"][0]
            copied = (isinstance(args[0], dict) and isinstance(args[1], dict)
                      and args[0].get("dtype") == args[1].get("dtype"))
        if not has_gpu or row["operation"].split(".")[1] in aliases or copied:
            omitted.append({key: row[key] for key in ("index", "operation", "phase")})
        else:
            selected.append(row)
    return selected, omitted


def value_description(value):
    if isinstance(value, dict):
        if {"sha256", "shape", "dtype"} <= value.keys():
            return {"shape": value["shape"], "dtype": value["dtype"],
                    "device_type": value["device_type"],
                    "hash": value.get("zero_canonical_sha256", value["sha256"])}
        return {key: value_description(item) for key, item in value.items()}
    if isinstance(value, list):
        return [value_description(item) for item in value]
    return value


def compare(left, right):
    if len(left) != len(right):
        raise ValueError("Different operation counts; do not compare unaligned traces")
    result = {"operations": len(left), "first_different": {}, "different_tensors": {},
              "first_different_nonzero_fp32_values": {}, "different_nonzero_fp32_tensors": {}}
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a["operation"] != b["operation"] or a["phase"] != b["phase"]:
            raise ValueError("Different operation sequence at " + str(index))
        x, y = tensors(a["output"]), tensors(b["output"])
        if x.keys() != y.keys():
            raise ValueError("Different output structure at " + str(index))
        for name in x:
            if x[name] != y[name]:
                key = a["phase"] + "/" + x[name]["dtype"]
                result["different_tensors"][key] = result["different_tensors"].get(key, 0) + 1
                if key not in result["first_different"]:
                    result["first_different"][key] = {
                        "index": index, "operation": a["operation"], "output_path": name,
                        "inputs_equal": a["inputs"] == b["inputs"], "left": a, "right": b}
            if (x[name].get("zero_canonical_sha256") is not None
                    and y[name].get("zero_canonical_sha256") is not None
                    and x[name]["zero_canonical_sha256"] != y[name]["zero_canonical_sha256"]):
                phase = a["phase"]
                result["different_nonzero_fp32_tensors"][phase] = result["different_nonzero_fp32_tensors"].get(phase, 0) + 1
                if phase not in result["first_different_nonzero_fp32_values"]:
                    result["first_different_nonzero_fp32_values"][phase] = {
                        "index": index, "operation": a["operation"], "output_path": name,
                        "inputs_equal_ignoring_zero_signs": value_description(a["inputs"]) == value_description(b["inputs"]),
                        "left": a, "right": b}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-only", action="store_true",
                        help="Exclude CPU-only observations, pure aliases/copies and uninitialized allocations; report exclusions")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing comparisons; use a new output file")
    rows = [json.loads(path.read_text()) for path in (args.left, args.right)]
    omissions = [[], []]
    if args.numeric_only:
        filtered = [numerical_rows(values) for values in rows]
        rows = [item[0] for item in filtered]
        omissions = [item[1] for item in filtered]
    result = compare(*rows)
    result.update(scope="Diagnostic only; full tensor hashes, not accuracy acceptance",
                  numeric_only=args.numeric_only, omitted_operations=omissions,
                  trace_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.left, args.right)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"operations": result["operations"],
                      "different_tensors": result["different_tensors"],
                      "first_different": {key: {k: row[k] for k in ("index", "operation", "inputs_equal")}
                                          for key, row in result["first_different"].items()},
                      "first_nonzero_fp32": {key: {k: row[k] for k in ("index", "operation", "inputs_equal_ignoring_zero_signs")}
                                             for key, row in result["first_different_nonzero_fp32_values"].items()}}, indent=2))


if __name__ == "__main__":
    main()
