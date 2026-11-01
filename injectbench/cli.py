"""injectbench download DIR | run --data DIR --detector NAME [--detector NAME ...] [--out report.md]"""
import argparse
import json
import sys

from .data import download_jailbreakhub, load_jailbreakhub, load_jsonl, temporal_split
from .detectors import load_detector
from .run import evaluate, to_markdown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="injectbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download the public JailbreakHub CSVs")
    d.add_argument("dest")
    r = sub.add_parser("run", help="evaluate detectors on a time-split dataset")
    r.add_argument("--data", required=True, help="directory from `download`, or a .jsonl file (text,label,date)")
    r.add_argument("--detector", action="append", required=True)
    r.add_argument("--train-frac", type=float, default=0.65)
    r.add_argument("--prevalence", type=float, default=0.05)
    r.add_argument("--boot", type=int, default=200)
    r.add_argument("--out", help="write markdown report here")
    a = ap.parse_args(argv)
    if a.cmd == "download":
        download_jailbreakhub(a.dest)
        print("downloaded to", a.dest)
        return 0
    rows = load_jsonl(a.data) if a.data.endswith(".jsonl") else load_jailbreakhub(a.data)
    train, test = temporal_split(rows, a.train_frac)
    info = (f"train {len(train)} ({sum(x.label for x in train)} attacks) | test {len(test)} "
            f"({sum(x.label for x in test)} attacks, {100 * sum(x.label for x in test) / len(test):.1f}% prevalence)")
    results = [evaluate(load_detector(s), train, test, a.prevalence, a.boot) for s in a.detector]
    md = to_markdown(results, info, a.prevalence)
    print(md)
    if a.out:
        open(a.out, "w").write(md)
        json.dump(results, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
