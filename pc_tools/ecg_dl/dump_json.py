#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump a JSON file structure for inspection."""
import json
import sys


def walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                walk(v, p + "/" + str(k))
            else:
                print(f"{p}/{k} = {v}")
    elif isinstance(o, list):
        print(f"{p} [list len={len(o)}]")
        for i in o[:3]:
            walk(i, p + "[0]")


if __name__ == "__main__":
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    walk(d)
