#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""iPinYou RTB -> Yodo1 广告业务全流程 Side-Project 入口。

本地 Mock 全闭环，命令示例：
  python main.py pipeline                # 阶段1：建事件宽表 + 基线质量报告
  python main.py quality --ad-id 2345    # 阶段1：单单元质量校验报告
  python main.py tools-demo              # 阶段2：演示 4 个 Mock 工具
  python main.py diagnose "..."          # 阶段3：故障排查 Agent(自动识别AdID)
  python main.py inspect [--watch]       # 阶段4：定时巡检报告
  python main.py report "..."            # 阶段4：自然语言分析 -> 业务报告
  python main.py eval [--mode mock|llm]  # 阶段5：评测闭环
  python main.py absorb                  # 阶段5：bad-case 沉淀进知识库
"""
from __future__ import annotations

import argparse
import json
import sys

from ipinyou_agent import config as C


def cmd_pipeline(args):
    C.ensure_dirs()
    from ipinyou_agent import quality as Q, warehouse as W
    print("[pipeline] build events wide table from raw multi-source logs ...")
    events = W.load_events()
    wide = W.load_wide()
    print(f"[pipeline] events rows={len(events):,} wide rows={len(wide):,}")
    print(f"[pipeline] advertisers={sorted(events['AdvertiserID'].unique().tolist())} "
          f"ad_units={len(W.known_campaign_ids())}")
    # 基线质量校验（整库 + 旗舰故障单元示例）
    base = Q.run_quality_checks()
    print(f"[pipeline] global DQ issues={len(base['issues'])} -> {base['summary'][:60]}")
    out = C.REPORT_DIR / "quality_baseline.md"
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(Q.quality_to_markdown(base), encoding="utf-8")
    print(f"[pipeline] quality baseline report: {out}")
    # 样例单元质量
    demo = Q.run_quality_checks(ad_id=2345)
    dout = C.REPORT_DIR / "quality_demo_2345.md"
    dout.write_text(Q.quality_to_markdown(demo), encoding="utf-8")
    print(f"[pipeline] demo unit 2345 DQ issues={len(demo['issues'])} report: {dout}")
    return 0


def cmd_gen(args):
    C.ensure_dirs()
    from ipinyou_agent import generator
    r = generator.generate_dataset(force=args.force)
    print(f"[gen] result={json.dumps(r, default=str, ensure_ascii=False)[:200]}")
    return 0


def cmd_quality(args):
    C.ensure_dirs()
    from ipinyou_agent import quality as Q, warehouse as W
    W.load_events()
    report = Q.run_quality_checks(ad_id=args.ad_id, window_hours=args.window_hours)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = C.REPORT_DIR / f"dq_ad{args.ad_id}.md"
    path.write_text(Q.quality_to_markdown(report), encoding="utf-8")
    print(f"[quality] ad_id={args.ad_id} issues={len(report['issues'])}")
    print(f"[quality] report written: {path}")
    for it in report["issues"]:
        print(f"  - {it['rule_id']} [{it['severity']}] {it['name']}")
    return 0


def cmd_tools_demo(args):
    C.ensure_dirs()
    from ipinyou_agent import warehouse as W
    W.load_events()
    from ipinyou_agent.tools import (get_campaign_events, get_campaign_metrics,
                                     run_data_quality_check, search_knowledge_base)
    for name, call in [
        ("get_campaign_metrics(2345)", lambda: get_campaign_metrics(2345)),
        ("get_campaign_events(2345)", lambda: get_campaign_events(2345)),
        ("run_data_quality_check(2345)", lambda: run_data_quality_check(2345, store_snapshot=False)),
        ("search_knowledge_base('出价正常但无曝光 消耗暴跌')",
         lambda: search_knowledge_base("出价正常但无曝光 消耗暴跌")),
    ]:
        res = call()
        print(f"[tools] {name}")
        if name.startswith("get_campaign_metrics"):
            print("  summary:", res["summary"], "| cur:", res["current_24h"])
        elif name.startswith("get_campaign_events"):
            print("  total_events:", res.get("total_events"), "counts:", res.get("counts"))
        elif name.startswith("run_data_quality_check"):
            print("  summary:", res["summary"])
            for i in res["issues"]:
                print(f"    - {i['rule_id']} [{i['severity']}] {i['name']}")
        else:
            print("  hits:", [(h["title"], h["root_cause_tag"]) for h in res["hits"]])
    return 0


def cmd_diagnose(args):
    C.ensure_dirs()
    query = args.query or "广告单元 AdID:2345 最近24小时消耗暴跌，但是出价没有调低，请排查原因，给出修复建议"
    from ipinyou_agent.agent_core import run_diagnosis
    res = run_diagnosis(query, ad_ids=[args.ad_id] if args.ad_id else None,
                        mode=args.mode)
    print(f"[diagnose] mode={res.mode} tool_calls={res.n_tool_calls} "
          f"used={res.used_tools} duration={res.duration_sec}s")
    print(f"[diagnose] summary: {res.report.get('summary','')[:120]}")
    print(f"[diagnose] root_cause_tags: {res.report.get('root_cause_tags')}")
    if args.save:
        C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        from ipinyou_agent.agent_core import report_to_markdown
        path = C.REPORT_DIR / "diagnosis_report.md"
        path.write_text(report_to_markdown(res), encoding="utf-8")
        print(f"[diagnose] markdown report written: {path}")
    else:
        print("[diagnose] (add --save to write markdown report)")
    return 0


def cmd_inspect(args):
    C.ensure_dirs()
    from ipinyou_agent.workflows import run_inspection
    run_inspection(watch=args.watch, interval_sec=args.interval)
    return 0


def cmd_report(args):
    C.ensure_dirs()
    query = args.query or "广告单元 AdID:2345 最近24小时消耗暴跌，但是出价没有调低，请排查原因，给出修复建议"
    from ipinyou_agent.workflows import nl_report
    res = nl_report(query, ad_ids=[args.ad_id] if args.ad_id else None, mode=args.mode)
    print(f"[report] mode={res.mode} tool_calls={res.n_tool_calls} tags={res.report.get('root_cause_tags')}")
    print(f"[report] written: {getattr(res, '_saved_path', 'N/A')}")
    return 0


def cmd_eval(args):
    C.ensure_dirs()
    from ipinyou_agent.evaluation import absorb_bad_cases, evaluate
    print(f"[eval] building eval cases and running agent (mode={args.mode or 'auto'}) ...")
    out = evaluate(mode=args.mode, max_cases=args.max_cases, use_judge=args.judge,
                   shuffle=args.shuffle)
    s = out["summary"]
    print("\n[eval] ===== SUMMARY =====")
    print(f"[eval] total_cases      : {s['total_cases']}")
    print(f"[eval] root_accuracy    : {s['root_accuracy']}")
    print(f"[eval] action_recall    : {s['action_recall_macro']} "
          f"(reasonable>=0.5: {s['action_reasonable_rate']})")
    print(f"[eval] avg_tool_calls   : {s['avg_tool_calls']}  avg_extra: {s['avg_extra_calls']}")
    print(f"[eval] tool_coverage    : {s['tool_coverage_macro']}")
    print(f"[eval] failed(bad)cases : {s['failed_cases']}")
    meta = out["meta_paths"]
    print(f"[eval] json={meta['json']}")
    print(f"[eval] markdown={meta['md']}")
    if args.absorb:
        n = absorb_bad_cases(result=out)
        print(f"[eval] absorbed {n} bad-cases into KB")
    return 0


def cmd_absorb(args):
    C.ensure_dirs()
    from ipinyou_agent.evaluation import absorb_bad_cases
    n = absorb_bad_cases()
    print(f"[absorb] absorbed {n} bad-cases into KB")
    return 0


def cmd_kb(args):
    C.ensure_dirs()
    from ipinyou_agent.knowledge_base import get_kb
    kb = get_kb()
    print(f"[kb] total docs={kb.count()} (including {len([d for d in kb.docs if d.get('source')!='industry-case'])} added)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ipinyou-agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("pipeline", help="阶段1: 生成->宽表join->质量校验").set_defaults(func=cmd_pipeline)
    g = sub.add_parser("gen", help="强制重新生成 iPinYou mock 日志")
    g.add_argument("--force", action="store_true")
    g.set_defaults(func=cmd_gen)

    q = sub.add_parser("quality", help="阶段1: 单单元质量报告")
    q.add_argument("--ad-id", type=int, default=2345)
    q.add_argument("--window-hours", type=int, default=None)
    q.set_defaults(func=cmd_quality)

    sub.add_parser("tools-demo", help="阶段2: 演示 4 个 Mock 工具").set_defaults(func=cmd_tools_demo)

    d = sub.add_parser("diagnose", help="阶段3: 故障排查 Agent")
    d.add_argument("query", nargs="?", default="")
    d.add_argument("--ad-id", type=int, default=None)
    d.add_argument("--mode", choices=["mock", "llm", "auto"], default=None)
    d.add_argument("--save", action="store_true")
    d.set_defaults(func=cmd_diagnose)

    i = sub.add_parser("inspect", help="阶段4: 定时巡检(可 --watch)")
    i.add_argument("--watch", action="store_true")
    i.add_argument("--interval", type=int, default=None)
    i.set_defaults(func=cmd_inspect)

    r = sub.add_parser("report", help="阶段4: 自然语言 -> 业务报告")
    r.add_argument("query", nargs="?", default="")
    r.add_argument("--ad-id", type=int, default=None)
    r.add_argument("--mode", choices=["mock", "llm", "auto"], default=None)
    r.set_defaults(func=cmd_report)

    e = sub.add_parser("eval", help="阶段5: 评测闭环")
    e.add_argument("--mode", choices=["mock", "llm", "auto"], default=None)
    e.add_argument("--max-cases", type=int, default=0)
    e.add_argument("--shuffle", action="store_true")
    e.add_argument("--judge", action="store_true", help="用 LLM 复核建议合理性")
    e.add_argument("--absorb", action="store_true", help="评测后自动沉淀 bad-case")
    e.set_defaults(func=cmd_eval)

    sub.add_parser("absorb", help="阶段5: bad-case 沉淀进知识库").set_defaults(func=cmd_absorb)
    sub.add_parser("kb", help="查看知识库文档数").set_defaults(func=cmd_kb)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        build_parser().print_help()
        return 1
    # 统一捕获并展示错误
    try:
        return args.func(args)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\n[fatal] {e!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
