from __future__ import annotations

import csv
import html
import json
from typing import Any

from .models import canonical


def number(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def write_report(coordinator: Any, project_id: str) -> dict[str, str]:
    report = coordinator.review(project_id)
    root = coordinator.report_directory(project_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / "definition.json").write_text(canonical(coordinator.store.project(project_id)["definition"]) + "\n")
    jobs = coordinator.store.jobs(project_id)
    (root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    fields = ["job_id", "dataset", "variant", "seed", "status", "metric", "value", "run_seconds",
              "code_hash", "split_hash", "protocol_hash", "metrics_sha256", "metrics_path", "error"]
    with (root / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in jobs:
            job, result = row["definition"], row["result"] or {}
            metrics = result.get("metrics", {})
            writer.writerow({
                "job_id": row["id"], "dataset": job["dataset"], "variant": job["variant"], "seed": job["seed"],
                "status": row["status"], "metric": job["metric"],
                "value": metrics.get("metrics", {}).get(job["metric"]), "run_seconds": result.get("elapsed_seconds"),
                "code_hash": job["code_hash"], "split_hash": metrics.get("split_hash", ""),
                "protocol_hash": metrics.get("protocol_hash", ""), "metrics_sha256": result.get("metrics_sha256", ""),
                "metrics_path": (f"runs/{row['id']}/metrics.json" if job.get("artifact_directory")
                                 else f"runs/{row['id']}/work/metrics.json"), "error": result.get("error", ""),
            })
    events = coordinator.store.events(project_id)
    (root / "events.jsonl").write_text("".join(canonical(event) + "\n" for event in events))
    (root / "decisions.jsonl").write_text("".join(canonical(row) + "\n" for row in coordinator.store.decisions(project_id)))
    lines = [
        f"# {project_id}：科研实验报告", "", f"假设：{report['hypothesis']}", "",
        f"- 假设来源：{report['hypothesis_origin']}。",
        f"- 数据分区：{report['partition']}；主要指标：{report['metric']}。",
        f"- 成功运行：{report['successful_runs']}/{report['planned_runs']}（{report['completion_rate']:.1%}）。",
        f"- 成功运行的指标产物覆盖率：{report['artifact_coverage']:.1%}。",
        f"- 累计实验运行时间：{report['run_seconds']:.2f} 秒（各作业时长之和，不是墙钟时间）。",
        f"- 执行后端：{report['backend']['kind']}；生成模型调用：0（本次执行预定义实验规格）。",
        "", "| 数据集 | 候选 | 配对数 | 基线均值 ± SD | 候选均值 ± SD | 改善量 | 配对 bootstrap 95% 区间 | 初步判断 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["comparisons"]:
        interval = row["bootstrap_95_interval"]
        ci = f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "—"
        lines.append(
            f"| {row['dataset']} | {row['candidate']} | {row['paired_runs']}/{row['expected_pairs']} | "
            f"{number(row['baseline_mean'])} ± {number(row['baseline_sd'])} | "
            f"{number(row['candidate_mean'])} ± {number(row['candidate_sd'])} | "
            f"{number(row['oriented_delta_mean'])} | {ci} | {row['verdict']} |"
        )
    lines.extend([
        "", "改善量按指标方向计算，正值表示候选更好。SD 是同一固定划分下不同训练种子的样本标准差。",
        "区间为探索性配对 bootstrap；少量种子不足以支持普遍有效性或新颖性结论。没有使用测试集选择方案。",
        "缺失、失败、协议不一致的运行保留在记录中，相应比较标为 inconclusive。",
        "", "本实验使用预定义假设和补丁验证系统闭环，未声称由模型自主提出新方法。",
        "", "[逐次运行 CSV](trials.csv) · [结构化结果](report.json) · [冻结实验定义](definition.json) · [事件记录](events.jsonl) · [路由与审查记录](decisions.jsonl)", "",
    ])
    failures = [j for j in jobs if j["status"] != "succeeded"]
    if failures:
        lines.extend(["## 未成功运行", ""])
        for row in failures:
            lines.append(f"- {row['id']}：{row['status']}；{(row['result'] or {}).get('error', '尚未完成')}")
    markdown = "\n".join(lines) + "\n"
    (root / "report.md").write_text(markdown)
    table_rows = "".join(
        "<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in
                        (r["dataset"], r["candidate"], f"{r['paired_runs']}/{r['expected_pairs']}",
                         number(r["baseline_mean"]), number(r["candidate_mean"]),
                         number(r["oriented_delta_mean"]), r["verdict"])) + "</tr>"
        for r in report["comparisons"]
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>{html.escape(project_id)} 实验报告</title><style>
body{{font:16px/1.7 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#172b3a}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #d8e0e6}}
.metric{{display:inline-block;padding:15px 25px;background:#eef5f8;margin:8px}}a{{color:#066b92}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:20px}}
</style><h1>{html.escape(project_id)}</h1><p>{html.escape(report['hypothesis'])}</p>
<div class="metric">成功运行 {report['successful_runs']} / {report['planned_runs']}</div>
<div class="metric">产物覆盖 {report['artifact_coverage']:.0%}</div>
<div class="metric">累计运行 {report['run_seconds']:.1f} 秒</div>
<table><thead><tr><th>数据集</th><th>候选</th><th>配对</th><th>基线</th><th>候选</th><th>改善量</th><th>判断</th></tr></thead>
<tbody>{table_rows}</tbody></table>
<p>预定义实验规格；验证集结果；生成模型未调用；未声称科研新颖性。</p>
<p><a href="trials.csv">原始 CSV</a> · <a href="report.json">结果 JSON</a> ·
<a href="definition.json">实验定义</a> · <a href="events.jsonl">事件记录</a></p>
<details><summary>完整报告</summary><pre>{html.escape(markdown)}</pre></details></html>"""
    (root / "report.html").write_text(page)
    return {name: str(root / f"report.{name}") for name in ("md", "json", "html")}
