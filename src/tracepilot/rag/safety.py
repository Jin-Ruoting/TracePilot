from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptInjectionFinding:
    category: str
    matched_text: str
    start: int
    end: int


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?"
            r"(?:(?:previous|prior)\s+)?(?:system\s+|developer\s+|user\s+)?"
            r"(?:instructions?|messages?|prompts?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"\b(?:reveal|print|return|exfiltrat(?:e|ion))\b.{0,50}"
            r"\b(?:system prompt|developer message|api key|secret|token|credentials?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "tool_coercion",
        re.compile(
            r"\b(?:assistant|agent|model|you)\b.{0,30}"
            r"\b(?:call|invoke|use|execute|run)\s+(?:the\s+)?"
            r"(?:shell|terminal|tool|command|python)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "destructive_instruction",
        re.compile(
            r"\b(?:delete|erase|destroy|overwrite)\s+(?:all\s+)?"
            r"(?:files?|data|repository|repo|database)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_impersonation",
        re.compile(
            r"(?:^|\n)\s*(?:system|developer|assistant)\s*:\s*",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_override",
        re.compile(
            r"(?:忽略|无视|覆盖|忘记)(?:所有|全部)?(?:之前|以上)?(?:的)?"
            r"(?:系统|开发者|用户)?(?:的)?(?:指令|提示|要求)",
        ),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"(?:泄露|显示|输出|返回).{0,30}(?:系统提示|开发者消息|密钥|令牌|凭证)",
            re.DOTALL,
        ),
    ),
    (
        "tool_coercion",
        re.compile(
            r"(?:你|助手|代理|模型).{0,20}"
            r"(?:调用|执行|运行)(?:该|这个)?(?:工具|终端|命令|脚本|Shell)",
            re.IGNORECASE,
        ),
    ),
)


def scan_prompt_injection(text: str) -> tuple[PromptInjectionFinding, ...]:
    findings: list[PromptInjectionFinding] = []
    for category, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                PromptInjectionFinding(
                    category=category,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
    findings.sort(key=lambda finding: (finding.start, finding.end, finding.category))
    return tuple(findings)


def has_prompt_injection(text: str) -> bool:
    return bool(scan_prompt_injection(text))


def findings_to_metadata(
    findings: tuple[PromptInjectionFinding, ...],
) -> list[dict[str, str | int]]:
    return [
        {
            "category": finding.category,
            "matched_text": finding.matched_text,
            "start": finding.start,
            "end": finding.end,
        }
        for finding in findings
    ]
