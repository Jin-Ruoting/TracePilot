from __future__ import annotations

import argparse
from pathlib import Path


def render_workbench(initial_task_id: str = "") -> str:
    return _HTML.replace("__INITIAL_TASK_ID__", initial_task_id)


def write_workbench(path: str | Path, initial_task_id: str = "") -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_workbench(initial_task_id), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the TracePilot workbench.")
    parser.add_argument("--out", default="traces/workbench.html")
    parser.add_argument("--task-id", default="")
    args = parser.parse_args()
    write_workbench(args.out, args.task_id)
    print(args.out)


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TracePilot Workbench</title>
  <style>
    :root {
      --bg: #f5f7f8;
      --surface: #ffffff;
      --surface-2: #eef2f4;
      --ink: #172126;
      --muted: #65727a;
      --line: #d8e0e4;
      --accent: #087f8c;
      --accent-dark: #075e68;
      --green: #247a4a;
      --green-bg: #e7f4ec;
      --amber: #996515;
      --amber-bg: #fff3d6;
      --red: #a23832;
      --red-bg: #fae9e7;
      --blue: #285f9e;
      --blue-bg: #e9f1fb;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, textarea { font: inherit; letter-spacing: 0; }
    button { cursor: pointer; }
    .app { min-height: 100vh; display: grid; grid-template-rows: 52px 1fr; }
    header {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 0 18px;
      color: white;
      background: #172126;
      border-bottom: 1px solid #25343b;
    }
    .brand { font-size: 17px; font-weight: 700; }
    .header-status { color: #a8c5c9; font-size: 12px; }
    .header-spacer { flex: 1; }
    .layout { min-height: 0; display: grid; grid-template-columns: 280px minmax(0, 1fr); }
    aside { min-height: 0; overflow: auto; background: var(--surface); border-right: 1px solid var(--line); }
    .sidebar-head { position: sticky; top: 0; z-index: 2; display: flex; align-items: center; gap: 8px; padding: 12px; background: var(--surface); border-bottom: 1px solid var(--line); }
    .sidebar-head strong { flex: 1; }
    .btn {
      min-height: 34px;
      padding: 7px 11px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink);
      background: var(--surface);
    }
    .btn:hover { border-color: #9baab1; background: #f8fafb; }
    .btn.primary { color: white; border-color: var(--accent); background: var(--accent); }
    .btn.primary:hover { background: var(--accent-dark); }
    .btn.danger { color: var(--red); border-color: #dcb2ae; }
    .task-row {
      width: 100%;
      display: grid;
      gap: 5px;
      padding: 12px;
      text-align: left;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: transparent;
    }
    .task-row:hover, .task-row.active { background: var(--surface-2); }
    .task-title { overflow: hidden; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
    .task-meta { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 12px; }
    main { min-width: 0; overflow: auto; }
    .main-head { padding: 18px 22px 10px; background: var(--surface); border-bottom: 1px solid var(--line); }
    h1 { max-width: 960px; margin: 0; font-size: 23px; line-height: 1.25; overflow-wrap: anywhere; }
    .subline { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 9px; color: var(--muted); }
    .tabs { display: flex; gap: 18px; margin-top: 16px; }
    .tab {
      min-height: 36px;
      padding: 7px 0;
      border: 0;
      border-bottom: 2px solid transparent;
      color: var(--muted);
      background: transparent;
    }
    .tab.active { color: var(--accent-dark); border-color: var(--accent); font-weight: 650; }
    .content { width: min(1180px, 100%); padding: 20px 22px 48px; }
    .band { padding: 18px 0; border-bottom: 1px solid var(--line); }
    .band:first-child { padding-top: 0; }
    .band h2 { margin: 0 0 12px; font-size: 15px; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
    .metric { min-width: 0; padding: 13px 14px; border-right: 1px solid var(--line); }
    .metric:last-child { border-right: 0; }
    .metric-label { color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .metric-value { margin-top: 4px; font-size: 18px; font-weight: 700; overflow-wrap: anywhere; }
    .timeline { display: grid; gap: 0; border-left: 2px solid var(--line); margin-left: 7px; }
    .step { position: relative; display: grid; grid-template-columns: minmax(160px, 240px) 1fr; gap: 18px; padding: 10px 0 10px 20px; }
    .step::before { content: ""; position: absolute; width: 10px; height: 10px; left: -6px; top: 15px; border-radius: 50%; background: #9baab1; }
    .step.done::before { background: var(--green); }
    .step.running::before { background: var(--accent); box-shadow: 0 0 0 3px #d9eff1; }
    .step.failed::before { background: var(--red); }
    .step-name { font-weight: 650; overflow-wrap: anywhere; }
    .step-detail { color: var(--muted); overflow-wrap: anywhere; }
    .badge { display: inline-flex; align-items: center; min-height: 22px; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
    .badge.completed, .badge.approved { color: var(--green); background: var(--green-bg); }
    .badge.awaiting_approval, .badge.pending { color: var(--amber); background: var(--amber-bg); }
    .badge.failed, .badge.rejected { color: var(--red); background: var(--red-bg); }
    .badge.researching, .badge.planning, .badge.executing, .badge.indexing { color: var(--blue); background: var(--blue-bg); }
    .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px 12px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); background: #f8fafb; font-size: 11px; text-transform: uppercase; }
    tr:last-child td { border-bottom: 0; }
    code, pre { font-family: var(--mono); }
    code { overflow-wrap: anywhere; }
    a { color: var(--accent-dark); }
    pre { max-height: 320px; margin: 0; padding: 12px; overflow: auto; color: #e8eef0; background: #1e2a30; border-radius: 6px; white-space: pre-wrap; }
    details { padding: 10px 0; border-bottom: 1px solid var(--line); scroll-margin-top: 16px; }
    details:last-child { border-bottom: 0; }
    details summary { margin-bottom: 8px; cursor: pointer; font-weight: 650; }
    .evidence-quote { max-width: 720px; color: #33434a; }
    .approval { display: grid; gap: 12px; padding: 14px; border: 1px solid #e3c98c; border-radius: 6px; background: #fffaf0; }
    .approval-actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .empty { padding: 28px 0; color: var(--muted); text-align: center; }
    dialog { width: min(620px, calc(100vw - 28px)); padding: 0; border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 18px 60px #17212633; }
    dialog::backdrop { background: #17212666; }
    .dialog-head, .dialog-body, .dialog-actions { padding: 16px 18px; }
    .dialog-head { border-bottom: 1px solid var(--line); font-weight: 700; }
    .dialog-body { display: grid; gap: 12px; }
    .dialog-actions { display: flex; justify-content: flex-end; gap: 8px; border-top: 1px solid var(--line); }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    input, textarea { width: 100%; padding: 9px 10px; color: var(--ink); border: 1px solid var(--line); border-radius: 5px; background: var(--surface); }
    textarea { min-height: 90px; resize: vertical; }
    .toast { position: fixed; right: 18px; bottom: 18px; max-width: min(420px, calc(100vw - 36px)); padding: 11px 14px; color: white; background: #26363d; border-radius: 6px; opacity: 0; transform: translateY(8px); pointer-events: none; transition: 160ms ease; }
    .toast.show { opacity: 1; transform: translateY(0); }
    @media (max-width: 820px) {
      .layout { grid-template-columns: 1fr; }
      aside { max-height: 220px; border-right: 0; border-bottom: 1px solid var(--line); }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric:nth-child(2) { border-right: 0; }
      .metric:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
      .step { grid-template-columns: 1fr; gap: 2px; }
    }
    @media (max-width: 480px) {
      header, .main-head, .content { padding-left: 14px; padding-right: 14px; }
      .metrics { grid-template-columns: 1fr; }
      .metric { border-right: 0; border-bottom: 1px solid var(--line); }
      .metric:last-child { border-bottom: 0; }
      .tabs { gap: 13px; overflow-x: auto; }
    }
  </style>
</head>
<body>
<div class="app">
  <header>
    <div class="brand">TracePilot</div>
    <div class="header-status" id="connection">Local workbench</div>
    <div class="header-spacer"></div>
    <button class="btn primary" id="new-task-button">New task</button>
  </header>
  <div class="layout">
    <aside>
      <div class="sidebar-head"><strong>Tasks</strong><button class="btn" id="refresh-button">Refresh</button></div>
      <div id="task-list"><div class="empty">No tasks</div></div>
    </aside>
    <main>
      <div class="main-head">
        <h1 id="task-goal">Select a task</h1>
        <div class="subline"><span id="task-status" class="badge pending">idle</span><span id="task-workspace"></span></div>
        <nav class="tabs" aria-label="Task views">
          <button class="tab active" data-tab="overview">Overview</button>
          <button class="tab" data-tab="evidence">Evidence</button>
          <button class="tab" data-tab="memory">Memory</button>
          <button class="tab" data-tab="execution">Execution</button>
        </nav>
      </div>
      <div class="content" id="content"><div class="empty">Create or select a task.</div></div>
    </main>
  </div>
</div>

<dialog id="new-task-dialog">
  <form method="dialog" id="new-task-form">
    <div class="dialog-head">New technical decision</div>
    <div class="dialog-body">
      <label>Goal<textarea name="goal" required placeholder="What should TracePilot investigate or decide?"></textarea></label>
      <label>Workspace path<input name="workspace_path" value="."></label>
      <label>Source URLs<input name="source_urls" placeholder="Comma-separated URLs"></label>
      <label>Constraints<input name="constraints" placeholder="Comma-separated constraints"></label>
    </div>
    <div class="dialog-actions"><button class="btn" value="cancel">Cancel</button><button class="btn primary" value="default">Create</button></div>
  </form>
</dialog>
<dialog id="modify-approval-dialog">
  <form method="dialog" id="modify-approval-form">
    <div class="dialog-head">Modify frozen arguments</div>
    <div class="dialog-body">
      <label>Arguments JSON<textarea name="arguments" required></textarea></label>
    </div>
    <div class="dialog-actions"><button class="btn" value="cancel">Cancel</button><button class="btn primary" value="default">Freeze changes</button></div>
  </form>
</dialog>
<div class="toast" id="toast"></div>

<script>
  const state = { tasks: [], task: null, tab: "overview", stream: null, initialTask: "__INITIAL_TASK_ID__" };
  const $ = (selector) => document.querySelector(selector);
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const badge = (value) => `<span class="badge ${esc(value)}">${esc(value || "unknown")}</span>`;
  const asList = (value) => Array.isArray(value) ? value : [];
  const toast = (message) => { const node = $("#toast"); node.textContent = message; node.classList.add("show"); setTimeout(() => node.classList.remove("show"), 2200); };

  async function jsonFetch(url, options = {}) {
    const response = await fetch(url, { headers: {"Content-Type": "application/json"}, ...options });
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
    return response.json();
  }

  async function loadTasks() {
    try {
      const data = await jsonFetch("/v1/tasks");
      state.tasks = asList(data.tasks);
      renderTaskList();
      const target = state.initialTask || state.task?.task_id || state.tasks[0]?.task_id;
      state.initialTask = "";
      if (target) await loadTask(target);
    } catch (error) {
      $("#connection").textContent = "API unavailable";
      toast(error.message);
    }
  }

  function renderTaskList() {
    $("#task-list").innerHTML = state.tasks.length ? state.tasks.map(task => `
      <button class="task-row ${state.task?.task_id === task.task_id ? "active" : ""}" data-task-id="${esc(task.task_id)}">
        <span class="task-title">${esc(task.goal || task.task_id)}</span>
        <span class="task-meta">${badge(task.status)}<span>${esc(task.updated_at || "")}</span></span>
      </button>`).join("") : '<div class="empty">No tasks</div>';
    document.querySelectorAll("[data-task-id]").forEach(node => node.addEventListener("click", () => loadTask(node.dataset.taskId)));
  }

  async function loadTask(taskId) {
    state.task = await jsonFetch(`/v1/tasks/${encodeURIComponent(taskId)}`);
    renderTaskList();
    renderTask();
    connectEvents(taskId);
  }

  function renderTask() {
    const task = state.task;
    if (!task) return;
    $("#task-goal").textContent = task.request?.goal || task.goal || task.task_id;
    $("#task-status").className = `badge ${task.status}`;
    $("#task-status").textContent = task.status;
    $("#task-workspace").textContent = task.request?.workspace_path || task.workspace_path || "";
    renderTab();
  }

  function renderTab() {
    document.querySelectorAll(".tab").forEach(node => node.classList.toggle("active", node.dataset.tab === state.tab));
    if (!state.task) {
      $("#content").innerHTML = '<div class="empty">Create or select a task.</div>';
      return;
    }
    const renderers = { overview: renderOverview, evidence: renderEvidence, memory: renderMemory, execution: renderExecution };
    $("#content").innerHTML = (renderers[state.tab] || renderOverview)();
    bindActions();
  }

  function renderOverview() {
    const task = state.task;
    const plan = task.plan || {};
    const steps = asList(plan.steps);
    const done = steps.filter(step => step.status === "completed").length;
    return `
      <section class="band">
        <div class="metrics">
          <div class="metric"><div class="metric-label">Plan steps</div><div class="metric-value">${steps.length}</div></div>
          <div class="metric"><div class="metric-label">Completed</div><div class="metric-value">${done}</div></div>
          <div class="metric"><div class="metric-label">Evidence</div><div class="metric-value">${asList(task.evidence).length}</div></div>
          <div class="metric"><div class="metric-label">Approvals</div><div class="metric-value">${asList(task.approvals).length}</div></div>
        </div>
      </section>
      <section class="band"><h2>Plan</h2><div class="timeline">${steps.length ? steps.map(step => `
        <div class="step ${esc(step.status)}"><div><div class="step-name">${esc(step.objective || step.step_id)}</div>${badge(step.status)}</div><div class="step-detail"><code>${esc(step.tool_name || "")}</code><br>${esc(step.error || step.detail || "")}</div></div>`).join("") : '<div class="empty">No plan yet</div>'}</div></section>
      <section class="band"><h2>Decision</h2><div>${task.summary ? esc(task.summary) : '<span class="empty">Pending</span>'}</div></section>`;
  }

  function renderEvidence() {
    const items = asList(state.task.evidence);
    const chunks = asList(state.task.evidence_chunks);
    return `<section class="band"><h2>Grounded claims</h2><div class="table-wrap"><table><thead><tr><th>Claim</th><th>Source</th><th>Confidence</th></tr></thead><tbody>${items.length ? items.map(item => `
      <tr><td><div class="evidence-quote">${esc(item.claim || item.text)}</div><div>${badge(item.supported === false ? "uncertain" : "support")}</div></td><td><a href="#source-${esc(item.source_id || "")}"><code>${esc(item.source_title || item.source_id || "")}</code></a><br>${esc(item.location || "")}</td><td>${Number(item.confidence ?? 0).toFixed(2)}</td></tr>`).join("") : '<tr><td colspan="3" class="empty">No evidence</td></tr>'}</tbody></table></div></section>
      <section class="band"><h2>Source chunks</h2>${chunks.length ? chunks.map(chunk => `<details id="source-${esc(chunk.chunk_id)}"><summary><code>${esc(chunk.chunk_id)}</code> ${esc(chunk.title || "")}</summary><pre>${esc(chunk.text || "")}</pre></details>`).join("") : '<div class="empty">No source chunks</div>'}</section>`;
  }

  function renderMemory() {
    const items = asList(state.task.memories);
    return `<section class="band"><h2>Task memory</h2><div class="table-wrap"><table><thead><tr><th>Kind</th><th>Content</th><th>Status</th><th></th></tr></thead><tbody>${items.length ? items.map(item => `
      <tr><td>${esc(item.kind)}</td><td>${esc(typeof item.content === "string" ? item.content : JSON.stringify(item.content))}</td><td>${badge(item.status)}</td><td><button class="btn danger" data-memory-delete="${esc(item.memory_id)}">Delete</button></td></tr>`).join("") : '<tr><td colspan="4" class="empty">No reusable memory</td></tr>'}</tbody></table></div></section>
      <section class="band"><button class="btn primary" id="promote-memory">Save decision to memory</button></section>`;
  }

  function renderExecution() {
    const approvals = asList(state.task.approvals);
    const results = asList(state.task.tool_results);
    return `<section class="band"><h2>Approvals</h2>${approvals.length ? approvals.map(item => `
      <div class="approval"><div><strong>${esc(item.tool_name)}</strong> ${badge(item.status)}</div><div>${esc(item.reason || "")}</div><pre>${esc(JSON.stringify(item.arguments || {}, null, 2))}</pre>${["pending", "modified"].includes(item.status) ? `<div class="approval-actions"><button class="btn primary" data-approval="${esc(item.approval_id)}" data-decision="approved">Approve</button><button class="btn" data-approval="${esc(item.approval_id)}" data-decision="modified">Modify</button><button class="btn danger" data-approval="${esc(item.approval_id)}" data-decision="rejected">Reject</button></div>` : ""}</div>`).join("") : '<div class="empty">No approvals</div>'}</section>
      <section class="band"><h2>Tool results</h2>${results.length ? results.map(item => `<div class="step ${item.ok ? "done" : "failed"}"><div class="step-name">${esc(item.tool_name)}</div><div class="step-detail"><pre>${esc(JSON.stringify(item.output ?? item.error, null, 2))}</pre></div></div>`).join("") : '<div class="empty">No execution results</div>'}</section>`;
  }

  function bindActions() {
    document.querySelectorAll("[data-approval]").forEach(node => node.addEventListener("click", async () => {
      if (node.dataset.decision === "modified") {
        const approval = asList(state.task.approvals).find(item => item.approval_id === node.dataset.approval);
        const dialog = $("#modify-approval-dialog");
        dialog.dataset.approvalId = node.dataset.approval;
        dialog.querySelector('textarea[name="arguments"]').value = JSON.stringify(approval?.arguments || {}, null, 2);
        dialog.showModal();
        return;
      }
      try {
        await jsonFetch(`/v1/tasks/${encodeURIComponent(state.task.task_id)}/approvals/${encodeURIComponent(node.dataset.approval)}`, {
          method: "POST", body: JSON.stringify({decision: node.dataset.decision, comment: ""})
        });
        await loadTask(state.task.task_id);
      } catch (error) { toast(error.message); }
    }));
    $("#promote-memory")?.addEventListener("click", async () => {
      try {
        await jsonFetch("/v1/memory/promote", {method: "POST", body: JSON.stringify({task_id: state.task.task_id})});
        await loadTask(state.task.task_id);
      } catch (error) { toast(error.message); }
    });
    document.querySelectorAll("[data-memory-delete]").forEach(node => node.addEventListener("click", async () => {
      try {
        await jsonFetch(`/v1/memory/${encodeURIComponent(node.dataset.memoryDelete)}`, {method: "DELETE"});
        await loadTask(state.task.task_id);
      } catch (error) { toast(error.message); }
    }));
  }

  function connectEvents(taskId) {
    state.stream?.close();
    state.stream = new EventSource(`/v1/tasks/${encodeURIComponent(taskId)}/events`);
    state.stream.onopen = () => $("#connection").textContent = "Live";
    state.stream.onerror = () => $("#connection").textContent = "Reconnecting";
    state.stream.onmessage = async () => {
      try { state.task = await jsonFetch(`/v1/tasks/${encodeURIComponent(taskId)}`); renderTask(); } catch (_) {}
    };
  }

  document.querySelectorAll(".tab").forEach(node => node.addEventListener("click", () => { state.tab = node.dataset.tab; renderTab(); }));
  $("#refresh-button").addEventListener("click", loadTasks);
  $("#new-task-button").addEventListener("click", () => $("#new-task-dialog").showModal());
  $("#new-task-form").addEventListener("submit", async event => {
    if (event.submitter?.value === "cancel") return;
    event.preventDefault();
    const form = new FormData(event.target);
    const payload = {
      goal: form.get("goal"),
      workspace_path: form.get("workspace_path") || ".",
      source_urls: String(form.get("source_urls") || "").split(",").map(value => value.trim()).filter(Boolean),
      constraints: String(form.get("constraints") || "").split(",").map(value => value.trim()).filter(Boolean)
    };
    try {
      const task = await jsonFetch("/v1/tasks", {method: "POST", body: JSON.stringify(payload)});
      $("#new-task-dialog").close();
      event.target.reset();
      await loadTasks();
      await loadTask(task.task_id);
    } catch (error) { toast(error.message); }
  });
  $("#modify-approval-form").addEventListener("submit", async event => {
    if (event.submitter?.value === "cancel") return;
    event.preventDefault();
    const dialog = $("#modify-approval-dialog");
    try {
      const argumentsValue = JSON.parse(new FormData(event.target).get("arguments"));
      await jsonFetch(`/v1/tasks/${encodeURIComponent(state.task.task_id)}/approvals/${encodeURIComponent(dialog.dataset.approvalId)}`, {
        method: "POST", body: JSON.stringify({decision: "modified", arguments: argumentsValue, comment: ""})
      });
      dialog.close();
      await loadTask(state.task.task_id);
    } catch (error) { toast(error.message); }
  });
  loadTasks();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
