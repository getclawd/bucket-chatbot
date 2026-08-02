const tg = window.Telegram?.WebApp;
if (tg) { tg.ready(); tg.expand(); }

// Every request carries Telegram's signed initData; the server verifies it
// against the bot token, so there is no separate login.
const INIT = tg?.initData || "";

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: {
      "X-Telegram-InitData": INIT,
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });
  const data = await res.json().catch(() => ({ error: "bad response" }));
  if (!res.ok) throw new Error(data.error || `http ${res.status}`);
  return data;
}

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const empty = (msg) => el("div", "empty", msg);

function debounce(fn, ms) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

/* ---------------- tabs ---------------- */
document.querySelectorAll("#tabs button").forEach((button) => {
  button.onclick = () => {
    document.querySelectorAll("#tabs button").forEach((b) => b.classList.remove("on"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("on"));
    button.classList.add("on");
    $("#" + button.dataset.tab).classList.add("on");
    if (button.dataset.tab === "people") loadPeople();
  };
});

/* ---------------- memory ---------------- */
async function loadStats() {
  try {
    const data = await api("/api/stats");
    const c = data.counts;
    $("#subtitle").textContent =
      `${c.utterances} things heard · ${c.factoids} facts · ${c.vectors} remembered`;

    const grid = $("#stats");
    grid.replaceChildren();
    [["utterances", "heard"], ["factoids", "facts"], ["pairs", "replies"],
     ["vocabulary", "words"], ["transitions", "n-grams"], ["inventory", "carrying"]]
      .forEach(([key, label]) => {
        const box = el("div", "stat");
        box.append(el("b", null, c[key] ?? 0), el("span", null, label));
        grid.append(box);
      });

    const ob = $("#obsession");
    ob.replaceChildren();
    if (data.obsession) {
      ob.append(
        el("em", null, `most repeated at it — said ${data.obsession.count} times`),
        document.createTextNode(data.obsession.text),
      );
    } else {
      ob.append(el("em", null, "no obsession yet"));
    }
  } catch (err) {
    $("#subtitle").textContent = err.message;
  }
}

async function loadFacts(query = "") {
  const box = $("#facts");
  try {
    const { facts } = await api("/api/facts?q=" + encodeURIComponent(query));
    box.replaceChildren();
    if (!facts.length) return box.append(empty("nothing matches that"));
    facts.forEach((f) => {
      const row = el("div", "row");
      const subject = el("span", "subject", f.subject);
      row.append(subject, document.createTextNode(` ${f.verb} ${f.object}`));
      const meta = el("div", "meta", `taught by ${f.author}`);
      if (f.count > 1) meta.append(el("span", "pill", `heard ${f.count}x`));
      row.append(meta);
      box.append(row);
    });
  } catch (err) {
    box.replaceChildren(empty(err.message));
  }
}

$("#fact-filter").oninput = debounce((e) => loadFacts(e.target.value), 220);

/* ---------------- people ---------------- */
async function loadPeople() {
  const box = $("#people-list");
  $("#person").replaceChildren();
  try {
    const { people } = await api("/api/people");
    box.replaceChildren();
    if (!people.length) return box.append(empty("nobody has talked to it yet"));
    people.forEach((p) => {
      const row = el("div", "row tappable");
      row.append(el("span", "subject", p.name));
      if (p.aliases.length) row.append(el("span", "pill", "aka " + p.aliases.join(", ")));
      row.append(el("div", "meta", `${p.lines} lines · ${p.facts} facts known`));
      row.onclick = () => showPerson(p.canonical);
      box.append(row);
    });
  } catch (err) {
    box.replaceChildren(empty(err.message));
  }
}

async function showPerson(name) {
  const box = $("#person");
  box.replaceChildren(empty("loading…"));
  try {
    const p = await api("/api/about?name=" + encodeURIComponent(name));
    box.replaceChildren();
    const title = el("h2", null, p.name);
    if (p.aliases.length) title.append(el("span", "pill", "aka " + p.aliases.join(", ")));
    box.append(title);

    box.append(el("h2", null, `what it's been told (${p.facts.length})`));
    if (!p.facts.length) box.append(empty("nothing yet"));
    p.facts.forEach((f) => {
      const row = el("div", "row", `${f.verb} ${f.object}`);
      row.append(el("div", "meta", `from ${f.author || "?"}`));
      box.append(row);
    });

    box.append(el("h2", null, `what they say (${p.total} lines)`));
    p.lines.forEach((l) => {
      const row = el("div", "row", l.text);
      if (l.count > 1) row.append(el("div", "meta", `said ${l.count} times`));
      box.append(row);
    });
    box.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    box.replaceChildren(empty(err.message));
  }
}

/* ---------------- recall ---------------- */
const runRecall = debounce(async (query) => {
  const box = $("#recall-out");
  if (!query.trim()) return box.replaceChildren();
  try {
    const data = await api("/api/recall?q=" + encodeURIComponent(query));
    box.replaceChildren();
    [["semantic", "by meaning"], ["lexical", "by shared words"]].forEach(([key, label]) => {
      box.append(el("h2", null, label));
      if (!data[key].length) return box.append(empty("nothing"));
      data[key].forEach((hit) => {
        const row = el("div", "row", hit.text);
        row.append(el("div", "meta", `${hit.score} · ${hit.author}`));
        box.append(row);
      });
    });
  } catch (err) {
    box.replaceChildren(empty(err.message));
  }
}, 320);

$("#recall-input").oninput = (e) => runRecall(e.target.value);

/* ---------------- talk ---------------- */
$("#chat-form").onsubmit = async (event) => {
  event.preventDefault();
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";

  const log = $("#log");
  log.append(el("div", "msg me", text));
  const pending = el("div", "msg it", "…");
  log.append(pending);
  pending.scrollIntoView({ block: "end" });

  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ text, learn: $("#learn-toggle").checked }),
    });
    pending.textContent = data.reply;

    // The point of the chat tab: show why it said that.
    const why = el("div", "why");
    why.append(el("b", null, data.strategy));
    let note = "";
    if (data.polish_rejected) {
      note = `  · polish discarded (invented: ${data.invented.join(", ")})`;
    } else if (data.polished_differs) {
      note = "  · polished";
    }
    why.append(document.createTextNode(note + "\nraw: " + data.raw));
    if (data.candidates.length) {
      const top = data.candidates.slice(0, 3)
        .map((c) => `  ${c.source} ${c.score} — ${c.text.slice(0, 52)}`).join("\n");
      why.append(document.createTextNode("\nrecalled:\n" + top));
    }
    if (data.learned) why.append(document.createTextNode("\n(learned from this)"));
    log.append(why);
    why.scrollIntoView({ behavior: "smooth", block: "end" });
  } catch (err) {
    pending.textContent = err.message;
  }
};

/* ---------------- go ---------------- */
loadStats();
loadFacts();
