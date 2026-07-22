import { app } from "../../scripts/app.js";

const PROVIDERS = ["kimi", "glm", "zai", "grok", "openai_compatible"];

async function requestJSON(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function element(tag, attrs = {}, text = "") {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "className") node.className = value;
    else node.setAttribute(key, value);
  }
  if (text) node.textContent = text;
  return node;
}

function field(label, input) {
  const wrap = element("label", { className: "pd-cred-field" });
  wrap.append(element("span", {}, label), input);
  return wrap;
}

function providerRows(profiles, onSelect) {
  const list = element("div", { className: "pd-cred-list" });
  for (const [profile, providers] of Object.entries(profiles || {})) {
    for (const [provider, config] of Object.entries(providers || {})) {
      const button = element("button", { type: "button", className: "pd-cred-row" });
      const state = config.configured ? "Configured" : "Missing key";
      button.append(
        element("span", { className: "pd-cred-name" }, `${profile} / ${provider}`),
        element("span", { className: config.configured ? "pd-cred-ok" : "pd-cred-muted" }, state),
      );
      button.onclick = () => onSelect(profile, provider, config);
      list.append(button);
    }
  }
  if (!list.children.length) list.append(element("div", { className: "pd-cred-empty" }, "No provider profiles"));
  return list;
}

async function render(root) {
  root.innerHTML = "";
  root.className = "pd-credentials";
  const title = element("div", { className: "pd-cred-title" }, "Provider Credentials");
  const layout = element("div", { className: "pd-cred-layout" });
  const sidebar = element("section", { className: "pd-cred-sidebar" });
  const editor = element("section", { className: "pd-cred-editor" });
  const status = element("div", { className: "pd-cred-status", role: "status" });

  const profile = element("input", { type: "text", value: "default", autocomplete: "off" });
  const provider = element("input", { type: "text", value: "kimi", list: "pd-provider-list", autocomplete: "off" });
  const datalist = element("datalist", { id: "pd-provider-list" });
  for (const name of PROVIDERS) datalist.append(element("option", { value: name }));
  const apiKey = element("input", { type: "password", placeholder: "Enter a new key", autocomplete: "new-password" });
  const baseUrl = element("input", { type: "url", placeholder: "Provider default" });
  const model = element("input", { type: "text", placeholder: "Provider default", autocomplete: "off" });
  const authStyle = element("input", { type: "text", value: "bearer", list: "pd-auth-list", autocomplete: "off" });
  const authList = element("datalist", { id: "pd-auth-list" });
  for (const value of ["bearer", "x-api-key", "header:X-API-Key", "query:key"]) {
    authList.append(element("option", { value }));
  }
  const headers = element("textarea", { rows: "4", placeholder: '{"Header-Name":"value"}', spellcheck: "false" });

  const setStatus = (message, error = false) => {
    status.textContent = message;
    status.classList.toggle("pd-cred-error", error);
  };
  const select = (profileName, providerName, config = {}) => {
    profile.value = profileName;
    provider.value = providerName;
    apiKey.value = "";
    apiKey.placeholder = config.configured ? "Key is set; type to replace" : "Enter a new key";
    baseUrl.value = config.base_url || "";
    model.value = config.model || "";
    authStyle.value = config.auth_style || "bearer";
    headers.value = "";
    setStatus(config.header_names?.length ? `Static headers set: ${config.header_names.join(", ")}` : "");
  };

  async function refresh() {
    const data = await requestJSON("/prompt_director/credentials");
    sidebar.replaceChildren(
      element("div", { className: "pd-cred-section-title" }, "Profiles"),
      providerRows(data.profiles, select),
    );
  }

  const save = element("button", { type: "button", className: "pd-cred-primary" }, "Save");
  save.onclick = async () => {
    try {
      setStatus("Saving...");
      const body = {
        profile: profile.value.trim(),
        provider: provider.value.trim(),
        api_key: apiKey.value,
        base_url: baseUrl.value.trim(),
        model: model.value.trim(),
        auth_style: authStyle.value.trim() || "bearer",
      };
      if (headers.value.trim()) body.headers = JSON.parse(headers.value);
      await requestJSON("/prompt_director/credentials", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      apiKey.value = "";
      setStatus("Saved");
      await refresh();
    } catch (error) {
      setStatus(error.message || String(error), true);
    }
  };

  const check = element("button", { type: "button" }, "Check");
  check.onclick = async () => {
    try {
      setStatus("Checking...");
      const data = await requestJSON("/prompt_director/credentials/check", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ profile: profile.value.trim(), provider: provider.value.trim() }),
      });
      setStatus(`Configured${data.model ? ` · ${data.model}` : ""}`);
    } catch (error) {
      setStatus(error.message || String(error), true);
    }
  };

  const clear = element("button", { type: "button", className: "pd-cred-danger" }, "Clear");
  clear.onclick = async () => {
    try {
      await requestJSON("/prompt_director/credentials", {
        method: "DELETE",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ profile: profile.value.trim(), provider: provider.value.trim() }),
      });
      select("default", "kimi");
      setStatus("Cleared");
      await refresh();
    } catch (error) {
      setStatus(error.message || String(error), true);
    }
  };

  const actions = element("div", { className: "pd-cred-actions" });
  actions.append(save, check, clear);
  editor.append(
    element("div", { className: "pd-cred-section-title" }, "Profile"),
    field("Profile name", profile),
    field("Provider", provider),
    datalist,
    field("API key", apiKey),
    field("Base URL", baseUrl),
    field("Model", model),
    field("Authentication", authStyle),
    authList,
    field("Static headers", headers),
    actions,
    status,
  );
  layout.append(sidebar, editor);
  root.append(title, layout);
  try {
    await refresh();
  } catch (error) {
    setStatus(error.message || String(error), true);
  }
}

app.registerExtension({
  name: "prompt.director.credentials",
  async setup() {
    app.extensionManager?.registerSidebarTab?.({
      id: "promptDirectorCredentials",
      icon: "pi pi-key",
      title: "AI Credentials",
      type: "custom",
      render,
    });
  },
});

const style = document.createElement("style");
style.textContent = `
.pd-credentials{height:100%;overflow:auto;color:var(--p-text-color,#eee);background:var(--p-content-background,#111318);font:13px/1.4 system-ui,sans-serif}
.pd-cred-title{padding:12px 14px;border-bottom:1px solid var(--p-content-border-color,#30323a);font-size:15px;font-weight:650}
.pd-cred-layout{display:grid;grid-template-columns:minmax(150px,0.8fr) minmax(230px,1.2fr);min-height:calc(100% - 48px)}
.pd-cred-sidebar{padding:12px;border-right:1px solid var(--p-content-border-color,#30323a)}
.pd-cred-editor{padding:12px 14px}.pd-cred-section-title{margin-bottom:9px;font-size:12px;font-weight:650;text-transform:uppercase;color:var(--p-text-muted-color,#999)}
.pd-cred-list{display:grid;gap:4px}.pd-cred-row{display:flex;justify-content:space-between;gap:8px;width:100%;padding:7px 8px;border:1px solid transparent;border-radius:5px;background:transparent;color:inherit;text-align:left;cursor:pointer}
.pd-cred-row:hover{background:var(--p-highlight-background,#232631);border-color:var(--p-content-border-color,#383b45)}.pd-cred-name{overflow:hidden;text-overflow:ellipsis}.pd-cred-ok{color:#72c98b;font-size:11px}.pd-cred-muted,.pd-cred-empty{color:var(--p-text-muted-color,#888);font-size:11px}
.pd-cred-field{display:grid;gap:4px;margin-bottom:10px}.pd-cred-field>span{font-size:12px;color:var(--p-text-muted-color,#aaa)}.pd-cred-field input,.pd-cred-field textarea{box-sizing:border-box;width:100%;border:1px solid var(--p-form-field-border-color,#3c404b);border-radius:5px;background:var(--p-form-field-background,#0d0f14);color:inherit;padding:7px 8px;font:inherit}.pd-cred-field textarea{resize:vertical;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.pd-cred-actions{display:flex;gap:6px}.pd-cred-actions button{border:1px solid var(--p-content-border-color,#3a3d48);border-radius:5px;background:var(--p-button-secondary-background,#252832);color:inherit;padding:6px 11px;cursor:pointer}.pd-cred-actions .pd-cred-primary{background:var(--p-primary-color,#3679d5);border-color:transparent;color:#fff}.pd-cred-actions .pd-cred-danger{margin-left:auto;color:#e68a8a}
.pd-cred-status{min-height:18px;margin-top:9px;color:#72c98b;font-size:12px}.pd-cred-status.pd-cred-error{color:#ef8b8b}
@media(max-width:560px){.pd-cred-layout{grid-template-columns:1fr}.pd-cred-sidebar{border-right:0;border-bottom:1px solid var(--p-content-border-color,#30323a)}}
`;
document.head.append(style);
