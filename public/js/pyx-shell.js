/**
 * Pyx Studio — shared top navigation shell.
 */
(function (global) {
  "use strict";

  var NAV = [
    { id: "studio", label: "Studio", href: "/studio.html" },
    { id: "workspace", label: "Workspace", href: "/pyx-workspace.html" },
    { id: "talk", label: "Talk", href: "/pyx-talk.html" },
    { id: "pyxels", label: "Pyxels", href: "/pyx-pyxels.html" },
    { id: "code", label: "Code", href: "/pyx-code.html" },
    { id: "write", label: "Write", href: "/pyx-write.html" },
    { id: "pyxel", label: "Pyxel", href: "/pyxel-image.html" },
    { id: "speak", label: "Speak", href: "/pyx-speak.html" },
    { id: "gallery", label: "Gallery", href: "/pyx-gallery.html" },
    { id: "downloads", label: "Downloads", href: "/pyx-download.html" },
    { id: "betas", label: "Betas", href: "/betas" },
  ];

  var STYLE_ID = "pyx-shell-styles";

  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    var s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent =
      ".pyx-studio-bar{position:sticky;top:0;z-index:9000;display:flex;align-items:center;gap:10px;padding:8px 12px;margin:0 0 12px;border:1px solid rgba(129,140,248,.35);border-radius:12px;background:rgba(15,23,42,.88);backdrop-filter:blur(10px);font-family:'Plus Jakarta Sans',ui-sans-serif,system-ui,sans-serif}" +
      ".pyx-studio-bar__brand{display:inline-flex;align-items:center;gap:8px;text-decoration:none;color:inherit;margin-right:4px}" +
      ".pyx-studio-bar__brand img{width:28px;height:28px;border-radius:7px}" +
      ".pyx-studio-bar__brand span{font-weight:800;font-size:.9rem;letter-spacing:-.02em;background:linear-gradient(135deg,#a5b4fc,#38bdf8);-webkit-background-clip:text;background-clip:text;color:transparent}" +
      ".pyx-studio-bar__nav{display:flex;flex-wrap:wrap;gap:4px;flex:1}" +
      ".pyx-studio-bar__link{padding:6px 10px;border-radius:999px;font-size:.78rem;font-weight:700;text-decoration:none;color:#94a3b8;border:1px solid transparent}" +
      ".pyx-studio-bar__link:hover{color:#e2e8f0;border-color:rgba(129,140,248,.35)}" +
      ".pyx-studio-bar__link.is-active{color:#e0e7ff;background:rgba(99,102,241,.25);border-color:rgba(129,140,248,.45)}" +
      ".pyx-studio-bar__menu{display:none;margin-left:auto;padding:6px 10px;border-radius:8px;border:1px solid rgba(148,163,184,.3);background:rgba(30,41,59,.6);color:#cbd5e1;font-weight:700;cursor:pointer}" +
      "@media(max-width:760px){.pyx-studio-bar__nav{display:none}.pyx-studio-bar__nav.is-open{display:flex;position:absolute;left:12px;right:12px;top:52px;flex-direction:column;padding:10px;background:rgba(15,23,42,.97);border:1px solid rgba(129,140,248,.4);border-radius:12px}.pyx-studio-bar__menu{display:inline-flex}}" +
      ".pyx-studio-footer{margin:28px auto 16px;padding:14px 0 0;max-width:72rem;font-size:.8rem;color:#94a3b8;text-align:center;border-top:1px solid rgba(129,140,248,.22)}" +
      ".pyx-studio-footer a{color:#a5b4fc;text-decoration:none}.pyx-studio-footer a:hover{text-decoration:underline}" +
      ".pyx-studio-footer .sep{margin:0 6px;opacity:.5}";
    document.head.appendChild(s);
  }

  function init(opts) {
    opts = opts || {};
    var active = opts.active || "studio";
    var mount =
      document.querySelector("[data-pyx-shell]") ||
      document.getElementById("pyxShellMount");
    if (!mount) {
      mount = document.createElement("div");
      mount.setAttribute("data-pyx-shell", "");
      var body = document.body;
      if (body.firstChild) body.insertBefore(mount, body.firstChild);
      else body.appendChild(mount);
    }
    injectStyles();

    var bar = document.createElement("header");
    bar.className = "pyx-studio-bar";
    bar.setAttribute("role", "navigation");
    bar.setAttribute("aria-label", "Pyx Studio");

    var brand = document.createElement("a");
    brand.className = "pyx-studio-bar__brand";
    brand.href = "/studio.html";
    brand.title = "Pyx Studio";
    brand.innerHTML =
      '<img src="/brand/pyx-app-icon.png" alt="" width="28" height="28" /><span>Pyx Studio</span>';

    var nav = document.createElement("nav");
    nav.className = "pyx-studio-bar__nav";
    NAV.forEach(function (item) {
      var a = document.createElement("a");
      a.className = "pyx-studio-bar__link" + (item.id === active ? " is-active" : "");
      a.href = item.href;
      a.textContent = item.label;
      nav.appendChild(a);
    });

    var menuBtn = document.createElement("button");
    menuBtn.type = "button";
    menuBtn.className = "pyx-studio-bar__menu";
    menuBtn.textContent = "Menu";
    menuBtn.setAttribute("aria-expanded", "false");
    menuBtn.addEventListener("click", function () {
      var open = nav.classList.toggle("is-open");
      menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    });

    bar.appendChild(brand);
    bar.appendChild(nav);
    bar.appendChild(menuBtn);
    mount.innerHTML = "";
    mount.appendChild(bar);

    if (
      global.PyxHandoff &&
      active !== "studio" &&
      active !== "downloads" &&
      active !== "gallery" &&
      active !== "workspace" &&
      active !== "pyxels"
    ) {
      global.PyxHandoff.touchRecent(active, active);
    }

    if (!document.getElementById("pyxStudioFooter")) {
      var footer = document.createElement("footer");
      footer.id = "pyxStudioFooter";
      footer.className = "pyx-studio-footer wrap pyx-wrap";
      footer.innerHTML =
        '<a href="/studio.html">Pyx Studio</a><span class="sep">·</span>' +
        '<a href="/mainlineintelligence">Mainline Intelligence</a><span class="sep">·</span>' +
        '<a href="/pyx-workspace.html">Workspace</a><span class="sep">·</span>' +
        '<a href="/workforpyx.php">Work with Pyx</a><span class="sep">·</span>' +
        '<a href="/pyx-trainer-auth.html">Trainer / Dev Workshop</a>';
      document.body.appendChild(footer);
    }
  }

  global.PyxShell = { init: init, NAV: NAV };
})(typeof window !== "undefined" ? window : globalThis);
