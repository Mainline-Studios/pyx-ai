#!/usr/bin/env node
"use strict";

const path = require("path");
const { spawn } = require("child_process");
const crypto = require("crypto");

const fs = require("fs");
const root = path.join(__dirname, "..");
const key = crypto.randomBytes(24).toString("hex");

const venvUnix = path.join(root, ".venv", "bin", "python3");
const venvWin = path.join(root, ".venv", "Scripts", "python.exe");
let py = process.platform === "win32" ? "python" : "python3";
if (fs.existsSync(venvUnix)) py = venvUnix;
else if (fs.existsSync(venvWin)) py = venvWin;

console.log("\n  Pyx API (dev)\n");
console.log("  API key (use in your game / client):");
console.log("  ", key);
console.log("");
console.log("  Server: http://localhost:8765");
console.log("  Health: http://localhost:8765/health");
console.log("  (Ctrl+C to stop)\n");

const env = { ...process.env, PYX_API_KEY: key, PORT: "8765" };
const child = spawn(py, ["app.py"], {
  env,
  stdio: "inherit",
  cwd: root,
});

child.on("error", (err) => {
  console.error("Failed to start Python:", err.message);
  console.error("Make sure python3 and Flask are installed (pip install flask).");
  process.exit(1);
});

child.on("exit", (code) => {
  process.exit(code ?? 0);
});
