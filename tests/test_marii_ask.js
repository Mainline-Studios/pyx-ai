#!/usr/bin/env node
"use strict";

var ask = require("../public/js/marii-ask.js");
var failed = 0;

function assert(cond, msg) {
  if (!cond) {
    failed += 1;
    console.error("FAIL  " + msg);
  } else {
    console.log("ok    " + msg);
  }
}

var echo = ask.ask("hello from the try-it");
assert(echo.ok === true, "echo ok");
assert(echo.request === "hello from the try-it", "echo request");
assert(echo.reply === "hello from the try-it", "echo reply");
assert(echo.ai === false, "no cloud ai");

var faq = ask.ask("what is marii");
assert(/Mainline Artificial Realtime Instant Intelligence/.test(faq.reply), "faq marii");

var missing = ask.ask("   ");
assert(missing.ok === false, "blank is an error");

var fromPath = ask.requestFromLocation({
  pathname: "/api/marii/ask/what%20is%20marii",
  search: "",
});
assert(fromPath === "what is marii", "path decode");

if (failed) {
  console.error("\n" + failed + " failed");
  process.exit(1);
}
console.log("\nAll marii-ask js tests passed.");
